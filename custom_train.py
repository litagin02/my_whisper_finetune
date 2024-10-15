import math
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Union

import evaluate
import torch
from accelerate import Accelerator, DataLoaderConfiguration
from datasets import IterableDatasetDict, disable_caching, load_dataset
from huggingface_hub import HfApi
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import (
    WhisperForConditionalGeneration,
    WhisperProcessor,
    WhisperTokenizer,
    get_constant_schedule_with_warmup,
)

from utils import log_format, logger

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.benchmark = True

disable_caching()

logger.add("logs/custom_train.log", format=log_format, level="INFO", rotation="10 MB")

ORIGINAL_REPO_ID = "openai/whisper-large-v3-turbo"
SAVE_DIR = "checkpoints/galgame-whisper-large-v3-turbo-clip-grad-norm"

NUM_EPOCHS = 10
LEARNING_RATE = 1e-5
BATCH_SIZE = 16
WEIGHT_DECAY = 0.0
NUM_TEST_SAMPLES = 128
NUM_WARMUP_STEPS = 5000

# Galgame_Speech_ASR_16kHzの全部のtarファイル数
# 0-indexedで、000から114までの115個
NUM_TAR_FILES = 115
# 0 - 102まで（90.1%）をtrainとして使うとして、そのtarファイルのデータ数
NUM_TRAIN_SAMPLES = 3_374_929
# 103 - 114までをtestとして使う
TEST_START_INDEX = 103

# https://huggingface.co/datasets/litagin/Galgame_Speech_ASR_16kHz
HF_PATH_TO_DATASET = "litagin/Galgame_Speech_ASR_16kHz"

REPO_ID = "litagin/galgame-whisper-large-v3-turbo-clip-grad-norm-logs2"
api = HfApi()

disable_caching()


def get_latest_checkpoint_path() -> str | None:
    # SAVE_DIR/checkpoint-<step>
    checkpoint_dirs = [d for d in Path(SAVE_DIR).rglob("checkpoint-*") if d.is_dir()]
    if not checkpoint_dirs:
        return None
    latest_checkpoint_dir = max(
        checkpoint_dirs, key=lambda d: int(d.name.split("-")[1])
    )
    return str(latest_checkpoint_dir)


def clean_checkpoint(num_to_keep: int = 10) -> None:
    checkpoint_dirs = [d for d in Path(SAVE_DIR).rglob("checkpoint-*") if d.is_dir()]
    checkpoint_dirs = sorted(
        checkpoint_dirs, key=lambda d: int(d.name.split("-")[1]), reverse=True
    )
    for checkpoint_dir in checkpoint_dirs[num_to_keep:]:
        logger.warning(f"Removing checkpoint {checkpoint_dir}")
        shutil.rmtree(checkpoint_dir)


@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    processor: Any

    def __call__(
        self, features: List[Dict[str, Union[List[int], torch.Tensor]]]
    ) -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lengths and need different padding methods
        # first treat the audio inputs by simply returning torch tensors
        input_features = [
            {"input_features": feature["input_features"][0]} for feature in features
        ]
        batch = self.processor.feature_extractor.pad(
            input_features, return_tensors="pt"
        )

        # get the tokenized label sequences
        label_features = [{"input_ids": feature["labels"]} for feature in features]
        # pad the labels to max length
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        # replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1), -100
        )

        # if bos token is appended in previous tokenization step,
        # cut bos token here as it's append later anyways
        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels

        return batch


def prepare_dataset_wrapper(processor):
    def prepare_dataset(example):
        audio = example["audio"]
        example = processor(
            audio=audio["array"],
            sampling_rate=audio["sampling_rate"],
            text=example["sentence"],
        )
        return example

    return prepare_dataset


# `cer`の他に`normalized cer`: 予測と正解を正規化してからCERも計算する
# 正規化は、簡単のため「マッチするもの以外を削除」するだけ
# 学習データは、句読点類に加えて以下のみからなる（英字数字は半角に変換済み）ので、
# それ以外を削除することで正規化を行う

# 許容するもの「以外」にマッチするパターン
INVALID_PATTERN = re.compile(
    r"[^\u3040-\u309F\u30A0-\u30FF\u4E00-\u9FFF\u3400-\u4DBF\u3005"  # ひらがな、カタカナ、漢字
    r"\u0041-\u005A\u0061-\u007A"  # 半角英字
    r"\u0030-\u0039]"  # 半角数字
)


def normalizer(text):
    text = INVALID_PATTERN.sub("", text)
    if text == "":
        # 正規化の結果、空文字列になった場合は「…」に置き換える
        # 正解ラベルに空文字列があるとエラーになるのでこうする
        text = "…"
    return text


def compute_metrics(pred_ids, label_ids, tokenizer):
    metric = evaluate.load("cer")
    # pred_ids = pred.predictions
    # label_ids = pred.label_ids

    # replace -100 with the pad_token_id
    label_ids[label_ids == -100] = tokenizer.pad_token_id

    # we do not want to group tokens when computing the metrics
    pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=True)

    normalized_pred_str = [normalizer(text) for text in pred_str]
    normalized_label_str = [normalizer(text) for text in label_str]

    cer = 100 * metric.compute(predictions=pred_str, references=label_str)
    normalized_cer = 100 * metric.compute(
        predictions=normalized_pred_str, references=normalized_label_str
    )
    return {"cer": cer, "normalized_cer": normalized_cer}


def _load_dataset(
    *,
    streaming: bool = True,
    use_local_dataset: bool = True,
    local_dataset_path: str | None = None,
) -> IterableDatasetDict:
    data_files = {
        "train": [
            f"galgame-speech-asr-16kHz-train-000{index:03d}.tar"
            for index in range(0, TEST_START_INDEX)
        ],
        "test": [
            f"galgame-speech-asr-16kHz-train-000{index:03d}.tar"
            for index in range(TEST_START_INDEX, NUM_TAR_FILES)
        ],
    }
    if use_local_dataset:
        assert local_dataset_path is not None
        path = local_dataset_path
    else:
        path = HF_PATH_TO_DATASET
    dataset: IterableDatasetDict = load_dataset(
        path=path, data_dir="data", data_files=data_files, streaming=streaming
    )  # type: ignore
    dataset = dataset.remove_columns(["__key__", "__url__"])
    dataset = dataset.rename_column("ogg", "audio")
    dataset = dataset.rename_column("txt", "sentence")

    return dataset


if __name__ == "__main__":
    logger.info("Preparing model...")
    processor = WhisperProcessor.from_pretrained(ORIGINAL_REPO_ID)
    tokenizer = WhisperTokenizer.from_pretrained(ORIGINAL_REPO_ID)
    model: WhisperForConditionalGeneration = (
        WhisperForConditionalGeneration.from_pretrained(ORIGINAL_REPO_ID)
    )  # type: ignore
    logger.success("Model prepared.")

    # dataset = load_dataset(
    #     path="C:/AI/galgame_dataset/Galgame_Speech_ASR_16kHz",
    #     data_dir="data",
    #     streaming=True,
    # )
    dataset = _load_dataset(
        streaming=True,
        use_local_dataset=True,
        local_dataset_path="C:/AI/galgame_dataset/Galgame_Speech_ASR_16kHz",
    )
    original_column_names = ["audio", "sentence"]
    prepare_dataset = prepare_dataset_wrapper(processor)
    dataset = dataset.map(
        prepare_dataset, remove_columns=original_column_names
    ).with_format("torch")

    data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)

    train_loader = DataLoader(
        dataset["train"],
        collate_fn=data_collator,
        batch_size=16,
        shuffle=False,
        pin_memory=True,
    )
    test_loader = DataLoader(
        dataset["test"].take(NUM_TEST_SAMPLES),
        collate_fn=data_collator,
        batch_size=16,
        shuffle=False,
    )

    dataloader_config = DataLoaderConfiguration(use_stateful_dataloader=True)

    accelerator = Accelerator(
        mixed_precision="fp16",
        dataloader_config=dataloader_config,
        log_with="tensorboard",
        project_dir=SAVE_DIR,
    )
    accelerator.init_trackers("my-project")
    logger.warning("Freezing encoder...")
    model.freeze_encoder()
    # model.gradient_checkpointing_enable()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    scheduler = get_constant_schedule_with_warmup(
        optimizer, num_warmup_steps=NUM_WARMUP_STEPS
    )
    model, optimizer, train_loader, scheduler = accelerator.prepare(
        model, optimizer, train_loader, scheduler
    )
    # test_loader = accelerator.prepare(test_loader)
    num_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Number of trainable parameters: {num_trainable_params}")

    latest_checkpoint_path = get_latest_checkpoint_path()
    if latest_checkpoint_path is not None:
        logger.info(f"Loading checkpoint from {latest_checkpoint_path}")
        accelerator.load_state(latest_checkpoint_path)
        current_step = int(latest_checkpoint_path.split("-")[-1])
    else:
        logger.info("No checkpoint found, starting from the beginning.")
        current_step = 0

    steps_per_epoch = math.ceil(NUM_TRAIN_SAMPLES / BATCH_SIZE)
    current_residue_step = current_step % steps_per_epoch
    for epoch in range(NUM_EPOCHS):
        logger.info(f"***** Epoch {epoch} *****")
        for batch_idx, batch in enumerate(
            tqdm(train_loader, total=steps_per_epoch, initial=current_residue_step)
        ):
            model.train()
            optimizer.zero_grad()
            output = model(**batch)
            loss = output.loss
            # loss.backward()
            accelerator.backward(loss)
            grad_norm = accelerator.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            current_step += 1
            if current_step % 25 != 0:
                continue
            log_dict = {
                "train/loss": loss.item(),
                "train/step": current_step,
                "train/epoch": current_step / steps_per_epoch,
                "train/lr": scheduler.get_last_lr()[0],
                "train/grad_norm": grad_norm,
            }
            accelerator.log(log_dict, step=current_step)
            tqdm.write(
                f"Step {current_step} | Loss: {loss.item():.4f} | LR: {scheduler.get_last_lr()[0]:.2e} | Grad Norm: {grad_norm:.4f}"
            )
            if current_step % 1000 != 0:
                continue
            logger.info("Saving checkpoint...")
            checkpoint_path = f"{SAVE_DIR}/checkpoint-{current_step}"
            accelerator.save_state(checkpoint_path)
            clean_checkpoint()
            logger.success(f"Checkpoint saved at {checkpoint_path}")
            logger.info("Starting uploading checkpoint...")
            api.create_repo(
                repo_id=REPO_ID,
                exist_ok=True,
                private=True,
            )
            api.upload_folder(
                repo_id=REPO_ID,
                folder_path=checkpoint_path,
                path_in_repo=f"checkpoints/{current_step}",
                run_as_future=True,
            )

            logger.info("Evaluating...")
            model.eval()
            # model.to(accelerator.device, dtype=torch.float16)
            cer_sum = 0.0
            normalized_cer_sum = 0.0
            eval_count = 0
            for test_batch in test_loader:
                eval_count += 1
                test_batch = {k: v.to("cuda") for k, v in test_batch.items()}
                with torch.no_grad():
                    output = model.generate(
                        input_features=test_batch["input_features"],
                        language="Japanese",
                        max_length=64,
                    )
                metrics = compute_metrics(
                    pred_ids=output,
                    label_ids=test_batch["labels"],
                    tokenizer=tokenizer,
                )
                cer_sum += metrics["cer"]
                normalized_cer_sum += metrics["normalized_cer"]
            log_dict = {
                "test/cer": cer_sum / eval_count,
                "test/normalized_cer": normalized_cer_sum / eval_count,
            }
            accelerator.log(log_dict, step=current_step)
            del output, metrics, log_dict, test_batch
            # torch.cuda.empty_cache()
            # gc.collect()
            logger.success("Evaluation done.")
