import argparse
import inspect
import logging
import math
import re
import sys
from dataclasses import dataclass
from functools import partial
from typing import Any, Dict, List, Union

import evaluate
import torch
import transformers
from datasets import IterableDatasetDict, disable_caching, load_dataset
from loguru import logger
from transformers import (
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    WhisperFeatureExtractor,
    WhisperForConditionalGeneration,
    WhisperProcessor,
    WhisperTokenizer,
)

# Galgame_Speech_ASR_16kHzの全部のtarファイル数
# 0-indexedで、000から114までの115個
NUM_TAR_FILES = 115
# 0 - 102まで（90.1%）をtrainとして使うとして、そのtarファイルのデータ数
NUM_TRAIN_SAMPLES = 3_374_929
# 103 - 114までをtestとして使う
TEST_START_INDEX = 103
# Hugging Face Datasetsのrepo_id
# https://huggingface.co/datasets/litagin/Galgame_Speech_ASR_16kHz
HF_PATH_TO_DATASET = "litagin/Galgame_Speech_ASR_16kHz"

disable_caching()


# transformersのloggerをloguruへ送るための設定
class InterceptHandler(logging.Handler):
    def emit(self, record: logging.LogRecord) -> None:
        # Get corresponding Loguru level if it exists.
        level: str | int
        try:
            level = logger.level(record.levelname).name
        except ValueError:
            level = record.levelno

        # Find caller from where originated the logged message.
        frame, depth = inspect.currentframe(), 0
        while frame and (depth == 0 or frame.f_code.co_filename == logging.__file__):
            frame = frame.f_back
            depth += 1

        logger.opt(depth=depth, exception=record.exc_info).log(
            level, record.getMessage()
        )


log_format = (
    "<green>{time:YYYY-MM-DD HH:mm:ss.S}</green> | "
    "<level>{level:^8}</level> | "
    "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | <level>{message}</level>"
)
# loguru に出力フォーマットを設定
logger.remove()  # デフォルトのハンドラを削除
logger.add(sys.stdout, format=log_format, level="INFO")
logger.add(
    "logs/large-v3-turbo.log",
    format=log_format,
    level="INFO",
    rotation="00:00",
)

logging.basicConfig(handlers=[InterceptHandler()], level=0, force=True)

# transformers.logging.set_verbosity_info()
transformers.logging.disable_default_handler()
transformers.logging.add_handler(InterceptHandler())


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


metric = evaluate.load("cer")

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


def normalizer(text, add_ellipsis=False):
    text = INVALID_PATTERN.sub("", text)
    if text == "" and add_ellipsis:
        # 正規化の結果、空文字列になった場合は「…」に置き換える
        # 正解ラベルに空文字列があるとエラーになるのでこうする
        text = "…"
    return text


def compute_metrics(pred):
    pred_ids = pred.predictions
    label_ids = pred.label_ids

    # replace -100 with the pad_token_id
    label_ids[label_ids == -100] = tokenizer.pad_token_id

    # we do not want to group tokens when computing the metrics
    pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
    label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=True)

    normalized_pred_str = [normalizer(text) for text in pred_str]
    normalized_label_str = [normalizer(text, add_ellipsis=True) for text in label_str]

    cer = 100 * metric.compute(predictions=pred_str, references=label_str)
    normalized_cer = 100 * metric.compute(
        predictions=normalized_pred_str, references=normalized_label_str
    )
    return {"cer": cer, "normalized_cer": normalized_cer}


def _load_dataset(
    *,
    streaming: bool = True,
    use_local_dataset: bool = False,
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
        path = local_dataset_path
    else:
        path = HF_PATH_TO_DATASET
    dataset: IterableDatasetDict = load_dataset(
        path=path, data_dir="data", data_files=data_files, streaming=streaming
    )

    dataset = dataset.remove_columns(["__key__", "__url__"])
    dataset = dataset.rename_column("ogg", "audio")
    dataset = dataset.rename_column("txt", "sentence")

    return dataset


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--original_repo_id", type=str, default="openai/whisper-large-v3-turbo"
    )
    parser.add_argument("--repo_id", type=str, default="galgame-whisper-large-v3-turbo")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--no_freeze_encoder", action="store_true")
    parser.add_argument("--use_local_dataset", "-l", action="store_true")
    parser.add_argument(
        "--local_dataset_path",
        type=str,
        # default="C:/AI/galgame_dataset/Galgame_Speech_ASR_16kHz",
        default=None,
    )
    parser.add_argument("--num_eval_step", type=int, default=25)
    parser.add_argument("--push_to_hub", action="store_true")
    parser.add_argument(
        "--hub_model_id",
        type=str,
        default=None,
        # default="litagin/galgame-whisper-large-v3-turbo-logs",
    )

    args = parser.parse_args()
    logger.info(f"Arguments: {args}")
    original_repo_id = str(args.original_repo_id)
    repo_id = str(args.repo_id)
    batch_size = int(args.batch_size)
    no_freeze_encoder = bool(args.no_freeze_encoder)
    use_local_dataset = bool(args.use_local_dataset)
    local_dataset_path = str(args.local_dataset_path)
    eval_size = int(args.num_eval_step) * batch_size
    push_to_hub = bool(args.push_to_hub)
    hub_model_id = str(args.hub_model_id)

    logger.info(f"Preparing the preprocessor for {original_repo_id}...")

    feature_extractor = WhisperFeatureExtractor.from_pretrained(original_repo_id)
    tokenizer = WhisperTokenizer.from_pretrained(
        original_repo_id, language="Japanese", task="transcribe"
    )
    processor: WhisperProcessor = WhisperProcessor.from_pretrained(
        original_repo_id, language="Japanese", task="transcribe"
    )  # type: ignore

    logger.info("Loading the pretrained model...")
    model: WhisperForConditionalGeneration = (
        WhisperForConditionalGeneration.from_pretrained(original_repo_id)
    )  # type: ignore

    if not no_freeze_encoder:
        logger.warning("Freezing the encoder...")
        model.freeze_encoder()

    data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor=processor)

    dataset = _load_dataset(
        streaming=True,
        use_local_dataset=use_local_dataset,
        local_dataset_path=local_dataset_path if use_local_dataset else None,
    )

    # disable cache during training since it's incompatible with gradient checkpointing
    model.config.use_cache = False

    # set language and task for generation and re-enable cache
    model.generate = partial(
        model.generate, language="Japanese", task="transcribe", use_cache=True
    )

    # ハイパラはお好きに変更してください
    training_args = Seq2SeqTrainingArguments(
        output_dir=repo_id,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=1,  # increase by 2x for every 2x decrease in batch size
        learning_rate=1e-5,
        lr_scheduler_type="constant_with_warmup",
        warmup_steps=5_000,
        max_steps=math.ceil(NUM_TRAIN_SAMPLES / batch_size),
        # num_train_epochs=1,
        gradient_checkpointing=True,
        fp16=True,
        fp16_full_eval=True,
        eval_strategy="steps",
        per_device_eval_batch_size=batch_size,
        predict_with_generate=True,
        generation_max_length=64,
        save_steps=1000,
        eval_steps=1000,
        logging_steps=25,
        report_to=["tensorboard"],
        # load_best_model_at_end=True,
        metric_for_best_model="normalized_cer",
        greater_is_better=False,
        push_to_hub=push_to_hub,
        hub_private_repo=True,
        dataloader_num_workers=0,
        dataloader_persistent_workers=False,
        dataloader_pin_memory=False,
        dataloader_prefetch_factor=None,
        save_total_limit=10,
        hub_model_id=hub_model_id,
        hub_strategy="all_checkpoints",
        eval_on_start=True,
    )

    original_column_names = list(next(iter(dataset.values())).features.keys())
    prepare_dataset = prepare_dataset_wrapper(processor)
    dataset = dataset.map(prepare_dataset, remove_columns=original_column_names)

    logger.info("Dataset prepared.")
    trainer = Seq2SeqTrainer(
        args=training_args,
        model=model,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"].take(eval_size),
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        tokenizer=processor,
    )

    logger.info(f"Starting training for {repo_id}...")
    try:
        # Resumeしようとする
        trainer.train(resume_from_checkpoint=True)
    except ValueError:
        # Resumeできなかったら、最初から
        trainer.train()
    except Exception as e:
        logger.exception(e)
        raise e

    logger.info(f"Finished 1 epoch training for {repo_id}.")
