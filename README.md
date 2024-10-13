# My code for fine-tuning Whisper on Galgame Dataset

- [Galgame_Speech_ASR_16kHz](https://huggingface.co/datasets/litagin/Galgame_Speech_ASR_16kHz)


## Install

Windows:
```bash
uv venv
.venv\Scripts\activate
uv pip install torch --index-url https://download.pytorch.org/whl/cu124
uv sync
```

## Train

```bash
uv run train.py [--original_repo_id <original_repo_id>] [--repo_id <repo_id>] [--no_freeze_encoder] [--batch_size <batch_size>] [--use_local_dataset] [--local_dataset_path <local_dataset_path>] [--num_eval_steps <num_eval_steps>] [--push_to_hub] [--hub_model_id <hub_model_id>]
```

- `--original_repo_id`: The base model to fine-tune. Default: `openai/whisper-large-v3-turbo`.
- `--repo_id`: The model name of the fine-tuned model. Default: `galgame-whisper-large-v3-turbo`.
- `--no_freeze_encoder`: Flag to NOT freeze the encoder. Default: `False`, so by default the encoder is frozen.
- `--batch_size`: The batch size. Default: `32`.
- `--use_local_dataset`: Flag to use local dataset. Default: `False`. If you have already downloaded the entire dataset (in tar format), you can use this flag to use the local dataset.
- `--local_dataset_path`: The path to the local dataset in case you use the local dataset.
- `--num_eval_steps`: The number of samples (in terms of steps, maybe) to evaluate the model. Default: `25`, and the number of samples is `batch_size * num_eval_steps`.
- `--push_to_hub` (store_true): Flag to push (backup) all the checkpoints to a private repository of 🤗. Default: `False`. Maybe you should have logged in to use it.
- `--hub_model_id`: The repo_id to push the checkpoints.

## Reference

- [Fine-tuning the ASR model](https://huggingface.co/learn/audio-course/en/chapter5/fine-tuning)
- [Whisper Fine-Tuning Event 🤗](https://github.com/huggingface/community-events/tree/main/whisper-fine-tuning-event)
- [Galgame_Speech_ASR_16kHz](https://huggingface.co/datasets/litagin/Galgame_Speech_ASR_16kHz)
- [Galgame_Dataset](https://huggingface.co/datasets/OOPPEENN/Galgame_Dataset)