# My code for fine-tuning Whisper on Galgame Dataset

- My code for fine-tuning [Whisper model in 🤗](https://huggingface.co/models?other=whisper) with [Galgame_Speech_ASR_16kHz](https://huggingface.co/datasets/litagin/Galgame_Speech_ASR_16kHz) using [🤗 Transformers](https://huggingface.co/docs/transformers/en/index)

- Default: use the base model `openai/whisper-large-v3-turbo` and fine-tune it on the Galgame dataset 1 epoch.

- This uses [Dataset Streaming in 🤗 Datasets](https://huggingface.co/docs/datasets/en/stream) by default so you don't have to (and the script will not) download all the 100GB data and the training data will be downloaded in a stream, and maybe only small amout of the disk is needed

- (Or you can whole download the dataset and use it locally)

- I may not maintain this repo, and the code is subject to change. I may not have fully checked the code, so something may be wrong. Just use it at your own risk.


## Install

- Install [uv](https://docs.astral.sh/uv/) (or you can use the usual venv and pip)

Windows:
```bash
uv venv
.venv\Scripts\activate
uv pip install torch --index-url https://download.pytorch.org/whl/cu124
uv sync
```

## Train

**See code by yourself for how to train and modify the code by yourself before training!**

- 2024-10-15: Add [custom_train.py](custom_train.py), which uses torchdata for data loading, which is faster when resuming the training!
  - But this doesn't use 🤗 Transformers library and I wrote a custom train loop, so something may be wrong
  - Also VRAM Usage is higher than the original (so by default batch size 16), and I don't know why
   (maybe 🤗 Trainer does a lot of optimization and I don't know how to do it, please tell me!)
  - Also the code doesn't use argparse, so you have to modify the code by yourself

- [train.py](train.py)
  - The script is subject to change, and something may be wrong (Not fully checked since I haven't trained 1 ecpoch!).

  - Basically the following command fine-tunes the model on the Galgame dataset **1 epoch** (some data are not used and will be used for eval).

  - Save and eval every 1000 steps, and eval on the very beginning.

  - Maybe you can resume the training after the training is interrupted (**but extrimely slow since we have to skip from the beginning of the streaming dataset**).

## Reference

- [Fine-tuning the ASR model](https://huggingface.co/learn/audio-course/en/chapter5/fine-tuning)
- [Whisper Fine-Tuning Event 🤗](https://github.com/huggingface/community-events/tree/main/whisper-fine-tuning-event)
- [Galgame_Speech_ASR_16kHz](https://huggingface.co/datasets/litagin/Galgame_Speech_ASR_16kHz)
- [Galgame_Dataset](https://huggingface.co/datasets/OOPPEENN/Galgame_Dataset)