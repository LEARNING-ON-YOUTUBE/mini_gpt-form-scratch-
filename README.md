# mini_gpt-form-scratch-

A small character-level GPT trained with PyTorch on the Project Gutenberg text
of *War and Peace*. The project includes the training code, a trained checkpoint,
an interactive text-generation script, and the notebook used to explore the
dataset.

## Project structure

```text
mini_gpt/
├── README.md
├── requirements.txt
├── gptdataunderstaing.ipynb
├── howtomake.txt
└── mini_gpt/
    ├── book.txt
    ├── train.py
    ├── model.py
    ├── tokenizer.py
    ├── config.py
    ├── genrate.py
    ├── checkpoint/
    │   └── best_model.pt
    └── output/  (created locally when generating text)
```

The duplicate dataset copy and Python cache files are not included. The
training script reads `mini_gpt/book.txt` relative to its own location.

## Setup

Use Python 3.10 or newer. Create and activate a virtual environment, then
install the project dependency:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

For NVIDIA GPU training, install the CUDA-enabled PyTorch build recommended by
the [official PyTorch selector](https://pytorch.org/get-started/locally/).
The training script uses CUDA when available and otherwise falls back to CPU.

## Train

From the project root:

```powershell
python .\mini_gpt\train.py
```

Training runs for 10,000 steps by default. The best validation checkpoint is
saved to `mini_gpt/checkpoint/best_model.pt`; a sample is written to
`mini_gpt/output/generated.txt`.

## Generate text interactively

From the project root:

```powershell
python -m mini_gpt.genrate
```

Enter prompts repeatedly. Each generated continuation is printed and appended
to `mini_gpt/output/generated.txt`. Generated prompts and outputs stay local
and are excluded from Git. Enter `q`, `quit`, or `exit` to stop.

## Dataset

The included `book.txt` is the text dataset used by the training script.
Project Gutenberg provides ebook information and licensing details at
https://www.gutenberg.org/ebooks/79701.
