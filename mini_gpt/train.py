# ============================================================
# Mini GPT - Training Script
# ============================================================
#
# Dataset:
#     War and Peace - Leo Tolstoy
#
# Tokenization:
#     Character-level tokenizer
#
# Architecture:
#     GPT / Transformer
#
# This file handles:
#     1. Loading dataset
#     2. Building vocabulary
#     3. Character -> integer encoding
#     4. Train/validation split
#     5. Random mini-batch generation
#     6. Model creation
#     7. Loss estimation
#     8. Training
#     9. Checkpoint saving
#    10. Text generation
#
# ============================================================


import os
import math
from pathlib import Path

import torch
import torch.nn as nn


# ============================================================
# 1. CONFIGURATION
# ============================================================

# ------------------------------------------------------------
# Dataset
# ------------------------------------------------------------

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_PATH = SCRIPT_DIR / "book.txt"


# ------------------------------------------------------------
# Training hyperparameters
# ------------------------------------------------------------

batch_size = 64

# Number of previous characters/tokens the GPT can see.
#
# Your notebook first tested block_size = 8
# and later used block_size = 64.
#
# We keep 64 as our starting point.
block_size = 64

# Learning rate.
#
# Your Bigram experiment successfully used:
#
#     1e-2
#
# A Transformer/GPT normally needs a considerably smaller
# learning rate, so we start with:
#
#     3e-4
#
learning_rate = 3e-4

# Total number of optimizer updates.
max_iters = 10000

# How often we evaluate the model.
eval_interval = 500

# Number of random batches used to estimate train/val loss.
eval_iters = 50

# Number of characters to generate after training.
max_new_tokens = 500

# Percentage of data used for training.
train_ratio = 0.90


# ------------------------------------------------------------
# Model hyperparameters
# ------------------------------------------------------------
#
# These values will be used by model.py.
# Keeping them here makes experimentation easier.
# ------------------------------------------------------------

n_embd = 128
n_head = 4
n_layer = 4

dropout = 0.0


# ------------------------------------------------------------
# Device
# ------------------------------------------------------------

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ------------------------------------------------------------
# Random seed
# ------------------------------------------------------------

torch.manual_seed(1337)


# ------------------------------------------------------------
# Checkpoint
# ------------------------------------------------------------

CHECKPOINT_DIR = SCRIPT_DIR / "checkpoint"
CHECKPOINT_DIR.mkdir(exist_ok=True)

CHECKPOINT_PATH = CHECKPOINT_DIR / "best_model.pt"


# ------------------------------------------------------------
# Output directory
# ------------------------------------------------------------

OUTPUT_DIR = SCRIPT_DIR / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

GENERATED_PATH = OUTPUT_DIR / "generated.txt"


# ============================================================
# 2. CHECK DEVICE
# ============================================================

print("=" * 60)
print("MINI GPT TRAINING")
print("=" * 60)

print(f"Device: {device}")

if device.type == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(
        f"CUDA memory: "
        f"{torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB"
    )

print()


# ============================================================
# 3. LOAD DATASET
# ============================================================

if not DATA_PATH.exists():
    raise FileNotFoundError(
        f"\nDataset not found: {DATA_PATH.resolve()}\n"
        f"Please place book.txt in the same folder as train.py."
    )


print("Loading dataset...")

with open(DATA_PATH, "r", encoding="utf-8") as file:
    text = file.read()


print(f"Characters: {len(text):,}")


# ============================================================
# 4. BUILD CHARACTER VOCABULARY
# ============================================================
#
# Example:
#
# text:
#
#     "hello"
#
# characters:
#
#     ['e', 'h', 'l', 'o']
#
# stoi:
#
#     e -> 0
#     h -> 1
#     l -> 2
#     o -> 3
#
# itos:
#
#     0 -> e
#     1 -> h
#     2 -> l
#     3 -> o
#
# ============================================================

chars = sorted(list(set(text)))

vocab_size = len(chars)

print(f"Vocabulary size: {vocab_size}")

print("Characters:")
print(chars)

print()


# ============================================================
# 5. CREATE TOKENIZER
# ============================================================

stoi = {
    character: index
    for index, character in enumerate(chars)
}

itos = {
    index: character
    for index, character in enumerate(chars)
}


# ------------------------------------------------------------
# Encode
# ------------------------------------------------------------

def encode(text_string):
    """
    Convert characters into integer token IDs.

    Example:

        "abc"

    might become:

        [5, 8, 12]
    """

    return [stoi[character] for character in text_string]


# ------------------------------------------------------------
# Decode
# ------------------------------------------------------------

def decode(token_ids):
    """
    Convert integer token IDs back into characters.
    """

    return "".join(itos[token_id] for token_id in token_ids)


# ------------------------------------------------------------
# Test tokenizer
# ------------------------------------------------------------

sample_text = text[:100]

encoded_sample = encode(sample_text)

decoded_sample = decode(encoded_sample)

print("Tokenizer test:")
print("----------------")
print("Original:")
print(repr(sample_text[:80]))

print()

print("Encoded:")
print(encoded_sample[:30])

print()

print("Decoded:")
print(repr(decoded_sample[:80]))

print()


# ============================================================
# 6. CONVERT ENTIRE DATASET TO PYTORCH TENSOR
# ============================================================

print("Converting dataset to PyTorch tensor...")

data = torch.tensor(
    encode(text),
    dtype=torch.long
)

print(f"Tensor shape: {data.shape}")
print(f"Tensor dtype: {data.dtype}")

print()


# ============================================================
# 7. TRAIN / VALIDATION SPLIT
# ============================================================
#
# Your notebook used:
#
#     split_idx = int(0.9 * len(data))
#
# We preserve that approach.
#
# 90% -> training
# 10% -> validation
#
# ============================================================

split_idx = int(train_ratio * len(data))

train_data = data[:split_idx]

val_data = data[split_idx:]


print("Dataset split:")
print("----------------")

print(
    f"Training tokens:   {len(train_data):,}"
)

print(
    f"Validation tokens: {len(val_data):,}"
)

print()


# ============================================================
# 8. RANDOM BATCH GENERATOR
# ============================================================
#
# This is one of the most important pieces of GPT training.
#
#
# Suppose block_size = 8
#
# Input:
#
#     T h e   q u i
#
# Target:
#
#       h e   q u i c
#
#
# In token form:
#
# x = [10, 20, 30, 40, 50, 60, 70, 80]
#
# y = [20, 30, 40, 50, 60, 70, 80, 90]
#
#
# GPT learns:
#
#     x[t] -> x[t+1]
#
# ============================================================


def get_batch(split):
    """
    Generate one random mini-batch.

    Returns:

        x -> input tokens
        y -> next-token targets

    Shapes:

        x: [batch_size, block_size]
        y: [batch_size, block_size]
    """

    if split == "train":
        source = train_data

    elif split == "val":
        source = val_data

    else:
        raise ValueError(
            "split must be either 'train' or 'val'"
        )


    # --------------------------------------------------------
    # Random starting positions
    # --------------------------------------------------------

    starts = torch.randint(
        low=0,
        high=len(source) - block_size,
        size=(batch_size,)
    )


    # --------------------------------------------------------
    # Create input sequences
    # --------------------------------------------------------

    x = torch.stack([
        source[start:start + block_size]
        for start in starts
    ])


    # --------------------------------------------------------
    # Create target sequences
    #
    # Target is shifted by one position.
    # --------------------------------------------------------

    y = torch.stack([
        source[start + 1:start + block_size + 1]
        for start in starts
    ])


    # --------------------------------------------------------
    # Move to GPU / CPU
    # --------------------------------------------------------

    x = x.to(device)
    y = y.to(device)


    return x, y


# ============================================================
# 9. TEST BATCH GENERATION
# ============================================================

xb, yb = get_batch("train")

print("Batch test:")
print("----------------")

print(f"Input shape:  {xb.shape}")
print(f"Target shape: {yb.shape}")

print()

print("Example input IDs:")
print(xb[0])

print()

print("Example target IDs:")
print(yb[0])

print()

print("Example input decoded:")
print(repr(decode(xb[0].tolist())))

print()

print("Example target decoded:")
print(repr(decode(yb[0].tolist())))

print()


# ============================================================
# 10. IMPORT GPT MODEL
# ============================================================
#
# model.py should eventually contain:
#
#     class GPTLanguageModel(nn.Module):
#
#         def forward(self, idx, targets=None):
#             ...
#
#         def generate(self, idx, max_new_tokens):
#             ...
#
# ============================================================

try:
    if __package__:
        from .model import GPTLanguageModel
    else:
        from model import GPTLanguageModel

except ImportError as error:

    print("\nERROR:")
    print("--------------------------------------------")
    print("Could not import GPTLanguageModel from model.py")
    print()
    print("Your model.py should contain:")
    print()
    print("    class GPTLanguageModel(nn.Module):")
    print("        ...")
    print()
    print("We will build model.py next.")
    print("--------------------------------------------")

    raise error


# ============================================================
# 11. CREATE GPT MODEL
# ============================================================

model = GPTLanguageModel(
    vocab_size=vocab_size,
    block_size=block_size,
    n_embd=n_embd,
    n_head=n_head,
    n_layer=n_layer,
    dropout=dropout
)

model = model.to(device)


# ============================================================
# 12. PRINT MODEL INFORMATION
# ============================================================

print("=" * 60)
print("MODEL")
print("=" * 60)

print(model)

print()


# ------------------------------------------------------------
# Count parameters
# ------------------------------------------------------------

number_of_parameters = sum(
    parameter.numel()
    for parameter in model.parameters()
    if parameter.requires_grad
)


print(
    f"Trainable parameters: "
    f"{number_of_parameters:,}"
)

print()


# ============================================================
# 13. LOSS ESTIMATION
# ============================================================
#
# We don't want to judge the model using only one random batch.
#
# Instead:
#
#     50 train batches
#     50 validation batches
#
# and calculate their average loss.
#
# ============================================================


@torch.no_grad()
def estimate_loss():

    model.eval()

    losses = {}

    for split in ("train", "val"):

        split_losses = torch.zeros(eval_iters)

        for step in range(eval_iters):

            inputs, targets = get_batch(split)

            logits, loss = model(
                inputs,
                targets
            )

            split_losses[step] = loss.item()


        losses[split] = split_losses.mean().item()


    model.train()

    return losses


# ============================================================
# 14. OPTIMIZER
# ============================================================

optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=learning_rate
)


# ============================================================
# 15. INITIAL LOSS
# ============================================================

print("=" * 60)
print("INITIAL LOSS")
print("=" * 60)

initial_losses = estimate_loss()

print(
    f"Train loss:      {initial_losses['train']:.4f}"
)

print(
    f"Validation loss: {initial_losses['val']:.4f}"
)

print()


# ============================================================
# 16. TRAINING LOOP
# ============================================================
#
# Each iteration:
#
#     1. Get random batch
#     2. Forward pass
#     3. Calculate loss
#     4. Clear old gradients
#     5. Backpropagation
#     6. Update parameters
#
# ============================================================

print("=" * 60)
print("TRAINING")
print("=" * 60)

best_val_loss = float("inf")


for step in range(max_iters):

    # --------------------------------------------------------
    # Get training batch
    # --------------------------------------------------------

    inputs, targets = get_batch("train")


    # --------------------------------------------------------
    # Forward pass
    # --------------------------------------------------------

    logits, loss = model(
        inputs,
        targets
    )


    # --------------------------------------------------------
    # Clear previous gradients
    # --------------------------------------------------------

    optimizer.zero_grad(set_to_none=True)


    # --------------------------------------------------------
    # Backpropagation
    # --------------------------------------------------------

    loss.backward()


    # --------------------------------------------------------
    # Update weights
    # --------------------------------------------------------

    optimizer.step()


    # --------------------------------------------------------
    # Evaluation
    # --------------------------------------------------------

    if (
        step == 0
        or (step + 1) % eval_interval == 0
    ):

        losses = estimate_loss()

        train_loss = losses["train"]
        val_loss = losses["val"]


        print(
            f"Step {step + 1:5d} | "
            f"Train loss: {train_loss:.4f} | "
            f"Val loss: {val_loss:.4f}"
        )


        # ----------------------------------------------------
        # Save best model
        # ----------------------------------------------------

        if val_loss < best_val_loss:

            best_val_loss = val_loss

            checkpoint = {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),

                "step": step + 1,

                "train_loss": train_loss,
                "val_loss": val_loss,

                "vocab_size": vocab_size,
                "block_size": block_size,

                "n_embd": n_embd,
                "n_head": n_head,
                "n_layer": n_layer,

                "dropout": dropout,

                # Save tokenizer too
                # so generation can be reproduced later.
                "stoi": stoi,
                "itos": itos,
            }


            torch.save(
                checkpoint,
                CHECKPOINT_PATH
            )


            print(
                f"  [OK] Best model saved "
                f"(val loss = {val_loss:.4f})"
            )


# ============================================================
# 17. FINAL LOSS
# ============================================================

print()
print("=" * 60)
print("FINAL LOSS")
print("=" * 60)

final_losses = estimate_loss()

print(
    f"Train loss:      {final_losses['train']:.4f}"
)

print(
    f"Validation loss: {final_losses['val']:.4f}"
)

print(
    f"Best validation loss: {best_val_loss:.4f}"
)

print()


# ============================================================
# 18. TEXT GENERATION
# ============================================================
#
# Start with newline.
#
# Then GPT predicts:
#
#     next character
#
# repeatedly.
#
# ============================================================


print("=" * 60)
print("GENERATING TEXT")
print("=" * 60)


# ------------------------------------------------------------
# Starting context
# ------------------------------------------------------------

start_character = "\n"

if start_character not in stoi:
    start_character = " "


start_token = stoi[start_character]


context = torch.tensor(
    [[start_token]],
    dtype=torch.long,
    device=device
)


# ------------------------------------------------------------
# Generate
# ------------------------------------------------------------

model.eval()

with torch.no_grad():

    generated_tokens = model.generate(
        context,
        max_new_tokens=max_new_tokens
    )


# ------------------------------------------------------------
# Convert tokens back to text
# ------------------------------------------------------------

generated_text = decode(
    generated_tokens[0].tolist()
)


print()
print(generated_text)
print()


# ============================================================
# 19. SAVE GENERATED TEXT
# ============================================================

with open(
    GENERATED_PATH,
    "w",
    encoding="utf-8"
) as file:

    file.write(generated_text)


print(
    f"Generated text saved to: "
    f"{GENERATED_PATH}"
)

print()


# ============================================================
# 20. TRAINING SUMMARY
# ============================================================

print("=" * 60)
print("TRAINING COMPLETE")
print("=" * 60)

print(f"Dataset:          {DATA_PATH}")

print(
    f"Dataset size:     {len(data):,} tokens"
)

print(
    f"Vocabulary size:  {vocab_size}"
)

print(
    f"Train tokens:     {len(train_data):,}"
)

print(
    f"Validation tokens:{len(val_data):,}"
)

print(
    f"Batch size:       {batch_size}"
)

print(
    f"Block size:       {block_size}"
)

print(
    f"Embedding size:   {n_embd}"
)

print(
    f"Attention heads:  {n_head}"
)

print(
    f"Transformer layers:{n_layer}"
)

print(
    f"Learning rate:    {learning_rate}"
)

print(
    f"Best val loss:    {best_val_loss:.4f}"
)

print(
    f"Checkpoint:       {CHECKPOINT_PATH}"
)

print(
    f"Generated text:   {GENERATED_PATH}"
)

print("=" * 60)