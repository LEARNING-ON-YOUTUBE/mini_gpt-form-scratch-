import argparse
from pathlib import Path
from typing import Callable

import torch

from .model import GPTLanguageModel

PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_CHECKPOINT_PATH = PROJECT_DIR / "checkpoint" / "best_model.pt"
DEFAULT_OUTPUT_PATH = PROJECT_DIR / "output" / "generated.txt"


def _load_model(
    checkpoint_path: Path,
) -> tuple[GPTLanguageModel, dict[str, int], dict[int, str], torch.device]:
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Model checkpoint was not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = GPTLanguageModel(
        vocab_size=checkpoint["vocab_size"],
        block_size=checkpoint["block_size"],
        n_embd=checkpoint["n_embd"],
        n_head=checkpoint["n_head"],
        n_layer=checkpoint["n_layer"],
        dropout=checkpoint["dropout"],
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    return model, checkpoint["stoi"], checkpoint["itos"], device


def generate_text(
    checkpoint_path: Path = DEFAULT_CHECKPOINT_PATH,
    prompt: str = "The ",
    max_new_tokens: int = 500,
    output_path: Path | None = DEFAULT_OUTPUT_PATH,
) -> str:
    if max_new_tokens < 0:
        raise ValueError("max_new_tokens must be non-negative")
    model, stoi, itos, device = _load_model(checkpoint_path)
    if any(character not in stoi for character in prompt):
        raise ValueError("Prompt contains characters outside the checkpoint vocabulary")

    context = torch.tensor(
        [[stoi[character] for character in prompt]], dtype=torch.long, device=device
    )
    generated_ids = model.generate(context, max_new_tokens)
    generated_text = "".join(itos[index] for index in generated_ids[0].tolist())

    if output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(generated_text, encoding="utf-8")
    return generated_text


def interactive_generate(
    checkpoint_path: Path = DEFAULT_CHECKPOINT_PATH,
    max_new_tokens: int = 300,
    output_path: Path | None = DEFAULT_OUTPUT_PATH,
    input_fn: Callable[[str], str] = input,
) -> None:
    if max_new_tokens < 0:
        raise ValueError("max_new_tokens must be non-negative")

    model, stoi, itos, device = _load_model(checkpoint_path)
    print(f"Model loaded on {device}. Enter a prompt, or type 'q' to quit.")

    while True:
        try:
            prompt = input_fn("Prompt: ")
        except (EOFError, KeyboardInterrupt):
            print("\nExiting prompt loop.")
            break

        if prompt.strip().lower() in {"q", "quit", "exit"}:
            print("Exiting prompt loop.")
            break
        if not prompt:
            print("Please enter a non-empty prompt.")
            continue
        if any(character not in stoi for character in prompt):
            print("Prompt contains characters outside the model vocabulary; try again.")
            continue

        context = torch.tensor(
            [[stoi[character] for character in prompt]], dtype=torch.long, device=device
        )
        generated_ids = model.generate(context, max_new_tokens)
        generated_text = "".join(itos[index] for index in generated_ids[0].tolist())
        print(f"Generated:\n{generated_text}\n")

        if output_path is not None:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with output_path.open("a", encoding="utf-8") as output_file:
                output_file.write(f"Prompt: {prompt}\n{generated_text}\n\n")
            print(f"Saved to {output_path}")


generate = generate_text


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Interactively generate text from the trained character-level model"
    )
    parser.add_argument("--tokens", type=int, default=300)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    args = parser.parse_args()
    interactive_generate(
        checkpoint_path=args.checkpoint,
        max_new_tokens=args.tokens,
        output_path=args.output,
    )


if __name__ == "__main__":
    main()