from dataclasses import dataclass


@dataclass(frozen=True)
class Config:
    batch_size: int = 32
    block_size: int = 8
    learning_rate: float = 1e-2
    max_steps: int = 1500
    eval_interval: int = 300
    eval_batches: int = 20
    train_fraction: float = 0.9
    seed: int = 1337
    max_new_tokens: int = 500


DEFAULT_CONFIG = Config()