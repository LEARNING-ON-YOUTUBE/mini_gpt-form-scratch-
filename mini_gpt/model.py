import torch
import torch.nn as nn
import torch.nn.functional as F


class BigramLanguageModel(nn.Module):
    def __init__(self, vocab_size: int) -> None:
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)

    def forward(
        self, idx: torch.Tensor, targets: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        logits = self.token_embedding_table(idx)
        loss = None
        if targets is not None:
            batch_size, sequence_length, vocab_size = logits.shape
            logits = logits.view(batch_size * sequence_length, vocab_size)
            targets = targets.view(batch_size * sequence_length)
            loss = F.cross_entropy(logits, targets)
        return logits, loss

    @torch.no_grad()
    def generate(self, idx: torch.Tensor, max_new_tokens: int) -> torch.Tensor:
        if idx.ndim != 2 or idx.shape[1] == 0:
            raise ValueError("idx must be a non-empty 2D tensor of token IDs")
        if max_new_tokens < 0:
            raise ValueError("max_new_tokens must be non-negative")

        for _ in range(max_new_tokens):
            logits, _ = self(idx)
            probabilities = F.softmax(logits[:, -1, :], dim=-1)
            next_token = torch.multinomial(probabilities, num_samples=1)
            idx = torch.cat((idx, next_token), dim=1)
        return idx


class CausalSelfAttention(nn.Module):
    def __init__(self, n_embd: int, n_head: int, dropout: float) -> None:
        super().__init__()
        if n_embd % n_head != 0:
            raise ValueError("n_embd must be divisible by n_head")

        self.n_head = n_head
        self.head_size = n_embd // n_head
        self.dropout = dropout
        self.query_key_value = nn.Linear(n_embd, 3 * n_embd)
        self.projection = nn.Linear(n_embd, n_embd)
        self.residual_dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, sequence_length, channels = x.shape
        query, key, value = self.query_key_value(x).split(channels, dim=2)
        query = query.view(batch_size, sequence_length, self.n_head, self.head_size).transpose(1, 2)
        key = key.view(batch_size, sequence_length, self.n_head, self.head_size).transpose(1, 2)
        value = value.view(batch_size, sequence_length, self.n_head, self.head_size).transpose(1, 2)

        attended = F.scaled_dot_product_attention(
            query,
            key,
            value,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=True,
        )
        attended = attended.transpose(1, 2).contiguous().view(
            batch_size, sequence_length, channels
        )
        return self.residual_dropout(self.projection(attended))


class FeedForward(nn.Module):
    def __init__(self, n_embd: int, dropout: float) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.GELU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class Block(nn.Module):
    def __init__(self, n_embd: int, n_head: int, dropout: float) -> None:
        super().__init__()
        self.attention = CausalSelfAttention(n_embd, n_head, dropout)
        self.feed_forward = FeedForward(n_embd, dropout)
        self.attention_norm = nn.LayerNorm(n_embd)
        self.feed_forward_norm = nn.LayerNorm(n_embd)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attention(self.attention_norm(x))
        return x + self.feed_forward(self.feed_forward_norm(x))


class GPTLanguageModel(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        block_size: int = 64,
        n_embd: int = 128,
        n_head: int = 4,
        n_layer: int = 4,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        if vocab_size <= 0 or block_size <= 0 or n_layer <= 0:
            raise ValueError("vocab_size, block_size, and n_layer must be positive")
        if n_embd <= 0 or n_head <= 0 or n_embd % n_head != 0:
            raise ValueError("n_embd must be positive and divisible by n_head")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("dropout must be in the range [0, 1)")

        self.block_size = block_size
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.embedding_dropout = nn.Dropout(dropout)
        self.blocks = nn.Sequential(
            *(Block(n_embd, n_head, dropout) for _ in range(n_layer))
        )
        self.final_norm = nn.LayerNorm(n_embd)
        self.language_model_head = nn.Linear(n_embd, vocab_size)
        self.apply(self._initialize_weights)

    @staticmethod
    def _initialize_weights(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self, idx: torch.Tensor, targets: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if idx.ndim != 2:
            raise ValueError("idx must have shape [batch_size, sequence_length]")
        batch_size, sequence_length = idx.shape
        if sequence_length > self.block_size:
            raise ValueError(
                f"Input sequence length {sequence_length} exceeds block_size {self.block_size}"
            )

        positions = torch.arange(sequence_length, device=idx.device)
        x = self.embedding_dropout(
            self.token_embedding_table(idx) + self.position_embedding_table(positions)
        )
        x = self.final_norm(self.blocks(x))
        logits = self.language_model_head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(
                logits.reshape(batch_size * sequence_length, -1),
                targets.reshape(batch_size * sequence_length),
            )
        return logits, loss

    @torch.no_grad()
    def generate(self, idx: torch.Tensor, max_new_tokens: int) -> torch.Tensor:
        if idx.ndim != 2 or idx.shape[1] == 0:
            raise ValueError("idx must be a non-empty 2D tensor of token IDs")
        if max_new_tokens < 0:
            raise ValueError("max_new_tokens must be non-negative")

        for _ in range(max_new_tokens):
            context = idx[:, -self.block_size :]
            logits, _ = self(context)
            probabilities = F.softmax(logits[:, -1, :], dim=-1)
            next_token = torch.multinomial(probabilities, num_samples=1)
            idx = torch.cat((idx, next_token), dim=1)
        return idx


GPT = GPTLanguageModel
