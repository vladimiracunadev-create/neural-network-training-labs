from __future__ import annotations

import math

import torch
from torch import nn

from ...core.registry import MODEL_REGISTRY


class MaskedTextRNN(nn.Module):
    def __init__(self, vocab_size: int, classes: int, kind: str = "rnn", embedding_dim: int = 96, hidden_dim: int = 128):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        recurrent = nn.LSTM if kind == "lstm" else nn.RNN
        self.recurrent = recurrent(embedding_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.head = nn.Sequential(nn.Dropout(0.2), nn.Linear(hidden_dim * 2, 1 if classes == 2 else classes))
        self.binary = classes == 2

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        tokens = tokens.long()
        lengths = tokens.ne(0).sum(dim=1).clamp_min(1).cpu()
        embedded = self.embedding(tokens)
        packed = nn.utils.rnn.pack_padded_sequence(embedded, lengths, batch_first=True, enforce_sorted=False)
        _output, hidden = self.recurrent(packed)
        hidden_tensor = hidden[0] if isinstance(hidden, tuple) else hidden
        pooled = torch.cat([hidden_tensor[-2], hidden_tensor[-1]], dim=1)
        logits = self.head(pooled)
        return logits.squeeze(-1) if self.binary else logits


class PositionalEncoding(nn.Module):
    def __init__(self, dimension: int, length: int = 512):
        super().__init__()
        position = torch.arange(length).unsqueeze(1)
        divisor = torch.exp(torch.arange(0, dimension, 2) * (-math.log(10000.0) / dimension))
        encoding = torch.zeros(length, dimension)
        encoding[:, 0::2] = torch.sin(position * divisor)
        encoding[:, 1::2] = torch.cos(position * divisor)
        self.register_buffer("encoding", encoding.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.encoding[:, : x.shape[1]]


class AttentionEncoderLayer(nn.Module):
    def __init__(self, dimension: int, heads: int, dropout: float = 0.1):
        super().__init__()
        self.attention = nn.MultiheadAttention(dimension, heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(dimension)
        self.norm2 = nn.LayerNorm(dimension)
        self.ffn = nn.Sequential(nn.Linear(dimension, dimension * 4), nn.GELU(), nn.Dropout(dropout), nn.Linear(dimension * 4, dimension))
        self.dropout = nn.Dropout(dropout)
        self.last_attention: torch.Tensor | None = None

    def forward(self, x: torch.Tensor, padding_mask: torch.Tensor | None = None) -> torch.Tensor:
        attended, weights = self.attention(x, x, x, key_padding_mask=padding_mask, need_weights=True, average_attn_weights=False)
        self.last_attention = weights.detach()
        x = self.norm1(x + self.dropout(attended))
        return self.norm2(x + self.dropout(self.ffn(x)))


class InspectableTextTransformer(nn.Module):
    def __init__(self, vocab_size: int, classes: int, dimension: int = 128, heads: int = 4, layers: int = 2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, dimension, padding_idx=0)
        self.position = PositionalEncoding(dimension)
        self.layers = nn.ModuleList([AttentionEncoderLayer(dimension, heads) for _ in range(layers)])
        self.head = nn.Linear(dimension, classes)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        tokens = tokens.long()
        mask = tokens.eq(0)
        x = self.position(self.embedding(tokens) * math.sqrt(self.embedding.embedding_dim))
        for layer in self.layers:
            x = layer(x, mask)
        valid = (~mask).unsqueeze(-1)
        pooled = (x * valid).sum(dim=1) / valid.sum(dim=1).clamp_min(1)
        return self.head(pooled)

    def attention_maps(self) -> list[torch.Tensor]:
        return [layer.last_attention for layer in self.layers if layer.last_attention is not None]


@MODEL_REGISTRY.register("rnn_text", domain="text", description="RNN bidireccional con padding enmascarado")
def create_rnn(*, input_shape, num_classes, metadata=None, **kwargs):
    del input_shape
    metadata = metadata or {}
    return MaskedTextRNN(int(metadata["vocab_size"]), int(num_classes or 2), kind=str(kwargs.get("kind", "rnn")))


@MODEL_REGISTRY.register("transformer_text", domain="text", description="Transformer inspeccionable con mapas de atención")
def create_transformer(*, input_shape, num_classes, metadata=None, **kwargs):
    del input_shape
    metadata = metadata or {}
    return InspectableTextTransformer(
        int(metadata["vocab_size"]),
        int(num_classes or 2),
        dimension=int(kwargs.get("dimension", 128)),
        heads=int(kwargs.get("heads", 4)),
        layers=int(kwargs.get("layers", 2)),
    )
