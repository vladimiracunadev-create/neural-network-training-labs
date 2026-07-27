from __future__ import annotations

import math

import torch
from torch import nn

from ..vision.models import DCGANDiscriminator, DCGANGenerator


class WGANGenerator(DCGANGenerator):
    pass


class WGANCritic(DCGANDiscriminator):
    pass


def gradient_penalty(critic: nn.Module, real: torch.Tensor, fake: torch.Tensor) -> torch.Tensor:
    alpha = torch.rand(real.shape[0], 1, 1, 1, device=real.device)
    interpolated = (alpha * real + (1 - alpha) * fake).requires_grad_(True)
    scores = critic(interpolated)
    gradients = torch.autograd.grad(
        scores,
        interpolated,
        grad_outputs=torch.ones_like(scores),
        create_graph=True,
        retain_graph=True,
    )[0]
    return ((gradients.flatten(1).norm(2, dim=1) - 1.0) ** 2).mean()


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dimension: int):
        super().__init__()
        self.dimension = dimension

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        half = self.dimension // 2
        frequencies = torch.exp(
            -math.log(10000) * torch.arange(half, device=timesteps.device).float() / max(half - 1, 1)
        )
        angles = timesteps.float().unsqueeze(1) * frequencies.unsqueeze(0)
        embedding = torch.cat([angles.sin(), angles.cos()], dim=1)
        return nn.functional.pad(embedding, (0, self.dimension - embedding.shape[1]))


class TinyDenoiser(nn.Module):
    def __init__(self, channels: int = 1, hidden: int = 64, time_dim: int = 64):
        super().__init__()
        self.time_embedding = nn.Sequential(SinusoidalTimeEmbedding(time_dim), nn.Linear(time_dim, hidden), nn.SiLU())
        self.input = nn.Conv2d(channels, hidden, 3, padding=1)
        self.middle = nn.Sequential(
            nn.GroupNorm(8, hidden), nn.SiLU(), nn.Conv2d(hidden, hidden, 3, padding=1),
            nn.GroupNorm(8, hidden), nn.SiLU(), nn.Conv2d(hidden, hidden, 3, padding=1),
        )
        self.output = nn.Conv2d(hidden, channels, 3, padding=1)

    def forward(self, noisy: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        x = self.input(noisy)
        time = self.time_embedding(timesteps).unsqueeze(-1).unsqueeze(-1)
        return self.output(self.middle(x + time))


def cosine_beta_schedule(steps: int, offset: float = 0.008) -> torch.Tensor:
    values = torch.linspace(0, steps, steps + 1)
    cumulative = torch.cos(((values / steps + offset) / (1 + offset)) * math.pi * 0.5) ** 2
    cumulative = cumulative / cumulative[0]
    betas = 1 - cumulative[1:] / cumulative[:-1]
    return betas.clamp(1e-5, 0.999)


def add_noise(clean: torch.Tensor, noise: torch.Tensor, timesteps: torch.Tensor, betas: torch.Tensor) -> torch.Tensor:
    alphas = 1.0 - betas
    cumulative = torch.cumprod(alphas, dim=0)
    selected = cumulative[timesteps].view(-1, 1, 1, 1).to(clean.device)
    return selected.sqrt() * clean + (1 - selected).sqrt() * noise
