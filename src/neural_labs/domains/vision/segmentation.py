from __future__ import annotations

import torch
from torch import nn


class DoubleConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class UNetSmall(nn.Module):
    """Compact U-Net for three-class Oxford-IIIT Pet trimap segmentation."""

    def __init__(self, in_channels: int = 3, classes: int = 3, base: int = 32):
        super().__init__()
        self.enc1 = DoubleConv(in_channels, base)
        self.enc2 = DoubleConv(base, base * 2)
        self.enc3 = DoubleConv(base * 2, base * 4)
        self.pool = nn.MaxPool2d(2)
        self.bottleneck = DoubleConv(base * 4, base * 8)
        self.up3 = nn.ConvTranspose2d(base * 8, base * 4, 2, stride=2)
        self.dec3 = DoubleConv(base * 8, base * 4)
        self.up2 = nn.ConvTranspose2d(base * 4, base * 2, 2, stride=2)
        self.dec2 = DoubleConv(base * 4, base * 2)
        self.up1 = nn.ConvTranspose2d(base * 2, base, 2, stride=2)
        self.dec1 = DoubleConv(base * 2, base)
        self.head = nn.Conv2d(base, classes, 1)

    @staticmethod
    def _align(skip: torch.Tensor, upsampled: torch.Tensor) -> torch.Tensor:
        if skip.shape[-2:] == upsampled.shape[-2:]:
            return skip
        return nn.functional.interpolate(skip, size=upsampled.shape[-2:], mode="bilinear", align_corners=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        original_size = x.shape[-2:]
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        x = self.bottleneck(self.pool(e3))
        x = self.up3(x)
        x = self.dec3(torch.cat([x, self._align(e3, x)], dim=1))
        x = self.up2(x)
        x = self.dec2(torch.cat([x, self._align(e2, x)], dim=1))
        x = self.up1(x)
        x = self.dec1(torch.cat([x, self._align(e1, x)], dim=1))
        logits = self.head(x)
        if logits.shape[-2:] != original_size:
            logits = nn.functional.interpolate(logits, size=original_size, mode="bilinear", align_corners=False)
        return logits


def mean_iou(logits: torch.Tensor, targets: torch.Tensor, classes: int = 3) -> float:
    predictions = logits.argmax(dim=1)
    values: list[torch.Tensor] = []
    for class_id in range(classes):
        predicted = predictions == class_id
        expected = targets == class_id
        union = (predicted | expected).sum()
        if union:
            values.append((predicted & expected).sum().float() / union.float())
    return float(torch.stack(values).mean().item()) if values else 0.0
