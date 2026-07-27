from __future__ import annotations

import torch
from torch import nn


class AudioCommandCNN(nn.Module):
    """CNN over log-mel spectrograms for SpeechCommands."""

    def __init__(self, classes: int, sample_rate: int = 16000, n_mels: int = 64):
        super().__init__()
        try:
            import torchaudio
        except ImportError as exc:
            raise RuntimeError('Instale el extra audio: pip install -e ".[audio]"') from exc
        self.mel = torchaudio.transforms.MelSpectrogram(sample_rate=sample_rate, n_mels=n_mels)
        self.amplitude_to_db = torchaudio.transforms.AmplitudeToDB()
        self.network = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 96, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d(1),
        )
        self.head = nn.Linear(96, classes)

    def forward(self, waveform: torch.Tensor) -> torch.Tensor:
        if waveform.ndim == 2:
            waveform = waveform.unsqueeze(1)
        spectrogram = self.amplitude_to_db(self.mel(waveform.squeeze(1))).unsqueeze(1)
        return self.head(self.network(spectrogram).flatten(1))
