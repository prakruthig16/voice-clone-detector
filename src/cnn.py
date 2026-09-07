"""
Optional deep-learning backend: a small CNN over log-mel spectrograms.

Only used if PyTorch is installed. Import guarded so the classical pipeline
works with zero deep-learning dependencies.
"""
from __future__ import annotations

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:  # pragma: no cover
    TORCH_AVAILABLE = False


if TORCH_AVAILABLE:

    class SpoofCNN(nn.Module):
        """Compact CNN classifier for (1, n_mels, T) log-mel inputs."""

        def __init__(self, n_classes: int = 2):
            super().__init__()
            self.features = nn.Sequential(
                nn.Conv2d(1, 16, 3, padding=1), nn.BatchNorm2d(16),
                nn.ReLU(), nn.MaxPool2d(2),
                nn.Conv2d(16, 32, 3, padding=1), nn.BatchNorm2d(32),
                nn.ReLU(), nn.MaxPool2d(2),
                nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64),
                nn.ReLU(), nn.AdaptiveAvgPool2d((4, 4)),
            )
            self.classifier = nn.Sequential(
                nn.Flatten(), nn.Dropout(0.3),
                nn.Linear(64 * 4 * 4, 64), nn.ReLU(),
                nn.Linear(64, n_classes),
            )

        def forward(self, x):
            return self.classifier(self.features(x))
