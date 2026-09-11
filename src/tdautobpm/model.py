"""TempoNet, the tempo model.

The architecture and the trained weights (``models/temponet_ckpt.pt``) are by shhhum,
from tdautobpmsync (https://github.com/shhhum/tdautobpmsync), and are used here
unchanged apart from packaging. See the README's Credits section.
"""

import time
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio


# -----------------------------
# Config
# -----------------------------
@dataclass
class AudioConfig:
    sample_rate: int = 22050
    n_fft: int = 1024
    hop_length: int = 256
    n_mels: int = 80
    f_min: int = 30
    f_max: int = 8000
    win_length: int = 1024


def soft_bpm_target(
    bpm: float,
    bpm_min: int,
    bpm_max: int,
    sigma_main: float = 1.2,
    sigma_side: float = 2.0,
    w_main: float = 1.0,
    w_half: float = 0.05,
    w_double: float = 0.05,
) -> torch.Tensor:
    bins = torch.arange(bpm_min, bpm_max + 1, dtype=torch.float32)

    def gauss(center, sigma):
        return torch.exp(-0.5 * ((bins - center) / sigma) ** 2)

    y = torch.zeros_like(bins)

    if bpm_min <= bpm <= bpm_max:
        y += w_main * gauss(bpm, sigma_main)

    half = bpm * 0.5
    dbl = bpm * 2.0

    if bpm_min <= half <= bpm_max:
        y += w_half * gauss(half, sigma_side)
    if bpm_min <= dbl <= bpm_max:
        y += w_double * gauss(dbl, sigma_side)

    return y / y.sum().clamp_min(1e-8)


# -----------------------------
# Model
# -----------------------------
class CausalConv1d(nn.Conv1d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.left_padding = (self.kernel_size[0] - 1) * self.dilation[0]

    def forward(self, x):
        x = F.pad(x, (self.left_padding, 0))
        return super().forward(x)


class TCNBlock(nn.Module):
    def __init__(self, channels: int, kernel_size: int, dilation: int, dropout: float):
        super().__init__()
        self.conv1 = CausalConv1d(
            channels, channels, kernel_size=kernel_size, dilation=dilation
        )
        self.conv2 = CausalConv1d(
            channels, channels, kernel_size=kernel_size, dilation=dilation
        )
        self.norm1 = nn.BatchNorm1d(channels)
        self.norm2 = nn.BatchNorm1d(channels)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        x = self.drop(F.relu(self.norm1(self.conv1(x))))
        x = self.drop(F.relu(self.norm2(self.conv2(x))))
        return x + residual


class TempoNet(nn.Module):
    def __init__(
        self,
        n_mels: int,
        bpm_min: int,
        bpm_max: int,
        base_channels: int = 32,
        tcn_layers: int = 6,
        tcn_kernel: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.bpm_min = int(bpm_min)
        self.bpm_max = int(bpm_max)
        self.num_bins = self.bpm_max - self.bpm_min + 1

        self.cnn = nn.Sequential(
            nn.Conv2d(
                1, base_channels, kernel_size=(5, 5), stride=(2, 1), padding=(2, 2)
            ),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(),
            nn.Conv2d(
                base_channels,
                base_channels,
                kernel_size=(3, 3),
                stride=(2, 1),
                padding=(1, 1),
            ),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(),
            nn.Conv2d(
                base_channels,
                base_channels,
                kernel_size=(3, 3),
                stride=(2, 1),
                padding=(1, 1),
            ),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(),
        )

        # infer freq bins after cnn for projection
        with torch.no_grad():
            dummy = torch.zeros(1, 1, n_mels, 10)
            y = self.cnn(dummy)
            f_bins = y.shape[2]

        self.proj = nn.Linear(base_channels * f_bins, base_channels)

        self.tcn = nn.Sequential(
            *[
                TCNBlock(
                    base_channels,
                    kernel_size=tcn_kernel,
                    dilation=2**i,
                    dropout=dropout,
                )
                for i in range(tcn_layers)
            ]
        )

        self.head = nn.Sequential(
            nn.Linear(base_channels, base_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(base_channels, self.num_bins),
        )
        self.enable_timing = False
        self.timing_stats = []

    def forward(self, logmel: torch.Tensor) -> torch.Tensor:
        # logmel: [B, n_mels, T]
        if self.enable_timing:
            t0 = time.time()

        x = logmel.unsqueeze(1)  # [B, 1, n_mels, T]
        x = self.cnn(x)  # [B, C, n_mels', T]
        B, C, Fm, T = x.shape
        x = x.permute(0, 3, 1, 2).contiguous().view(B, T, C * Fm)  # [B, T, C*Fm]
        x = self.proj(x)  # [B, T, C]
        x = x.transpose(1, 2)  # [B, C, T]
        x = self.tcn(x)
        feat = x[:, :, -1]  # causal last step
        logits = self.head(feat)  # [B, bins]

        if self.enable_timing:
            elapsed = time.time() - t0
            self.timing_stats.append(elapsed)

        return logits
