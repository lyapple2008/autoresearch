from __future__ import annotations

import torch
import torch.nn as nn


class RNNoiseTorch(nn.Module):
    """PyTorch implementation mirroring the classic RNNoise training graph."""

    def __init__(self, in_dim: int = 42, bands: int = 22):
        super().__init__()
        self.input_dense = nn.Linear(in_dim, 24)
        self.vad_gru = nn.GRU(input_size=24, hidden_size=24, batch_first=True)
        self.vad_out = nn.Linear(24, 1)

        self.noise_gru = nn.GRU(input_size=24 + 24 + in_dim, hidden_size=48, batch_first=True)
        self.denoise_gru = nn.GRU(input_size=24 + 48 + in_dim, hidden_size=96, batch_first=True)
        self.denoise_out = nn.Linear(96, bands)
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor):
        # x: [B, T, 42]
        tmp = torch.tanh(self.input_dense(x))
        vad_gru, _ = self.vad_gru(tmp)
        vad = torch.sigmoid(self.vad_out(vad_gru))

        noise_in = torch.cat([tmp, vad_gru, x], dim=-1)
        noise_gru, _ = self.noise_gru(noise_in)

        denoise_in = torch.cat([vad_gru, noise_gru, x], dim=-1)
        denoise_gru, _ = self.denoise_gru(denoise_in)
        gains = torch.sigmoid(self.denoise_out(denoise_gru))
        return gains, vad

