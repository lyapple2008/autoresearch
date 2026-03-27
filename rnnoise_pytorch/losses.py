from __future__ import annotations

import torch
import torch.nn.functional as F


def rnnoise_gain_loss(target: torch.Tensor, pred: torch.Tensor) -> torch.Tensor:
    mask = torch.minimum(target + 1.0, torch.ones_like(target))
    sqrt_err = torch.sqrt(torch.clamp(pred, min=1e-8)) - torch.sqrt(torch.clamp(target, min=1e-8))
    term = 10.0 * torch.square(torch.square(sqrt_err)) + torch.square(sqrt_err)
    bce = 0.01 * F.binary_cross_entropy(pred, target, reduction="none")
    return torch.mean(mask * (term + bce))


def rnnoise_vad_loss(target: torch.Tensor, pred: torch.Tensor) -> torch.Tensor:
    weights = 2.0 * torch.abs(target - 0.5)
    return torch.mean(weights * F.binary_cross_entropy(pred, target, reduction="none"))


def rnnoise_total_loss(
    gain_target: torch.Tensor,
    gain_pred: torch.Tensor,
    vad_target: torch.Tensor,
    vad_pred: torch.Tensor,
    gain_weight: float = 10.0,
    vad_weight: float = 0.5,
) -> torch.Tensor:
    return gain_weight * rnnoise_gain_loss(gain_target, gain_pred) + vad_weight * rnnoise_vad_loss(
        vad_target, vad_pred
    )

