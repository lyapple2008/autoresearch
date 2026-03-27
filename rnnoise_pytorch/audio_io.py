from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np
import soundfile as sf
from scipy.signal import resample_poly


def _to_mono(audio: np.ndarray) -> np.ndarray:
    if audio.ndim == 1:
        return audio
    return np.mean(audio, axis=1)


def load_audio_16k_mono(path: Path, target_sr: int = 16_000) -> np.ndarray:
    wav, sr = sf.read(str(path), always_2d=False)
    wav = _to_mono(np.asarray(wav, dtype=np.float32))
    if sr != target_sr:
        wav = resample_poly(wav, target_sr, sr).astype(np.float32)
    peak = np.max(np.abs(wav)) + 1e-8
    if peak > 1.0:
        wav = wav / peak
    return wav.astype(np.float32)


def align_pair(clean: np.ndarray, noisy: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    n = min(clean.shape[0], noisy.shape[0])
    return clean[:n], noisy[:n]

