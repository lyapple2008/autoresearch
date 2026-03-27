from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
from scipy.fft import rfft

from .config import AUDIO, FEATURE


@dataclass
class FeatureBundle:
    features: np.ndarray
    gains: np.ndarray
    vad: np.ndarray
    noisy_mag: np.ndarray
    noisy_phase: np.ndarray


def _preemphasis(x: np.ndarray, coeff: float) -> np.ndarray:
    y = np.copy(x)
    y[1:] = x[1:] - coeff * x[:-1]
    y[0] = x[0]
    return y


def _frame_signal(x: np.ndarray, frame_size: int, hop: int) -> np.ndarray:
    if x.shape[0] < frame_size:
        x = np.pad(x, (0, frame_size - x.shape[0]))
    n_frames = 1 + (x.shape[0] - frame_size) // hop
    total_len = (n_frames - 1) * hop + frame_size
    x = x[:total_len]
    shape = (n_frames, frame_size)
    strides = (x.strides[0] * hop, x.strides[0])
    return np.lib.stride_tricks.as_strided(x, shape=shape, strides=strides).copy()


def erb_filterbank(num_fft_bins: int, sr: int, num_bands: int) -> np.ndarray:
    freqs = np.linspace(0, sr / 2, num_fft_bins)
    erb = 21.4 * np.log10(1 + 0.00437 * freqs)
    erb_edges = np.linspace(erb.min(), erb.max(), num_bands + 2)
    fb = np.zeros((num_bands, num_fft_bins), dtype=np.float32)
    for i in range(num_bands):
        l, c, r = erb_edges[i], erb_edges[i + 1], erb_edges[i + 2]
        up = np.clip((erb - l) / (c - l + 1e-8), 0.0, 1.0)
        down = np.clip((r - erb) / (r - c + 1e-8), 0.0, 1.0)
        fb[i] = np.minimum(up, down)
    fb /= np.maximum(fb.sum(axis=1, keepdims=True), 1e-8)
    return fb


def _pitch_corr_feature(frame: np.ndarray, sr: int) -> np.ndarray:
    lag_min = max(1, sr // FEATURE.pitch_max_hz)
    lag_max = max(lag_min + 1, sr // FEATURE.pitch_min_hz)
    corr = np.correlate(frame, frame, mode="full")
    corr = corr[len(corr) // 2 :]
    region = corr[lag_min:lag_max]
    if region.size == 0:
        return np.zeros(6, dtype=np.float32)
    idx = np.argmax(region)
    best_corr = region[idx] / (corr[0] + 1e-8)
    lag = lag_min + idx
    pitch_hz = sr / max(lag, 1)
    frame_energy = np.log10(np.mean(frame * frame) + 1e-8)
    return np.array(
        [
            best_corr,
            pitch_hz / FEATURE.pitch_max_hz,
            frame_energy,
            np.std(frame),
            np.max(np.abs(frame)),
            float(np.mean(np.abs(np.diff(frame)))),
        ],
        dtype=np.float32,
    )


def _build_feature_matrix(band_energy: np.ndarray, pitch_feats: np.ndarray) -> np.ndarray:
    log_band = np.log10(np.maximum(band_energy, 1e-8))
    delta = np.zeros_like(log_band)
    delta[1:] = log_band[1:] - log_band[:-1]
    feat = np.concatenate([log_band, pitch_feats, delta[:, :14]], axis=1)
    assert feat.shape[1] == FEATURE.feature_dim
    return feat.astype(np.float32)


def extract_features_and_targets(noisy: np.ndarray, clean: np.ndarray) -> FeatureBundle:
    noisy = _preemphasis(noisy.astype(np.float32), AUDIO.preemphasis)
    clean = _preemphasis(clean.astype(np.float32), AUDIO.preemphasis)

    noisy_frames = _frame_signal(noisy, AUDIO.frame_size, AUDIO.hop_size)
    clean_frames = _frame_signal(clean, AUDIO.frame_size, AUDIO.hop_size)
    n = min(noisy_frames.shape[0], clean_frames.shape[0])
    noisy_frames = noisy_frames[:n]
    clean_frames = clean_frames[:n]

    window = np.hanning(AUDIO.frame_size).astype(np.float32)
    noisy_spec = rfft(noisy_frames * window[None, :], n=AUDIO.n_fft, axis=1)
    clean_spec = rfft(clean_frames * window[None, :], n=AUDIO.n_fft, axis=1)
    noisy_mag = np.abs(noisy_spec).astype(np.float32)
    clean_mag = np.abs(clean_spec).astype(np.float32)
    noisy_phase = np.angle(noisy_spec).astype(np.float32)

    fb = erb_filterbank(noisy_mag.shape[1], AUDIO.sample_rate, AUDIO.num_bands)
    noisy_band = noisy_mag @ fb.T
    clean_band = clean_mag @ fb.T
    gains = np.clip(clean_band / (noisy_band + 1e-8), 0.0, 1.0).astype(np.float32)

    pitch_feats = np.stack(
        [_pitch_corr_feature(frm, AUDIO.sample_rate) for frm in noisy_frames], axis=0
    )
    features = _build_feature_matrix(noisy_band, pitch_feats)

    rms = np.sqrt(np.mean(clean_frames * clean_frames, axis=1) + 1e-12)
    vad = (20.0 * np.log10(rms + 1e-8) > FEATURE.vad_energy_threshold_db).astype(np.float32)
    vad = vad[:, None]

    return FeatureBundle(
        features=features,
        gains=gains,
        vad=vad,
        noisy_mag=noisy_mag,
        noisy_phase=noisy_phase,
    )


def band_gains_to_bin_gains(num_bins: int, gains: np.ndarray) -> np.ndarray:
    fb = erb_filterbank(num_bins, AUDIO.sample_rate, AUDIO.num_bands)
    denom = np.maximum(fb.sum(axis=0, keepdims=True), 1e-8)
    proj = gains @ fb
    return (proj / denom).astype(np.float32)


def reconstruct_from_gains(noisy_mag: np.ndarray, noisy_phase: np.ndarray, gains: np.ndarray) -> np.ndarray:
    bin_gains = band_gains_to_bin_gains(noisy_mag.shape[1], gains)
    enhanced_mag = noisy_mag * np.clip(bin_gains, 0.0, 1.0)
    complex_spec = enhanced_mag * np.exp(1j * noisy_phase)
    frames = np.fft.irfft(complex_spec, n=AUDIO.n_fft, axis=1)[:, : AUDIO.frame_size]
    hop = AUDIO.hop_size
    frame_size = AUDIO.frame_size
    out_len = (frames.shape[0] - 1) * hop + frame_size
    out = np.zeros(out_len, dtype=np.float32)
    win = np.hanning(frame_size).astype(np.float32)
    norm = np.zeros(out_len, dtype=np.float32)
    for i, frame in enumerate(frames):
        s = i * hop
        out[s : s + frame_size] += frame * win
        norm[s : s + frame_size] += win * win
    out /= np.maximum(norm, 1e-8)
    return out.astype(np.float32)

