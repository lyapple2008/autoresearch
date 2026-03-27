from __future__ import annotations

import argparse
from pathlib import Path
import json

import numpy as np
import torch
from pesq import pesq
from pystoi import stoi

from .audio_io import align_pair, load_audio_16k_mono
from .config import AUDIO, TRAIN
from .dataset import discover_voicebank_test_pairs
from .features import extract_features_and_targets, reconstruct_from_gains
from .model import RNNoiseTorch


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate RNNoise model on VoiceBank test set")
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--dataset-root", type=Path, default=TRAIN.dataset_root)
    p.add_argument("--max-items", type=int, default=0, help="0 means all test pairs")
    p.add_argument("--data-fraction", type=float, default=1.0, help="Use only a fraction of test pairs, e.g. 0.1")
    p.add_argument("--device", type=str, default="cpu")
    p.add_argument("--save-json", type=Path, default=Path("rnnoise_pytorch/outputs/test_metrics.json"))
    return p.parse_args()


def si_sdr(ref: np.ndarray, est: np.ndarray) -> float:
    ref, est = align_pair(ref, est)
    ref = ref.astype(np.float64)
    est = est.astype(np.float64)
    ref_energy = np.dot(ref, ref) + 1e-8
    scale = np.dot(est, ref) / ref_energy
    s_target = scale * ref
    e_noise = est - s_target
    return float(10 * np.log10((np.dot(s_target, s_target) + 1e-8) / (np.dot(e_noise, e_noise) + 1e-8)))


def enhance_with_model(model: RNNoiseTorch, noisy: np.ndarray, device: torch.device) -> np.ndarray:
    bundle = extract_features_and_targets(noisy=noisy, clean=noisy)
    x = torch.from_numpy(bundle.features)[None, ...].to(device)
    with torch.no_grad():
        gains, _ = model(x)
        gains = gains[0].cpu().numpy()
    return reconstruct_from_gains(bundle.noisy_mag, bundle.noisy_phase, gains)


def safe_pesq(sr: int, clean: np.ndarray, enh: np.ndarray) -> float:
    clean, enh = align_pair(clean, enh)
    try:
        return float(pesq(sr, clean, enh, "wb"))
    except Exception:
        return float("nan")


def safe_stoi(sr: int, clean: np.ndarray, enh: np.ndarray) -> float:
    clean, enh = align_pair(clean, enh)
    try:
        return float(stoi(clean, enh, sr, extended=False))
    except Exception:
        return float("nan")


def main():
    args = parse_args()
    device = torch.device(args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
    ckpt = torch.load(args.checkpoint, map_location=device)
    model = RNNoiseTorch(in_dim=42, bands=22).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    pairs = discover_voicebank_test_pairs(args.dataset_root)
    if args.max_items > 0:
        pairs = pairs[: args.max_items]
    frac = max(0.0, min(args.data_fraction, 1.0))
    if frac < 1.0:
        pairs = pairs[: max(1, int(len(pairs) * frac))]
    if len(pairs) == 0:
        raise FileNotFoundError(f"No test clean/noisy wav pairs found under {args.dataset_root}")
    print(f"eval_pairs={len(pairs)}")

    metrics = {
        "si_sdr_noisy": [],
        "si_sdr_enhanced": [],
        "pesq_noisy": [],
        "pesq_enhanced": [],
        "stoi_noisy": [],
        "stoi_enhanced": [],
    }

    for i, pair in enumerate(pairs, start=1):
        clean = load_audio_16k_mono(pair.clean, target_sr=AUDIO.sample_rate)
        noisy = load_audio_16k_mono(pair.noisy, target_sr=AUDIO.sample_rate)
        clean, noisy = align_pair(clean, noisy)
        enhanced = enhance_with_model(model, noisy, device)
        clean, enhanced = align_pair(clean, enhanced)

        metrics["si_sdr_noisy"].append(si_sdr(clean, noisy))
        metrics["si_sdr_enhanced"].append(si_sdr(clean, enhanced))
        metrics["pesq_noisy"].append(safe_pesq(AUDIO.sample_rate, clean, noisy))
        metrics["pesq_enhanced"].append(safe_pesq(AUDIO.sample_rate, clean, enhanced))
        metrics["stoi_noisy"].append(safe_stoi(AUDIO.sample_rate, clean, noisy))
        metrics["stoi_enhanced"].append(safe_stoi(AUDIO.sample_rate, clean, enhanced))
        if i % 50 == 0 or i == len(pairs):
            print(f"processed {i}/{len(pairs)}")

    summary = {
        "num_files": len(pairs),
        "SI_SDR": {
            "noisy": float(np.nanmean(metrics["si_sdr_noisy"])),
            "enhanced": float(np.nanmean(metrics["si_sdr_enhanced"])),
            "improvement": float(np.nanmean(metrics["si_sdr_enhanced"]) - np.nanmean(metrics["si_sdr_noisy"])),
        },
        "PESQ": {
            "noisy": float(np.nanmean(metrics["pesq_noisy"])),
            "enhanced": float(np.nanmean(metrics["pesq_enhanced"])),
            "improvement": float(np.nanmean(metrics["pesq_enhanced"]) - np.nanmean(metrics["pesq_noisy"])),
        },
        "STOI": {
            "noisy": float(np.nanmean(metrics["stoi_noisy"])),
            "enhanced": float(np.nanmean(metrics["stoi_enhanced"])),
            "improvement": float(np.nanmean(metrics["stoi_enhanced"]) - np.nanmean(metrics["stoi_noisy"])),
        },
    }

    args.save_json.parent.mkdir(parents=True, exist_ok=True)
    args.save_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"saved metrics: {args.save_json}")


if __name__ == "__main__":
    main()

