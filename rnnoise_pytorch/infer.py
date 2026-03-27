from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

from .audio_io import load_audio_16k_mono
from .config import AUDIO
from .features import extract_features_and_targets, reconstruct_from_gains
from .model import RNNoiseTorch


def parse_args():
    p = argparse.ArgumentParser(description="RNNoise-style PyTorch inference")
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--device", type=str, default="cpu")
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))

    ckpt = torch.load(args.checkpoint, map_location=device)
    model = RNNoiseTorch(in_dim=42, bands=22).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    noisy = load_audio_16k_mono(args.input, target_sr=AUDIO.sample_rate)
    # Targets are ignored in inference; pass noisy as placeholder clean.
    bundle = extract_features_and_targets(noisy=noisy, clean=noisy)
    x = torch.from_numpy(bundle.features)[None, ...].to(device)

    with torch.no_grad():
        gains, _ = model(x)
        gains = gains[0].cpu().numpy()

    enhanced = reconstruct_from_gains(bundle.noisy_mag, bundle.noisy_phase, gains)
    enhanced = np.clip(enhanced, -1.0, 1.0)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(args.output), enhanced, AUDIO.sample_rate)
    print(f"saved enhanced wav: {args.output}")


if __name__ == "__main__":
    main()

