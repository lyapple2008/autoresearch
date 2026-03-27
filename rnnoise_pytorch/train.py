from __future__ import annotations

import argparse
from pathlib import Path
import random

import numpy as np
import torch
from torch.utils.data import DataLoader

from .config import TRAIN
from .dataset import RNNoisePairDataset, load_debug_pairs, load_full_train_pairs
from .losses import rnnoise_total_loss
from .model import RNNoiseTorch


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def parse_args():
    p = argparse.ArgumentParser(description="Train RNNoise-style PyTorch model")
    p.add_argument("--max-items", type=int, default=0, help="0 means use all pairs for selected mode")
    p.add_argument("--mode", type=str, default="full", choices=["debug", "full"])
    p.add_argument("--data-fraction", type=float, default=1.0, help="Use only a fraction of selected data, e.g. 0.1")
    p.add_argument("--epochs", type=int, default=TRAIN.epochs)
    p.add_argument("--batch-size", type=int, default=TRAIN.batch_size)
    p.add_argument("--lr", type=float, default=TRAIN.learning_rate)
    p.add_argument("--device", type=str, default=TRAIN.device)
    p.add_argument("--output-dir", type=Path, default=TRAIN.output_dir)
    p.add_argument("--log-interval", type=int, default=20, help="Print training progress every N steps")
    return p.parse_args()


def evaluate(model, loader, device):
    model.eval()
    losses = []
    with torch.no_grad():
        total = len(loader)
        for i, (x, y_gain, y_vad) in enumerate(loader, start=1):
            x, y_gain, y_vad = x.to(device), y_gain.to(device), y_vad.to(device)
            gain_pred, vad_pred = model(x)
            loss = rnnoise_total_loss(y_gain, gain_pred, y_vad, vad_pred)
            losses.append(loss.item())
            if i % 20 == 0 or i == total:
                print(f"  [val] step {i}/{total} loss={loss.item():.6f}")
    return float(np.mean(losses)) if losses else float("nan")


def main():
    args = parse_args()
    seed_all(TRAIN.seed)
    device = torch.device(args.device if args.device != "auto" else ("cuda" if torch.cuda.is_available() else "cpu"))
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.mode == "debug":
        max_items = args.max_items if args.max_items > 0 else TRAIN.max_items
        train_pairs, val_pairs = load_debug_pairs(max_items=max_items, seed=TRAIN.seed)
    else:
        train_pairs, val_pairs = load_full_train_pairs(seed=TRAIN.seed, max_items=args.max_items)
    frac = max(0.0, min(args.data_fraction, 1.0))
    if frac < 1.0:
        train_pairs = train_pairs[: max(1, int(len(train_pairs) * frac))]
        val_pairs = val_pairs[: max(1, int(len(val_pairs) * frac))]
    print(f"train_pairs={len(train_pairs)} val_pairs={len(val_pairs)} mode={args.mode}")
    train_ds = RNNoisePairDataset(train_pairs, chunk_frames=TRAIN.chunk_frames)
    val_ds = RNNoisePairDataset(val_pairs, chunk_frames=TRAIN.chunk_frames)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=TRAIN.num_workers)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=TRAIN.num_workers)

    model = RNNoiseTorch(in_dim=42, bands=22).to(device)
    optim = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=TRAIN.weight_decay)

    best_val = float("inf")
    for epoch in range(1, args.epochs + 1):
        model.train()
        step_losses = []
        total_steps = len(train_loader)
        for step, (x, y_gain, y_vad) in enumerate(train_loader, start=1):
            x, y_gain, y_vad = x.to(device), y_gain.to(device), y_vad.to(device)
            gain_pred, vad_pred = model(x)
            loss = rnnoise_total_loss(y_gain, gain_pred, y_vad, vad_pred)
            optim.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optim.step()
            step_losses.append(loss.item())
            if step % args.log_interval == 0 or step == total_steps:
                print(
                    f"[train] epoch={epoch:03d} step={step:04d}/{total_steps} "
                    f"loss={loss.item():.6f}"
                )

        train_loss = float(np.mean(step_losses))
        val_loss = evaluate(model, val_loader, device)
        print(f"epoch={epoch:03d} train_loss={train_loss:.6f} val_loss={val_loss:.6f}")

        ckpt = {
            "model_state": model.state_dict(),
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
        }
        torch.save(ckpt, args.output_dir / "last.pt")
        if val_loss < best_val:
            best_val = val_loss
            torch.save(ckpt, args.output_dir / "best.pt")

    print(f"done. checkpoints at {args.output_dir}")


if __name__ == "__main__":
    main()

