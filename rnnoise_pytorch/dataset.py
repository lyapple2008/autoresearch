from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence, Tuple
import random

import numpy as np
import torch
from torch.utils.data import Dataset

from .audio_io import align_pair, load_audio_16k_mono
from .config import TRAIN
from .features import extract_features_and_targets


@dataclass(frozen=True)
class PairItem:
    clean: Path
    noisy: Path


def _candidate_dirs(root: Path) -> List[Tuple[Path, Path]]:
    return [
        (root / "clean_trainset_28spk_wav", root / "noisy_trainset_28spk_wav"),
        (root / "clean_testset_wav", root / "noisy_testset_wav"),
        (root / "datasets" / "clean_trainset_28spk_wav", root / "datasets" / "noisy_trainset_28spk_wav"),
        (root / "datasets" / "clean_testset_wav", root / "datasets" / "noisy_testset_wav"),
    ]


def _discover_pairs_for_dir(clean_dir: Path, noisy_dir: Path) -> List[PairItem]:
    pairs: List[PairItem] = []
    if not clean_dir.exists() or not noisy_dir.exists():
        return pairs
    clean_map = {p.name: p for p in clean_dir.glob("*.wav")}
    for noisy_path in sorted(noisy_dir.glob("*.wav")):
        clean_path = clean_map.get(noisy_path.name)
        if clean_path is not None:
            pairs.append(PairItem(clean=clean_path, noisy=noisy_path))
    return pairs


def discover_voicebank_pairs(dataset_root: Path) -> List[PairItem]:
    pairs: List[PairItem] = []
    for clean_dir, noisy_dir in _candidate_dirs(dataset_root):
        pairs.extend(_discover_pairs_for_dir(clean_dir, noisy_dir))
    # Deduplicate if root has duplicated directory variants.
    uniq = {}
    for p in pairs:
        uniq[(p.clean.resolve(), p.noisy.resolve())] = p
    return list(uniq.values())


def discover_voicebank_train_pairs(dataset_root: Path) -> List[PairItem]:
    return discover_voicebank_pairs_in_split(dataset_root, split="train")


def discover_voicebank_test_pairs(dataset_root: Path) -> List[PairItem]:
    return discover_voicebank_pairs_in_split(dataset_root, split="test")


def discover_voicebank_pairs_in_split(dataset_root: Path, split: str) -> List[PairItem]:
    assert split in {"train", "test"}
    if split == "train":
        cands = [
            (dataset_root / "clean_trainset_28spk_wav", dataset_root / "noisy_trainset_28spk_wav"),
            (
                dataset_root / "datasets" / "clean_trainset_28spk_wav",
                dataset_root / "datasets" / "noisy_trainset_28spk_wav",
            ),
        ]
    else:
        cands = [
            (dataset_root / "clean_testset_wav", dataset_root / "noisy_testset_wav"),
            (dataset_root / "datasets" / "clean_testset_wav", dataset_root / "datasets" / "noisy_testset_wav"),
        ]
    pairs: List[PairItem] = []
    for clean_dir, noisy_dir in cands:
        pairs.extend(_discover_pairs_for_dir(clean_dir, noisy_dir))
    uniq = {}
    for p in pairs:
        uniq[(p.clean.resolve(), p.noisy.resolve())] = p
    return list(uniq.values())


def split_pairs(pairs: Sequence[PairItem], train_split: float, seed: int) -> Tuple[List[PairItem], List[PairItem]]:
    items = list(pairs)
    rng = random.Random(seed)
    rng.shuffle(items)
    cut = max(1, int(len(items) * train_split))
    return items[:cut], items[cut:] if cut < len(items) else items[:1]


class RNNoisePairDataset(Dataset):
    def __init__(self, pairs: Sequence[PairItem], chunk_frames: int = 200):
        self.pairs = list(pairs)
        self.chunk_frames = chunk_frames

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int):
        item = self.pairs[idx]
        clean = load_audio_16k_mono(item.clean)
        noisy = load_audio_16k_mono(item.noisy)
        clean, noisy = align_pair(clean, noisy)
        bundle = extract_features_and_targets(noisy=noisy, clean=clean)
        n = bundle.features.shape[0]
        if n <= self.chunk_frames:
            start = 0
            end = n
        else:
            start = np.random.randint(0, n - self.chunk_frames)
            end = start + self.chunk_frames
        x = torch.from_numpy(bundle.features[start:end])
        y_gain = torch.from_numpy(bundle.gains[start:end])
        y_vad = torch.from_numpy(bundle.vad[start:end])
        if x.shape[0] < self.chunk_frames:
            pad = self.chunk_frames - x.shape[0]
            x = torch.nn.functional.pad(x, (0, 0, 0, pad))
            y_gain = torch.nn.functional.pad(y_gain, (0, 0, 0, pad))
            y_vad = torch.nn.functional.pad(y_vad, (0, 0, 0, pad))
        return x.float(), y_gain.float(), y_vad.float()


def load_debug_pairs(max_items: int = TRAIN.max_items, seed: int = TRAIN.seed) -> Tuple[List[PairItem], List[PairItem]]:
    pairs = discover_voicebank_pairs(TRAIN.dataset_root)
    if len(pairs) == 0:
        raise FileNotFoundError(f"No clean/noisy wav pairs found under {TRAIN.dataset_root}")
    rng = random.Random(seed)
    rng.shuffle(pairs)
    pairs = pairs[: max_items] if max_items > 0 else pairs
    return split_pairs(pairs, TRAIN.train_split, seed)


def load_full_train_pairs(seed: int = TRAIN.seed, max_items: int = 0) -> Tuple[List[PairItem], List[PairItem]]:
    pairs = discover_voicebank_train_pairs(TRAIN.dataset_root)
    if len(pairs) == 0:
        raise FileNotFoundError(f"No train clean/noisy wav pairs found under {TRAIN.dataset_root}")
    rng = random.Random(seed)
    rng.shuffle(pairs)
    if max_items > 0:
        pairs = pairs[:max_items]
    return split_pairs(pairs, TRAIN.train_split, seed)

