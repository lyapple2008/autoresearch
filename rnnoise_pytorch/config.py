from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class AudioConfig:
    sample_rate: int = 16_000
    frame_size: int = 320
    hop_size: int = 160
    n_fft: int = 320
    num_bands: int = 22
    preemphasis: float = 0.85


@dataclass(frozen=True)
class FeatureConfig:
    feature_dim: int = 42
    pitch_min_hz: int = 60
    pitch_max_hz: int = 400
    vad_energy_threshold_db: float = -45.0


@dataclass(frozen=True)
class TrainConfig:
    dataset_root: Path = Path("/Volumes/tiger/Workspace/datasets/voicebank-demand/archive")
    output_dir: Path = Path("rnnoise_pytorch/outputs")
    batch_size: int = 8
    epochs: int = 3
    learning_rate: float = 3e-4
    weight_decay: float = 1e-6
    max_items: int = 50
    train_split: float = 0.9
    chunk_frames: int = 200
    seed: int = 42
    num_workers: int = 0
    device: str = "cpu"


AUDIO = AudioConfig()
FEATURE = FeatureConfig()
TRAIN = TrainConfig()

