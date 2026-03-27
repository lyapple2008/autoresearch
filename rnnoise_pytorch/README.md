# RNNoise PyTorch (16k/mono)

This directory contains a PyTorch implementation of an RNNoise-style model with:

- Python feature extraction (no C frontend)
- 16kHz mono training/inference path
- VoiceBank-DEMAND pairing logic
- Debug-first train/infer scripts

## Environment

```bash
conda activate autoresearch
python3 -m pip install -U pip
python3 -m pip install torch numpy scipy soundfile pesq pystoi
```

## Dataset

Expected root:

`/Volumes/tiger/Workspace/datasets/voicebank-demand/archive`

Supported folder patterns include:

- `clean_trainset_28spk_wav` + `noisy_trainset_28spk_wav`
- `clean_testset_wav` + `noisy_testset_wav`
- the same folders under an extra `datasets/` subdirectory

## Train (full train set)

From repo root:

```bash
python3 -m rnnoise_pytorch.train \
  --mode full \
  --max-items 0 \
  --epochs 3 \
  --batch-size 8 \
  --device cpu
```

Checkpoints are written to `rnnoise_pytorch/outputs`.

## Evaluate on full test set (SI-SDR / PESQ / STOI)

```bash
python3 -m rnnoise_pytorch.evaluate \
  --checkpoint rnnoise_pytorch/outputs/best.pt \
  --max-items 0 \
  --device cpu
```

This prints and saves metrics to `rnnoise_pytorch/outputs/test_metrics.json`.

## Inference

```bash
python3 -m rnnoise_pytorch.infer \
  --checkpoint rnnoise_pytorch/outputs/best.pt \
  --input /path/to/noisy.wav \
  --output rnnoise_pytorch/outputs/enhanced.wav \
  --device cpu
```

## Debug-only quick run

```bash
python3 -m rnnoise_pytorch.train \
  --mode debug \
  --max-items 50 \
  --epochs 1 \
  --batch-size 4 \
  --device cpu
```

