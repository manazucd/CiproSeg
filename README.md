# CiproSeg

A PyTorch framework for binary cell segmentation using U-Net with Leave-One-Series-Out (LOSO) cross-validation. Built for reproducible biomedical microscopy experiments on a single GPU.

## What it does

CiproSeg trains a U-Net on time-series microscopy data, holding out one series at a time for testing. Each fold gets its own checkpoint, validation curves, and test metrics. At the end you have a per-fold breakdown of segmentation performance across the entire dataset with no data leakage, deterministic seeds, one config file.

## Installation

```bash
git clone https://github.com/manazucd/CiproSeg.git
cd CiproSeg
pip install -e .
```

For development (adds pytest, ruff, mypy):

```bash
pip install -e ".[dev]"
```

Requires Python 3.9+ and PyTorch 2.2+. A CUDA GPU is recommended for training but not required.

Verify it works:

```bash
ciproseg-train --help
ciproseg-eval --help
ciproseg-infer --help
```

## Dataset layout

CiproSeg expects images and masks in matched directory trees. Each subdirectory under `images/` and `masks/` is a "series", one series is held out per LOSO fold.

```
your_data/
  images/
    series_A/
      frame_000.tif
      frame_001.tif
      ...
    series_B/
      ...
  masks/
    series_A/
      frame_000.tif
      frame_001.tif
      ...
    series_B/
      ...
```

Filenames must match exactly between `images/<series>/` and `masks/<series>/`. Supported formats: `.tif`, `.tiff`, `.png`, `.jpg`, `.jpeg`. Masks should be binary (any nonzero pixel = foreground).

## Configuration

Everything is controlled through a single YAML file:

```yaml
dataset:
  name: ciproseg_ecoli
  root: /path/to/your_data
  train_frac: 0.8
  patch: 256
  stride: 128

model:
  name: unet4          # unet2 (smaller) or unet4 (larger)
  in_channels: 1
  out_channels: 1
  base_channels: 64

trainer:
  epochs: 500
  batch_size: 2
  num_workers: 4
  seed: 1234
  thr: 0.5
  target_patches_per_epoch: 1000
  deterministic: false
  benchmark: true
  cpu: false
  output_dir: runs

optim:
  name: adamw
  lr: 1.0e-4
  weight_decay: 1.0e-4

sched:
  name: cosine          # none, cosine, step, onecycle
  tmax: 250

logger:
  name: ciproseg
  level: INFO
```

## Training

```bash
ciproseg-train --config configs/experiment.yaml
```

This builds a LOSO cross-validation loop: for N series, N folds run sequentially. Each fold trains on N−1 series (split 80/20 into train/val by time), then evaluates on the held-out series. Best checkpoints are selected by validation Dice.

Override seed or force CPU from the command line:

```bash
ciproseg-train --config configs/experiment.yaml --seed 42
ciproseg-train --config configs/experiment.yaml --cpu
```

### Output structure

Each run gets a unique timestamped directory:

```
runs/unet4/2026-04-03_153000/
  config.yaml
  config_resolved.yaml
  train.log
  run_summary.pt
  run_summary.json
  fold_01_test_seriesA/
    best.pt
    best.json
    last.pt
    last.json
    loss.png
    dice.png
  fold_02_test_seriesB/
    ...
```

`config.yaml` is the experiment config as passed in. `config_resolved.yaml` includes all runtime overrides and the resolved output path. `train.log` contains the full training log (epoch losses, validation metrics, fold summaries). The fold directories each contain the best and latest checkpoints, JSON sidecars with metadata, and training curve plots.

## Evaluation

Point `ciproseg-eval` at any checkpoint. Everything it needs (model architecture, weights, dataset root, fold identity) is stored inside the `.pt` file.

```bash
ciproseg-eval --ckpt runs/unet4/2026-04-03_153000/fold_01_test_seriesA/best.pt
```

Output:

```
────────────────────────────────────────────────────────────
  Checkpoint : runs/.../fold_01_test_seriesA/best.pt
  Fold       : fold_01_test_seriesA
  Test series: seriesA
────────────────────────────────────────────────────────────
  Dice       : 0.8234
  IoU        : 0.7012
  Precision  : 0.8456
  Recall     : 0.8019
  Loss (BCE) : 0.14523
────────────────────────────────────────────────────────────
```

Useful flags:

```bash
# Save metrics to a file
ciproseg-eval --ckpt .../best.pt --json metrics.json

# Force CPU
ciproseg-eval --ckpt .../best.pt --cpu

# Override dataset path (if data moved since training)
ciproseg-eval --ckpt .../best.pt --dataset-root /new/path/to/data

# Different tiling parameters
ciproseg-eval --ckpt .../best.pt --patch 512 --stride 256 --thr 0.45
```

## Inference

Run a trained model on a directory of images to produce segmentation masks and overlays:

```bash
ciproseg-infer \
  --ckpt runs/unet4/2026-04-03_153000/fold_01_test_seriesA/best.pt \
  --input path/to/images/seriesA/
```

This creates for each input image (e.g. `frame_001.tif`):

| File | Description |
|---|---|
| `frame_001_prob.png` | 16-bit probability map (0 = background, 65535 = cell) |
| `frame_001_binary.png` | Binary mask after thresholding (0 or 255) |
| `frame_001_pred_overlay.png` | Original image with predicted boundaries in green |

If you provide ground-truth masks with `--masks`, two additional files are generated:

| File | Description |
|---|---|
| `frame_001_gt_overlay.png` | Original image with ground-truth boundaries in cyan |
| `frame_001_diff.png` | TP (green) / FP (red) / FN (blue) / TN (dark) |

Full example with masks and custom parameters:

```bash
ciproseg-infer \
  --ckpt .../best.pt \
  --input path/to/images/seriesA/ \
  --masks path/to/masks/seriesA/ \
  --output predictions/seriesA/ \
  --threshold 0.45 \
  --patch 512 \
  --stride 256
```

Other options: `--cpu`, `--device cuda:1`, `--verbose`.

## SLURM

Job scripts are provided in `scripts/` for HPC clusters:

```bash
sbatch scripts/slurm_gpu.sh    # GPU partition
sbatch scripts/slurm_cpu.sh    # CPU-only partition
```

Submit from the project root or the `scripts/` directory, the scripts auto-detect the project location. Training output goes to `train.log` inside the timestamped run directory; SLURM's own `slurm-<jobid>.out` catches system-level messages (driver errors, OOM kills).

## Testing

```bash
pytest tests/
```

The test suite covers model forward passes, dataset indexing and splitting, loss computation, checkpointing, and training orchestration. For static analysis:

```bash
ruff check src/
mypy src/
```

## How it works

**Engine** orchestrates runs but contains no task logic. It lazily builds the Trainer on first access, so evaluation-only workflows never touch training code.

**Trainer** owns the full LOSO loop: builds the dataset index, creates N folds, trains each fold with per-epoch logging, selects checkpoints by validation Dice, and runs final test evaluation on the held-out series. Each fold gets a deterministic seed derived from the base seed.

**Evaluator** (`evaluate_checkpoint`) is a standalone function. It loads a checkpoint, rebuilds the model from the config stored inside it, identifies the held-out test series from the fold name, and runs tiled inference. No config file needed, the checkpoint is self-contained.

**Checkpointer** handles atomic file writes (temp file + `os.replace`) for both fold-level checkpoints and the run summary. JSON sidecars carry metadata without requiring `torch.load` to browse results.

**Tiled inference** handles images larger than the patch size by splitting them into overlapping tiles, running each through the model, and averaging predictions in overlap regions. Padding is applied automatically for images smaller than the patch.

## Design choices

- **Single GPU only.** No DDP, no torchrun, no DeepSpeed. This keeps the code simple and the experiments reproducible. If you need multi-GPU, this isn't the right tool.
- **No mid-fold resume.** Checkpoints are saved per fold, but training always restarts from fold 1. LOSO folds are fast enough on a single GPU that this is rarely a problem.
- **Self-contained checkpoints.** Every `.pt` file stores the full config, model weights, optimizer state, and fold metadata. You can evaluate or run inference months later without hunting for the original config file.
- **Timestamp-based run directories.** Each training run gets a unique directory (`runs/model/YYYY-MM-DD_HHMMSS/`). Nothing is ever overwritten.

## Citation

If you use CiproSeg in your work, please cite:

```bibtex
@software{ciproseg2026,
  title   = {CiproSeg: LOSO U-Net Segmentation Framework},
  year    = {2026},
  url     = {https://github.com/manazucd/CiproSeg}
}
```

## License

Apache-2.0. See [LICENSE](LICENSE) for details.
