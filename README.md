# CiproSeg

A PyTorch framework for binary cell segmentation using U-Net with Leave-One-Series-Out (LOSO) cross-validation. Built for reproducible biomedical microscopy experiments on a single GPU.

## What it does

CiproSeg trains a U-Net on time-series microscopy data, holding out one series at a time for testing. Each fold gets its own checkpoint, validation curves, and test metrics. At the end you have a per-fold breakdown of segmentation performance, validation-selected thresholds, and final held-out test metrics across the entire dataset with no test-set threshold tuning, deterministic seeds, and one config file.

## Installation

```bash
git clone https://github.com/manazucd/CiproSeg.git
cd CiproSeg
pip install -e .
```

For development tools (pytest, ruff, mypy):
```bash
pip install -e ".[dev]"
```

For the web-based GUI:
```bash
pip install -e ".[gui]"
```

Requires Python 3.9+ and PyTorch 2.2+. A CUDA GPU is recommended for training but not required.

## Quick start

Verify the installation:

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

Everything is controlled through a single YAML file. Validation thresholds are kept in their own `validation` section so checkpoint selection and post-training threshold selection are explicit and reproducible.

```yaml
dataset:
  name: ciproseg_ecoli
  root: /path/to/your_data
  train_frac: 0.8
  patch: 384
  stride: 192

model:
  name: unet4
  in_channels: 1
  out_channels: 1
  base_channels: 64
  dropout_p: 0.0
  norm: batch

trainer:
  epochs: 500
  batch_size: 4
  num_workers: 4
  seed: 1234
  target_patches_per_epoch: 1500
  deterministic: false
  benchmark: true
  cpu: false
  output_dir: runs/config_A
  amp: true
  early_stop_patience: 100
  test_tta: false
  compute_object_metrics: true

validation:
  checkpoint_threshold: 0.5
  threshold_sweep: true
  sweeping_thresholds: [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60]

loss:
  name: dice_bce
  dice_weight: 0.5
  bce_weight: 0.5

optim:
  name: adamw
  lr: 2.0e-4
  weight_decay: 1.0e-2

sched:
  name: cosine
  tmax: 500
  warmup_epochs: 10

logger:
  name: ciproseg
  level: INFO
```

`validation.checkpoint_threshold` is used during periodic validation while training (currently every 10 epochs) to compute validation metrics and decide whether the current model becomes the new `best.pt`. After training finishes for that fold, the best checkpoint is loaded again and the thresholds listed in `validation.sweeping_thresholds` are evaluated on the **validation set only**. The threshold with the highest mean validation Dice is selected for the final held-out test evaluation.

## Training

```bash
ciproseg-train --config configs/experiment.yaml
```

This builds a LOSO cross-validation loop: for N series, N folds run sequentially. In each fold, one entire series is held out as the test series. The remaining series are split into training and validation frames by temporal order using `dataset.train_frac`.

During training, validation is run periodically (currently every 10 epochs) using `validation.checkpoint_threshold`. Validation Dice determines which checkpoint is stored as `best.pt`. After training for the fold is complete, CiproSeg reloads `best.pt`, sweeps the candidate thresholds on the **validation set**, and selects the threshold with the highest mean validation Dice. That selected threshold is then applied once to the held-out test series. The test set is never used to choose the threshold.

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

### `run_summary.json`

`run_summary.json` is the easiest file to inspect after a complete LOSO run. It contains the most useful cross-fold results and, importantly, the validation-selected threshold for each fold.

Important top-level fields include:

| Field | Meaning |
|---|---|
| `best_validation_thresholds` | Compact list of the selected threshold for each fold. This is the quickest place to look before inference. |
| `best_validation_dice_across_folds` | Highest best-checkpoint validation Dice observed among the folds. |
| `mean_test_metrics_across_folds` | Mean final test Dice, IoU, Precision, Recall, and any other aggregated test metrics across folds. |
| `fold_results` | Detailed record for every LOSO fold. |
| `cfg` | Resolved configuration used for the run. |

A compact threshold entry looks like:

```json
{
  "fold_id": 1,
  "held_out_series": "controlnonsilica",
  "threshold": 0.45
}
```

Inside each item of `fold_results`, the `threshold_selection` object records how the threshold was chosen:

```json
{
  "threshold_selection": {
    "selection_set": "validation",
    "selection_metric": "dice",
    "selected_threshold": 0.45,
    "selected_metric_value": 0.89,
    "candidate_thresholds": [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60],
    "metrics_by_threshold": {
      "...": "Dice, IoU, Precision and Recall for each candidate threshold"
    }
  }
}
```

The key rule is:

> **Use the threshold selected on the validation set for the same fold/checkpoint. Do not choose a threshold by looking at the test results.**

For example, if `fold_01_test_controlnonsilica/best.pt` corresponds to a `best_validation_thresholds` entry with threshold `0.45`, use `0.45` when evaluating or running inference with that checkpoint.

## Evaluation

Point `ciproseg-eval` at any checkpoint. Everything it needs (model architecture, weights, dataset root, fold identity) is stored inside the `.pt` file.

```bash
ciproseg-eval \
  --ckpt runs/unet4/2026-04-03_153000/fold_01_test_seriesA/best.pt \
  --thr 0.45
```

Use the threshold selected for that same fold in `run_summary.json`. If you intentionally omit `--thr`, check the CLI/default behaviour rather than assuming that `0.5` is the validation-selected threshold.

Output:

```
------------------------------------------------------------
  Checkpoint : runs/.../fold_01_test_seriesA/best.pt
  Fold       : fold_01_test_seriesA
  Test series: seriesA
------------------------------------------------------------
  Dice       : 0.8234
  IoU        : 0.7012
  Precision  : 0.8456
  Recall     : 0.8019
  Loss (BCE) : 0.14523
------------------------------------------------------------
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

Run a trained model on a directory of images to produce probability maps, segmentation masks, and overlays. Use the **validation-selected threshold from the same fold**:

```bash
ciproseg-infer \
  --ckpt runs/unet4/2026-04-03_153000/fold_01_test_seriesA/best.pt \
  --input path/to/images/seriesA/ \
  --threshold 0.45
```

This creates for each input image (e.g. `frame_001.tif`):

| File | Description |
|---|---|
| `frame_001_prob.npy` | Authoritative float32 probability map with values in `[0.0, 1.0]` |
| `frame_001_prob.png` | 16-bit visualisation of the probability map (`0` to `65535`) |
| `frame_001_binary.png` | Final binary mask after applying the selected threshold (`0` or `255`) |
| `frame_001_pred_overlay.png` | Original image with predicted boundaries in green |

If you provide ground-truth masks with `--masks`, two additional files are generated:

| File | Description |
|---|---|
| `frame_001_gt_overlay.png` | Original image with ground-truth boundaries in cyan |
| `frame_001_diff.png` | TP (green) / FP (red) / FN (blue) / TN (dark) |

The three main prediction files have different purposes:

- `*_prob.npy` stores the original floating-point probabilities and is the numerical source of truth.
- `*_prob.png` is a convenient 16-bit image representation for visual inspection.
- `*_binary.png` is the thresholded segmentation mask. This is the file normally used for Connected Components, watershed, cell counting, and morphology measurements.

Saving `*_prob.npy` also means you can apply a different threshold later without rerunning the U-Net. For example, the same probability map can be converted to a new binary mask with threshold `0.45`, `0.50`, or another pre-specified value.

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

## Web GUI

CiproSeg ships with an optional browser-based interface for users who prefer not to work from the command line. The GUI wraps the same CLI commands described above; it does not replace them.

```bash
ciproseg-gui
```

This opens a browser tab at `http://127.0.0.1:7860` with pages for training configuration, live training monitoring, evaluation, inference with side-by-side image viewing, results browsing, and cross-fold comparison.

On launch, the GUI loads `configs/experiment.yaml` (if present) and populates the form fields automatically. All training, evaluation, and inference parameters can be edited directly in the browser without touching any YAML files.

A few other options:

```bash
ciproseg-gui --port 8080            # custom port
ciproseg-gui --no-browser           # skip auto-opening the browser
ciproseg-gui --host 0.0.0.0        # bind to all interfaces (remote access)
```

If a port is already in use (e.g. another GUI instance is running), the server automatically finds the next available port. Multiple instances can run side by side without manual port management.

### Multiple GUI instances

For example, if the default port is already occupied, a second CiproSeg GUI instance automatically moves to the next available port:

```bash
# First instance: normally http://127.0.0.1:7860
ciproseg-gui

# If 7860 is already in use, the next instance can use 7861
ciproseg-gui
```

You can still request a specific port explicitly with `--port`.

The GUI requires the `[gui]` optional dependencies (`fastapi`, `uvicorn`). If these are not installed, the core CLI tools remain fully functional.

## Configuration reference

The experiment config is the single source of truth for a CiproSeg training run. All sections are described below.

### `dataset`

| Field | Description |
|---|---|
| `name` | Dataset loader identifier. Currently only `ciproseg_ecoli` is supported. |
| `root` | Absolute path to the dataset directory containing image/mask pairs. |
| `train_frac` | Fraction of non-test frames used for training within each LOSO fold. The remainder becomes the validation set. |
| `patch` | Crop size (pixels) for training patches. Must be divisible by 16 to satisfy U-Net pooling requirements. |
| `stride` | Stride used during tiled inference on validation and test frames. Smaller values mean more overlap and smoother predictions but slower evaluation. |

### `model`

| Field | Description |
|---|---|
| `name` | Architecture variant. `unet2` uses 2 encoder/decoder levels (lighter); `unet4` uses 4 levels (standard, recommended). |
| `in_channels` | Number of input image channels. Set to `1` for grayscale microscopy. |
| `out_channels` | Number of output channels. Set to `1` for binary segmentation. |
| `base_channels` | Number of feature maps in the first encoder block. Doubles at each subsequent level. |

### `trainer`

| Field | Description |
|---|---|
| `epochs` | Total number of training epochs per fold. |
| `batch_size` | Number of patches per gradient update step. |
| `num_workers` | DataLoader worker processes for parallel data loading. Set to `0` to load on the main process (useful for debugging). |
| `seed` | Base random seed. Each fold derives its own seed as `seed + 1000 * fold_id` for reproducibility. |
| `target_patches_per_epoch` | Controls how many random crops are sampled per epoch across all training frames. The per-frame crop count is derived from this target. |
| `deterministic` | If `true`, enables PyTorch deterministic mode for full reproducibility at the cost of some speed. |
| `benchmark` | If `true`, enables `torch.backends.cudnn.benchmark` for faster training on fixed input sizes. Disable when input sizes vary. |
| `cpu` | Force training on CPU even when a GPU is available. Useful for debugging. |
| `output_dir` | Root directory for all training outputs. Each run creates a timestamped subdirectory containing fold checkpoints, curves, and the run summary. |
| `amp` | Enable automatic mixed precision during training. |
| `early_stop_patience` | Number of validation checks tolerated without improvement before early stopping. |
| `test_tta` | Enable test-time augmentation for the final full-image evaluation. |
| `compute_object_metrics` | Also compute object-level metrics during final test evaluation. |

### `validation`

| Field | Description |
|---|---|
| `checkpoint_threshold` | Fixed threshold used during periodic validation while training. It is used to compute validation metrics for checkpoint selection; it is not necessarily the final inference threshold. |
| `threshold_sweep` | If `true`, run a post-training threshold sweep on the validation set after loading the best checkpoint. |
| `sweeping_thresholds` | Candidate thresholds to evaluate on the validation set. Any non-empty list of values strictly between `0` and `1` is allowed. The threshold with the highest mean validation Dice is selected. |

### `optim`

| Field | Description |
|---|---|
| `name` | Optimizer name. `adamw` is the default and recommended choice. |
| `lr` | Initial learning rate. |
| `weight_decay` | L2 regularization coefficient applied to all non-bias parameters. |

### `sched`

| Field | Description |
|---|---|
| `name` | Learning rate scheduler. Set to `none` to keep the learning rate constant. Supported alternatives include `cosine` (cosine annealing). |
| `tmax` | (Cosine only) Number of epochs over which the learning rate anneals from `lr` to zero. |

### `logger`

| Field | Description |
|---|---|
| `name` | Logger identifier used in log output. |
| `level` | Logging verbosity. Standard Python logging levels apply: `DEBUG`, `INFO`, `WARNING`, `ERROR`. |

## SLURM

Job scripts are provided in `scripts/` for HPC clusters:

```bash
sbatch scripts/slurm_gpu.sh    # GPU partition
sbatch scripts/slurm_cpu.sh    # CPU-only partition
```

Submit from the project root or the `scripts/` directory; the scripts auto-detect the project location. Training output goes to `train.log` inside the timestamped run directory. SLURM's own `slurm-<jobid>.out` catches system-level messages (driver errors, OOM kills).

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

**Trainer** owns the full LOSO loop: builds the dataset index, creates N folds, trains each fold with periodic validation, selects checkpoints by validation Dice using `validation.checkpoint_threshold`, reloads the best checkpoint, sweeps `validation.sweeping_thresholds` on the validation set, selects the best-Dice threshold, and finally evaluates the held-out test series once with that selected threshold. Each fold gets a deterministic seed derived from the base seed.

**Evaluator** (`evaluate_checkpoint`) is a standalone function. It loads a checkpoint, rebuilds the model from the config stored inside it, identifies the held-out test series from the fold name, and runs tiled inference. No config file needed; the checkpoint is self-contained.

**Checkpointer** handles atomic file writes (temp file + `os.replace`) for both fold-level checkpoints and the run summary. JSON sidecars carry metadata without requiring `torch.load` to browse results.

**Tiled inference** handles images larger than the patch size by splitting them into overlapping tiles, running each through the model, and averaging predictions in overlap regions. Padding is applied automatically for images smaller than the patch.

## Design choices

- **Single GPU only.** No DDP, no torchrun, no DeepSpeed. This keeps the code simple and the experiments reproducible. If you need multi-GPU, this is not the right tool.
- **No mid-fold resume.** Checkpoints are saved per fold, but training always restarts from fold 1. LOSO folds are fast enough on a single GPU that this is rarely a problem.
- **Self-contained checkpoints.** Every `.pt` file stores the full config, model weights, optimizer state, and fold metadata. You can evaluate or run inference months later without hunting for the original config file.
- **Timestamp-based run directories.** Each training run gets a unique directory (`runs/model/YYYY-MM-DD_HHMMSS/`). Nothing is ever overwritten.
- **Periodic validation.** Validation is run every 10 epochs (and at the final epoch) to limit overhead while still tracking checkpoint quality.
- **Validation-only threshold selection.** Candidate probability thresholds are swept only on the validation set after fold training. The held-out test series is evaluated only after the threshold has been fixed.
- **Decoupled GUI.** The web interface is entirely optional and calls the CLI tools as subprocesses. If the GUI is not installed, corrupted, or otherwise unavailable, all core functionality works from the command line exactly as documented above.

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

## AI assistance

This README is created with the assistence of Claude (Anthropic). The authors reviewed and edited them and take responsibility for their content. Parts of the code were written with Claude; the authors reviewed and tested all of it.