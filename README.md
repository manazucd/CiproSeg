# CiproSeg

A modular, research grade PyTorch framework for **LOSO cross validation cell segmentation** using U-Net.

Designed for:

- Reproducible biomedical image segmentation
- Leave-One-Series-Out (LOSO) evaluation
- Clean experiment configuration
- Single GPU training (no DDP)
- Lightweight checkpointing & metrics tracking

# CiproSeg Installation Guide

## Quick Install

### From PyPI (when published)
```bash
pip install ciproseg
```

### From Source (Development)

#### 1. Clone the repository
```bash
git clone https://github.com/manazucd/CiproSeg.git
cd CiproSeg
```

#### 2. Install in development mode
```bash
# Basic installation
pip install -e .

# With development tools
pip install -e ".[dev]"
```

## Requirements

- Python 3.9 or higher
- CUDA-capable GPU (recommended for training)

## Dependencies

The package automatically installs:
- **PyTorch** (>=2.2): Deep learning framework
- **NumPy** (>=1.23): Numerical computations
- **OpenCV** (>=4.8): Image processing
- **tifffile** (>=2023.7.0): TIFF file handling
- **matplotlib** (>=3.7.0): Visualization
- **PyYAML** (>=6.0): Configuration files
- **tqdm** (>=4.66): Progress bars

## Verify Installation

After installation, verify by running:

```bash
# Check if CLI commands are available
ciproseg-train --help
ciproseg-eval --help

# Test import in Python
python -c "import ciproseg; print(f'CiproSeg version: {ciproseg.__version__}')"
```

## Using the Command-Line Interface

### Training
```bash
ciproseg-train --config path/to/config.yaml
```

### Evaluation
```bash
ciproseg-eval --config path/to/config.yaml --checkpoint path/to/checkpoint.pth
```

### CiproSeg Inference Guide
CiproSeg provides a command-line interface for running tiled full-image inference on a directory of microscopy images.

The model performs **pixel-wise semantic segmentation**, predicting for each pixel the probability that it belongs to a bacterial cell (foreground) versus background.

## Basic Inference (Prediction Only)

Run inference on a directory of images:

``` bash
ciproseg-infer \
  --ckpt runs/ciproseg_ecoli_unet/fold_01/best.pt \
  --input Data/miniture_4_debug/images/controlnonsilica/
```

This will:
  - Load the trained checkpoint
  - Run tiled inference on all images in the input directory
  - Create an output directory next to the input folder
  - Generate prediction artefacts for each image

## Inference with Ground Truth (Evaluation Visualisation)

If ground-truth masks are available, provide the `--masks` directory:

``` bash
ciproseg-infer \
  --ckpt runs/ciproseg_ecoli_unet/fold_01/best.pt \
  --input Data/miniture_4_debug/images/controlnonsilica/ \
  --masks Data/miniture_4_debug/masks/controlnonsilica/ \
  --output /data/predictions/ \
  --threshold 0.45 \
  --patch 512 --stride 256
```

> **Important:**\
> Ground-truth mask filenames must match image filenames (same stem).

Example:
    images/controlnonsilica/01.tif
    masks/controlnonsilica/01.tif

# Generated Output Files
For each input image `01.tif`, the following PNG files are created:

## 1 `01_prob.png` --- Probability Map (16-bit)

Represents the raw model output after sigmoid activation.
-   Pixel-wise probability in range `[0, 1]`
-   Stored as 16-bit PNG for precision

Interpretation:
-   Dark -> high confidence background
-   Bright -> high confidence cell
-   Mid-gray -> uncertain region

## 2 `01_binary.png` --- Binary Segmentation Mask

Thresholded probability map.
-   255 -> Cell (foreground)
-   0 -> Background

This is the final segmentation used for quantitative analysis.

## 3 `01_pred_overlay.png` --- Prediction Boundary Overlay
-   Original grayscale microscopy image
-   Predicted cell boundaries drawn in **green**

The model predicts full cell regions. Only the contour (outline) is drawn so that biological texture remains
visible.

## 4 `01_gt_overlay.png` *(only if `--masks` provided)*
-   Original image
-   Ground-truth boundaries drawn in **cyan**

## 5 `01_diff.png` *(only if `--masks` provided)*

Color coded comparison between prediction and ground truth:
  - Green -> True Positive (correct cell pixel)
  - Red -> False Positive (model over segmentation)
  - Blue -> False Negative (model missed cell pixel)
  - Dark -> True Negative (correct background)

This is useful for diagnosing model behavior.

# Notes
  - Inference is tiled internally to support large images.
  - Images smaller than patch size are automatically padded.
  - Non divisible dimensions are handled via overlap averaging.
  - Output files are written atomically for robustness.

## Common Installation Issues

### Issue: CUDA not available
**Solution**: Install PyTorch with CUDA support:
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

### Issue: OpenCV conflicts
**Solution**: Uninstall conflicting opencv packages:
```bash
pip uninstall opencv-python opencv-python-headless opencv-contrib-python -y
pip install opencv-python>=4.8
```

### Issue: Permission denied
**Solution**: Install in user space:
```bash
pip install --user -e .
```

## For Contributors

Install development dependencies:
```bash
pip install -e ".[dev]"
```

This includes:
- **pytest**: Testing framework
- **ruff**: Fast Python linter
- **mypy**: Static type checker

Run tests:
```bash
pytest tests/
```

Run linting:
```bash
ruff check src/
```

## Uninstall

```bash
pip uninstall ciproseg
```

## Training

Run training with a YAML config:

```bash
ciproseg-train --config configs/experiment.yaml
```

or:

```bash
python -m ciproseg.cli.train_cli --config configs/experiment.yaml
```

### What happens during training?

- Dataset is indexed from:

  ```
  dataset_root/
    images/<series>/*.png
    masks/<series>/*.png
  ```
- Leave-One-Series-Out (LOSO) folds are created
- For each fold:
  - Train on N-1 series
  - Validate on split of training series
  - Test on held out series
- Best validation Dice is tracked per fold
- Mean test metrics across folds are reported

## Evaluation

```bash
ciproseg-eval --config configs/experiment.yaml
```

This runs evaluation using the configured dataset and prints summary metrics.

## Project Structure
In order to print the project structure, please run the following command from the project home directory
```bash
python3 tools/print_tree.py 
```

## Checkpointing

This project uses **run-level checkpointing**, not per-epoch resume.

After training, a run summary is saved:

```
runs/
  run_summary.pt
  run_summary.json
```

Saved information:

* Best validation metric (global across folds)
* Mean test metrics
* Full experiment config

This design keeps LOSO training simple and reproducible.

## Tests

Run tests with:

```bash
pytest
```

### Test Types

**Unit tests (white-box)**

* `test_dataset.py`
* `test_model.py`
* `test_loss.py`
* `test_checkpoint.py`

These validate individual components in isolation.

**Integration / smoke test (black-box-ish)**

* `test_engine.py`

Validates that `Engine.fit()` and evaluation wiring work correctly without running heavy training.

---

## Configuration

All experiments are controlled via YAML:

```yaml
dataset:
  root: /path/to/data
  patch: 256
  stride: 128

trainer:
  epochs: 40
  batch_size: 8
  seed: 1234

optim:
  name: adamw
  lr: 1e-4

sched:
  name: cosine
  interval: epoch
```

## Hardware
- Single GPU or CPU
- No Distributed Data Parallel (DDP)
- No torchrun required
- No DeepSpeed/ZeRO

## Design Principles
- Engine orchestrates only
- Trainer contains task logic (LOSO)
- No data leakage between folds
- Reproducible seeds per fold
- Clean separation of concerns

## Intended Use
- Biomedical image segmentation research
- Cell segmentation
- LOSO cross-validation experiments
- Reproducible academic studies
