#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SPDX-License-Identifier: Apache-2.0
# =======================================================================
#  Project       : CiproSeg - LOSO U-Net Segmentation Framework
#  File          : src/ciproseg/core/evaluator.py
#  Purpose       : Evaluator
#
#  Copyright (c) 2026 Manaz Kaleel
#  Authors       : Manaz Kaleel <manaz.ucd@gmail.com>
#
#  Created on    : 2025-11-01
#  Last modified : 2026-10-03, MK
#  Version       : 0.1.2
#  License       : Apache-2.0
#  Repository    : https://github.com/manazucd/CiproSeg
# =======================================================================
from __future__ import annotations

import os
import re
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn

from ..data.datasets.ciproseg_ecoli import (
    FrameItem,
    build_index,
    read_image_gray,
    read_mask_01,
)
from ..metrics.segmentation import evaluate_full_image_numpy
from ..metrics.objects import compute_object_metrics, aggregate_object_metrics
from ..models import build_model_from_cfg
from ..utils.logger import get_logger
from .hooks import HookList




# Mask binarisation threshold. Project convention: mask pixel values in
# 1..50 are treated as background (CiproSeg quirk). 128 is the safe cutoff.
_MASK_BIN_THRESHOLD = 128

# Floor for the Hann blending weights. Keeps pixels at the very edge of
# the image (where only one tile contributes and the window value is 0)
# from being divided by zero, without affecting interior blending.
_HANN_MIN_WEIGHT = 1e-3

# Boundary padding mode used to extend the image up to a tile-grid-aligned
# size before tiled inference. "reflect" matches BORDER_REFLECT_101 used
# elsewhere in the pipeline (mirrors without repeating the edge pixel).
_BOUNDARY_PAD_MODE = "reflect"


def tile_positions(h: int, w: int, patch: int, stride: int) -> List[Tuple[int, int]]:
    """Deterministic (x, y) top-left tile positions that fully cover the image."""
    ys = list(range(0, max(1, h - patch + 1), stride))
    xs = list(range(0, max(1, w - patch + 1), stride))
    if ys[-1] != max(0, h - patch):
        ys.append(max(0, h - patch))
    if xs[-1] != max(0, w - patch):
        xs.append(max(0, w - patch))
    return [(x, y) for y in ys for x in xs]


def _hann_window_2d(patch: int, min_weight: float = _HANN_MIN_WEIGHT) -> np.ndarray:
    """
    Build a 2D Hann window for tile blending.

    The outer product of a 1D Hann window with itself, clamped to a small
    positive value so corner pixels at the image boundary still receive a
    non-zero contribution after the final accumulator division.

    With stride = patch / 2 (the project default), a 2D Hann window sums
    to a constant across the interior of the image, so seams are smoothed
    without changing the overall probability scale.
    """
    if patch < 2:
        raise ValueError(f"patch must be >= 2 for Hann blending, got {patch}")
    win_1d = np.hanning(patch).astype(np.float32)
    win_2d = np.outer(win_1d, win_1d)
    return np.maximum(win_2d, np.float32(min_weight))


def _grid_aligned_size(size: int, patch: int, stride: int) -> int:
    """
    Smallest size >= `size` such that (size_aligned - patch) is a
    non-negative multiple of stride. Padding the image up to this
    aligned size puts every tile position on the regular stride grid,
    which is the condition under which the Hann window constant-sum
    property holds across the whole image (no residual seam at the
    bottom or right edge from an irregular last-tile offset).
    """
    if size <= patch:
        return patch
    extra = size - patch
    n_strides = (extra + stride - 1) // stride
    return patch + n_strides * stride


@torch.no_grad()
def predict_full_mask_tiled(
    model: nn.Module,
    img_gray01: np.ndarray,
    patch: int = 256,
    stride: int = 128,
    device: str = "cuda",
) -> np.ndarray:
    """
    Full image tiled inference with Hann-windowed overlap blending.

    Steps:
    1. Validate input shape and tiling parameters
    2. Reflect-pad the image up to a tile-grid-aligned size
    3. Run Hann-windowed tiled inference on the padded image
    4. Crop the result back to the original image shape

    Expects a 2D float array in [0, 1]; returns a probability map of
    the same shape.

    Why padding to a grid-aligned size matters. The Hann window only
    sums to a constant in the overlap region when adjacent tiles are
    offset by exactly `stride`. If the image dimension is not of the
    form patch + k * stride, tile_positions appends one extra tile at
    an irregular offset to cover the boundary. That irregular tile
    breaks the constant-sum guarantee and leaves a faint residual band
    aligned with the irregular tile edge. Padding to a grid-aligned
    size removes the irregular tile entirely.
    """
    if img_gray01.ndim != 2:
        raise ValueError(f"img_gray01 must be 2D (H,W), got {img_gray01.shape}")
    if patch < 2:
        raise ValueError(f"patch must be >= 2, got {patch}")
    if stride <= 0 or stride > patch:
        raise ValueError(
            f"stride must satisfy 0 < stride <= patch; got stride={stride}, patch={patch}"
        )

    model.eval()
    h, w = img_gray01.shape

    # Pad to a tile-grid-aligned size with reflection at the boundary.
    h_pad = _grid_aligned_size(h, patch, stride)
    w_pad = _grid_aligned_size(w, patch, stride)
    img_padded = np.pad(
        img_gray01,
        ((0, h_pad - h), (0, w_pad - w)),
        mode=_BOUNDARY_PAD_MODE,
    )

    acc = np.zeros((h_pad, w_pad), dtype=np.float32)
    wgt = np.zeros((h_pad, w_pad), dtype=np.float32)
    window = _hann_window_2d(patch)

    # After padding, every tile crop is exactly (patch, patch) and every
    # tile position is on the regular stride grid. The Hann constant-sum
    # property holds across the whole padded image.
    for x, y in tile_positions(h_pad, w_pad, patch, stride):
        crop = img_padded[y:y + patch, x:x + patch]
        inp = torch.from_numpy(crop.astype(np.float32)).unsqueeze(0).unsqueeze(0).to(device)
        prob = torch.sigmoid(model(inp)).squeeze().detach().cpu().numpy().astype(np.float32)

        acc[y:y + patch, x:x + patch] += prob * window
        wgt[y:y + patch, x:x + patch] += window

    # Crop the accumulator and weight buffers back to the original image
    # shape before the final division. Reflected boundary contributions
    # are discarded here.
    return acc[:h, :w] / np.maximum(wgt[:h, :w], 1e-8)


# Forward and inverse D4 group operations. Each forward op transforms
# the input image, each inverse op transforms a probability map back to
# the original orientation. Pairs are matched by index.
_TTA_FORWARD = (
    lambda a: a,
    lambda a: np.fliplr(a).copy(),
    lambda a: np.flipud(a).copy(),
    lambda a: np.flipud(np.fliplr(a)).copy(),
    lambda a: np.rot90(a, 1).copy(),
    lambda a: np.fliplr(np.rot90(a, 1)).copy(),
    lambda a: np.rot90(a, 2).copy(),
    lambda a: np.rot90(a, 3).copy(),
)
_TTA_INVERSE = (
    lambda a: a,
    lambda a: np.fliplr(a).copy(),
    lambda a: np.flipud(a).copy(),
    lambda a: np.fliplr(np.flipud(a)).copy(),
    lambda a: np.rot90(a, -1).copy(),
    lambda a: np.rot90(np.fliplr(a), -1).copy(),
    lambda a: np.rot90(a, -2).copy(),
    lambda a: np.rot90(a, -3).copy(),
)


@torch.no_grad()
def predict_full_mask_tta(
    model: nn.Module,
    img_gray01: np.ndarray,
    patch: int = 256,
    stride: int = 128,
    device: str = "cuda",
) -> np.ndarray:
    """
    Test-time augmentation over the D4 symmetry group.

    Runs tiled inference 8 times, once for each (rotation x flip)
    transform, inverts the transform on each output probability map,
    and averages the 8 maps. Output shape matches the input.
    """
    if img_gray01.ndim != 2:
        raise ValueError(f"img_gray01 must be 2D, got {img_gray01.shape}")
    if len(_TTA_FORWARD) != len(_TTA_INVERSE):
        raise RuntimeError("TTA forward/inverse op count mismatch")

    h, w = img_gray01.shape
    acc = np.zeros((h, w), dtype=np.float32)

    for fwd, inv in zip(_TTA_FORWARD, _TTA_INVERSE):
        view = fwd(img_gray01)
        prob = predict_full_mask_tiled(model, view, patch=patch, stride=stride, device=device)
        acc += inv(prob)

    return acc / float(len(_TTA_FORWARD))


def _predict_one(
    model: nn.Module,
    img: np.ndarray,
    patch: int,
    stride: int,
    device: str,
    use_tta: bool,
) -> np.ndarray:
    """Dispatch to TTA or plain tiled inference."""
    if use_tta:
        return predict_full_mask_tta(model, img, patch=patch, stride=stride, device=device)
    return predict_full_mask_tiled(model, img, patch=patch, stride=stride, device=device)


@torch.no_grad()
def evaluate_series_full(
    model: nn.Module,
    frames: List[FrameItem],
    patch: int,
    stride: int,
    device: str,
    thr: float = 0.5,
    compute_loss: bool = False,
    use_tta: bool = False,
    eps: float = 1e-7,
) -> Dict[str, float]:
    """
    Tiled full image evaluation across a list of frames; returns averaged metrics.

    When compute_loss is True, also returns mean per-pixel BCE on probabilities.
    When use_tta is True, runs D4 test-time augmentation (8x slower).
    """
    if not frames:
        out: Dict[str, float] = {"dice": 0.0, "iou": 0.0, "precision": 0.0, "recall": 0.0}
        if compute_loss:
            out["loss"] = 0.0
        return out

    dices, ious, precs, recs, losses = [], [], [], [], []

    for it in frames:
        img = read_image_gray(it.img_path)
        mask = read_mask_01(it.mask_path).astype(np.float32)
        prob = _predict_one(model, img, patch, stride, device, use_tta)

        met = evaluate_full_image_numpy(prob, mask, thr=thr)
        dices.append(met["dice"])
        ious.append(met["iou"])
        precs.append(met["precision"])
        recs.append(met["recall"])

        if compute_loss:
            p = np.clip(prob, eps, 1.0 - eps)
            y = np.clip(mask, 0.0, 1.0)
            losses.append(float(-(y * np.log(p) + (1.0 - y) * np.log(1.0 - p)).mean()))

    out = {
        "dice": float(np.mean(dices)),
        "iou": float(np.mean(ious)),
        "precision": float(np.mean(precs)),
        "recall": float(np.mean(recs)),
    }
    if compute_loss:
        out["loss"] = float(np.mean(losses)) if losses else 0.0
    return out


@torch.no_grad()
def evaluate_with_threshold_sweep(
    model: nn.Module,
    frames: List[FrameItem],
    patch: int,
    stride: int,
    device: str,
    thresholds: Sequence[float],
    use_tta: bool = False,
) -> Dict[float, Dict[str, float]]:
    """
    Run inference once per frame, score under multiple thresholds.

    Returns a mapping threshold -> {dice, iou, precision, recall}.
    Probability maps are computed once and reused across thresholds.
    """
    if not frames:
        return {float(t): {"dice": 0.0, "iou": 0.0, "precision": 0.0, "recall": 0.0}
                for t in thresholds}
    if not thresholds:
        raise ValueError("thresholds must contain at least one value")

    accumulators: Dict[float, Dict[str, List[float]]] = {
        float(t): {"dice": [], "iou": [], "precision": [], "recall": []}
        for t in thresholds
    }

    for it in frames:
        img = read_image_gray(it.img_path)
        mask = read_mask_01(it.mask_path).astype(np.float32)
        prob = _predict_one(model, img, patch, stride, device, use_tta)

        for t in thresholds:
            met = evaluate_full_image_numpy(prob, mask, thr=float(t))
            accumulators[float(t)]["dice"].append(met["dice"])
            accumulators[float(t)]["iou"].append(met["iou"])
            accumulators[float(t)]["precision"].append(met["precision"])
            accumulators[float(t)]["recall"].append(met["recall"])

    return {
        t: {k: float(np.mean(v)) for k, v in metrics.items()}
        for t, metrics in accumulators.items()
    }


def find_best_threshold(
    sweep_result: Dict[float, Dict[str, float]],
    metric: str = "dice",
) -> Tuple[float, Dict[str, float]]:
    """
    Pick the threshold that maximises the chosen metric in a sweep result.

    Returns (best_threshold, metrics_at_best).
    """
    if not sweep_result:
        raise ValueError("sweep_result is empty")
    if metric not in next(iter(sweep_result.values())):
        raise ValueError(
            f"metric '{metric}' not in sweep result keys: "
            f"{list(next(iter(sweep_result.values())).keys())}"
        )

    best_t = max(sweep_result.keys(), key=lambda t: sweep_result[t][metric])
    return float(best_t), dict(sweep_result[best_t])


@torch.no_grad()
def evaluate_with_object_metrics(
    model: nn.Module,
    frames: List[FrameItem],
    patch: int,
    stride: int,
    device: str,
    thr: float,
    use_tta: bool = False,
    iou_threshold: float = 0.5,
) -> Dict[str, float]:
    """
    Full evaluation including pixel and object-level metrics.

    Returns the standard pixel metrics (dice, iou, precision, recall)
    plus object-level metrics (obj_precision, obj_recall, obj_f1,
    obj_splits, obj_merges, obj_tp, obj_fp, obj_fn).
    """
    if not frames:
        out = {"dice": 0.0, "iou": 0.0, "precision": 0.0, "recall": 0.0}
        out.update(aggregate_object_metrics([]))
        return out

    dices, ious, precs, recs = [], [], [], []
    object_metrics_per_image = []

    for it in frames:
        img = read_image_gray(it.img_path)
        mask = read_mask_01(it.mask_path).astype(np.float32)
        prob = _predict_one(model, img, patch, stride, device, use_tta)

        met = evaluate_full_image_numpy(prob, mask, thr=thr)
        dices.append(met["dice"])
        ious.append(met["iou"])
        precs.append(met["precision"])
        recs.append(met["recall"])

        pred_bin = (prob >= thr).astype(np.uint8)
        gt_bin = (mask >= (_MASK_BIN_THRESHOLD / 255.0)).astype(np.uint8)
        # mask values from read_mask_01 are already in {0, 1}; the >=
        # comparison above is a no-op safeguard for any caller passing
        # a uint8 mask through.
        object_metrics_per_image.append(
            compute_object_metrics(pred_bin, gt_bin, iou_threshold=iou_threshold)
        )

    out = {
        "dice": float(np.mean(dices)),
        "iou": float(np.mean(ious)),
        "precision": float(np.mean(precs)),
        "recall": float(np.mean(recs)),
    }
    out.update(aggregate_object_metrics(object_metrics_per_image))
    return out


_FOLD_RE = re.compile(r"^fold_\d+_test_(.+)$")


def _extract_test_series(fold_name: str, available: List[str]) -> Optional[str]:
    """
    Extract the test series name from a fold_name string and verify it
    exists in the dataset index. Returns None on failure.
    """
    m = _FOLD_RE.match(fold_name)
    if m and m.group(1) in available:
        return m.group(1)
    for s in available:
        if fold_name.endswith(s):
            return s
    return None


def evaluate_checkpoint(
    ckpt_path: str,
    *,
    dataset_root: Optional[str] = None,
    patch: Optional[int] = None,
    stride: Optional[int] = None,
    thr: float = 0.5,
    device: Optional[str] = None,
    use_tta: bool = False,
    compute_object_metrics_flag: bool = False,
) -> Dict[str, float]:
    """
    Evaluate a single CiproSeg checkpoint on its LOSO held-out test series.

    Everything needed (model architecture, weights, dataset root, fold
    identity) is read from the checkpoint. Optional keyword arguments
    let the caller override dataset_root / patch / stride / device, and
    enable TTA or object-level metrics for the final evaluation.

    Returns a dict with dice, iou, precision, recall (and loss).
    Object metrics are also returned when compute_object_metrics_flag=True.
    """
    logger = get_logger()

    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    state = torch.load(ckpt_path, map_location="cpu")
    for required_key in ("model", "cfg", "fold_name"):
        if required_key not in state:
            raise KeyError(
                f"Checkpoint {ckpt_path!r} is missing required key '{required_key}'. "
                "Expected a CiproSeg checkpoint saved by Checkpointer."
            )

    ckpt_cfg = state["cfg"]
    fold_name = state["fold_name"]

    dcfg = ckpt_cfg.get("dataset") or {}
    _dataset_root = dataset_root or dcfg.get("root")
    if _dataset_root is None:
        raise ValueError(
            "Cannot determine dataset root: not in checkpoint cfg and "
            "no --dataset-root override provided."
        )
    _patch = patch if patch is not None else int(dcfg.get("patch", 256))
    _stride = stride if stride is not None else int(dcfg.get("stride", 128))

    if device is not None:
        _device = device
    else:
        tcfg = ckpt_cfg.get("trainer") or {}
        force_cpu = bool(tcfg.get("cpu", False))
        _device = (
            "cuda"
            if (not force_cpu) and torch.cuda.is_available()
            else "cpu"
        )

    model = build_model_from_cfg(ckpt_cfg)
    model.load_state_dict(state["model"])
    model.to(_device).eval()

    logger.info(
        "[Evaluator] Loaded %s  (fold=%s, best_val=%.4f, device=%s, tta=%s, obj=%s)",
        ckpt_path,
        fold_name,
        float(state.get("best_val_metric", float("nan"))),
        _device,
        use_tta,
        compute_object_metrics_flag,
    )

    index = build_index(_dataset_root)
    test_series = _extract_test_series(fold_name, list(index.keys()))
    if test_series is None:
        raise ValueError(
            f"Could not determine the test series from fold_name={fold_name!r}. "
            f"Available series in dataset: {sorted(index.keys())}"
        )

    test_frames = index[test_series]
    logger.info(
        "[Evaluator] Test series: %s  (%d frames)  patch=%d  stride=%d  thr=%.2f",
        test_series,
        len(test_frames),
        _patch,
        _stride,
        thr,
    )

    with torch.inference_mode():
        if compute_object_metrics_flag:
            metrics = evaluate_with_object_metrics(
                model,
                test_frames,
                patch=_patch,
                stride=_stride,
                device=_device,
                thr=thr,
                use_tta=use_tta,
            )
        else:
            metrics = evaluate_series_full(
                model,
                test_frames,
                patch=_patch,
                stride=_stride,
                device=_device,
                thr=thr,
                compute_loss=True,
                use_tta=use_tta,
            )

    metrics["fold_name"] = fold_name
    metrics["test_series"] = test_series
    metrics["ckpt_path"] = ckpt_path
    metrics["thr"] = float(thr)

    logger.info(
        "[Evaluator] %s  dice=%.4f  iou=%.4f  prec=%.4f  rec=%.4f",
        fold_name,
        metrics["dice"],
        metrics["iou"],
        metrics["precision"],
        metrics["recall"],
    )
    return metrics


class Evaluator:
    """
    Thin Engine-facing wrapper. Engine.evaluate() and Engine.test() call
    through here; the real work is done by evaluate_checkpoint().
    """

    def __init__(
        self,
        cfg: Dict[str, Any],
        datamodule: Any = None,
        hooks: Optional[HookList] = None,
    ) -> None:
        self.cfg = cfg
        self.datamodule = datamodule
        self.hooks = hooks or HookList([])
        self.logger = get_logger()

    def evaluate(self, ckpt_path: Optional[str] = None) -> Dict[str, float]:
        if ckpt_path is None:
            raise ValueError(
                "No checkpoint path provided. Use: ciproseg-eval --ckpt <path/to/best.pt>"
            )
        ecfg = self.cfg.get("eval") or {}
        return evaluate_checkpoint(
            ckpt_path,
            thr=float(ecfg.get("thr", 0.5)),
            use_tta=bool(ecfg.get("tta", False)),
            compute_object_metrics_flag=bool(ecfg.get("object_metrics", False)),
        )

    def test(self, ckpt_path: Optional[str] = None) -> Dict[str, float]:
        return self.evaluate(ckpt_path=ckpt_path)
    