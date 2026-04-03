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
#  Last modified : 2026-04-03, MK
#  Version       : 0.1.0
#  License       : Apache-2.0
#  Repository    : https://github.com/manazucd/CiproSeg
# =======================================================================
from __future__ import annotations

import os
import re
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from ..data.datasets.ciproseg_ecoli import (
    FrameItem,
    build_index,
    read_image_gray,
    read_mask_01,
    pad_to_patch,
)
from ..metrics.segmentation import evaluate_full_image_numpy
from ..models import build_model_from_cfg
from ..utils.logger import get_logger
from .hooks import HookList

def tile_positions(h: int, w: int, patch: int, stride: int) -> List[Tuple[int, int]]:
    """Deterministic (x, y) top-left tile positions that fully cover the image."""
    ys = list(range(0, max(1, h - patch + 1), stride))
    xs = list(range(0, max(1, w - patch + 1), stride))
    if ys[-1] != max(0, h - patch):
        ys.append(max(0, h - patch))
    if xs[-1] != max(0, w - patch):
        xs.append(max(0, w - patch))
    return [(x, y) for y in ys for x in xs]


@torch.no_grad()
def predict_full_mask_tiled(
    model: nn.Module,
    img_gray01: np.ndarray,
    patch: int = 256,
    stride: int = 128,
    device: str = "cuda",
) -> np.ndarray:
    """
    Full image tiled inference with overlap averaging.
    Expects a 2D float array in [0, 1]; returns a probability map of the same shape.
    """
    if img_gray01.ndim != 2:
        raise ValueError(f"img_gray01 must be 2D (H,W), got {img_gray01.shape}")

    model.eval()
    h, w = img_gray01.shape
    acc = np.zeros((h, w), dtype=np.float32)
    wgt = np.zeros((h, w), dtype=np.float32)

    for x, y in tile_positions(h, w, patch, stride):
        crop = img_gray01[y:y + patch, x:x + patch]

        if crop.shape != (patch, patch):
            crop, _ = pad_to_patch(crop, np.zeros_like(crop, dtype=np.uint8), patch)

        inp = torch.from_numpy(crop.astype(np.float32)).unsqueeze(0).unsqueeze(0).to(device)
        prob = torch.sigmoid(model(inp)).squeeze().detach().cpu().numpy().astype(np.float32)

        ph, pw = min(patch, h - y), min(patch, w - x)
        acc[y:y + ph, x:x + pw] += prob[:ph, :pw]
        wgt[y:y + ph, x:x + pw] += 1.0

    return acc / np.maximum(wgt, 1e-8)


@torch.no_grad()
def evaluate_series_full(
    model: nn.Module,
    frames: List[FrameItem],
    patch: int,
    stride: int,
    device: str,
    thr: float = 0.5,
    compute_loss: bool = False,
    eps: float = 1e-7,
) -> Dict[str, float]:
    """
    Tiled full image evaluation across a list of frames; returns averaged metrics.
    When compute_loss=True, also returns mean per-pixel BCE on probabilities.
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
        prob = predict_full_mask_tiled(model, img, patch=patch, stride=stride, device=device)

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


_FOLD_RE = re.compile(r"^fold_\d+_test_(.+)$")

def _extract_test_series(fold_name: str, available: List[str]) -> Optional[str]:
    """
    Extract the test series name from a fold_name string and verify it
    exists in the dataset index.  Returns None on failure.
    """
    m = _FOLD_RE.match(fold_name)
    if m and m.group(1) in available:
        return m.group(1)
    # Fallback: check if any known series name is a suffix of fold_name.
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
) -> Dict[str, float]:
    """
    Evaluate a single CiproSeg checkpoint on its LOSO held-out test series.

    Everything needed (model architecture, weights, dataset root, fold
    identity) is read from the checkpoint.  Optional keyword arguments
    let the caller override dataset_root / patch / stride / device if the
    data has moved or different tiling is desired.

    Returns a dict with dice, iou, precision, recall (and loss).
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
        "[Evaluator] Loaded %s  (fold=%s, best_val=%.4f, device=%s)",
        ckpt_path,
        fold_name,
        float(state.get("best_val_metric", float("nan"))),
        _device,
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
        metrics = evaluate_series_full(
            model,
            test_frames,
            patch=_patch,
            stride=_stride,
            device=_device,
            thr=thr,
            compute_loss=True,
        )

    metrics["fold_name"] = fold_name
    metrics["test_series"] = test_series
    metrics["ckpt_path"] = ckpt_path

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
        return evaluate_checkpoint(ckpt_path)

    def test(self, ckpt_path: Optional[str] = None) -> Dict[str, float]:
        return self.evaluate(ckpt_path=ckpt_path)
    