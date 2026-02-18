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
#  Last modified : 2026-02-18, MK
#  Version       : 0.1.0
#  License       : Apache-2.0
#  Repository    : https://github.com/manazucd/CiproSeg
# =======================================================================
from __future__ import annotations

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

        # Trim contribution to the valid (non padded) region only.
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


class Evaluator:
    """
    Lightweight Engine facing evaluator. Primary evaluation runs inside Trainer
    as part of LOSO; this class exposes dataset level stats so Engine.evaluate()
    and Engine.test() remain functional throughout development.
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
        self.dataset_root = cfg["dataset"]["root"]

    def evaluate(self) -> Dict[str, float]:
        index = build_index(self.dataset_root)
        return {
            "num_series": float(len(index)),
            "num_frames": float(sum(len(v) for v in index.values())),
        }

    def test(self) -> Dict[str, float]:
        return self.evaluate()
    