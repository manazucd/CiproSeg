#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SPDX-License-Identifier: Apache-2.0
# =======================================================================
#  Project       : CiproSeg - LOSO U-Net Segmentation Framework
#  File          : src/ciproseg/metrics/segmentation.py
#  Purpose       : Segmentation metrics
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

from typing import Dict, Tuple

import numpy as np
import torch


def _confusion_from_probs(
    probs: torch.Tensor, target: torch.Tensor, thr: float = 0.5
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    pred = (probs >= thr).float()
    gt = (target >= 0.5).float()
    tp = (pred * gt).sum()
    fp = (pred * (1 - gt)).sum()
    fn = ((1 - pred) * gt).sum()
    return tp, fp, fn


def dice_from_probs(probs: torch.Tensor, target: torch.Tensor, thr: float = 0.5, eps: float = 1e-8) -> float:
    tp, fp, fn = _confusion_from_probs(probs, target, thr)
    return float(((2 * tp + eps) / (2 * tp + fp + fn + eps)).detach().cpu())


def iou_from_probs(probs: torch.Tensor, target: torch.Tensor, thr: float = 0.5, eps: float = 1e-8) -> float:
    tp, fp, fn = _confusion_from_probs(probs, target, thr)
    return float(((tp + eps) / (tp + fp + fn + eps)).detach().cpu())


def precision_from_probs(probs: torch.Tensor, target: torch.Tensor, thr: float = 0.5, eps: float = 1e-8) -> float:
    tp, fp, _ = _confusion_from_probs(probs, target, thr)
    return float(((tp + eps) / (tp + fp + eps)).detach().cpu())


def recall_from_probs(probs: torch.Tensor, target: torch.Tensor, thr: float = 0.5, eps: float = 1e-8) -> float:
    tp, _, fn = _confusion_from_probs(probs, target, thr)
    return float(((tp + eps) / (tp + fn + eps)).detach().cpu())


def evaluate_full_image_numpy(
    prob_map: np.ndarray, mask01: np.ndarray, thr: float = 0.5, eps: float = 1e-8
) -> Dict[str, float]:
    """Full image metric evaluation for tiled inference outputs. Returns dice, iou, precision, recall."""
    if prob_map.ndim != 2 or mask01.ndim != 2:
        raise ValueError(f"Expected 2D arrays, got prob_map {prob_map.shape}, mask {mask01.shape}")

    pred = (prob_map >= thr).astype(np.uint8)
    gt = (mask01 > 0).astype(np.uint8)

    tp = float((pred & gt).sum())
    fp = float((pred & (1 - gt)).sum())
    fn = float(((1 - pred) & gt).sum())

    return {
        "dice":      (2 * tp + eps) / (2 * tp + fp + fn + eps),
        "iou":       (tp + eps) / (tp + fp + fn + eps),
        "precision": (tp + eps) / (tp + fp + eps),
        "recall":    (tp + eps) / (tp + fn + eps),
    }
