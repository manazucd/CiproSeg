#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SPDX-License-Identifier: Apache-2.0
# =======================================================================
#  Project       : CiproSeg - LOSO U-Net Segmentation Framework
#  File          : src/ciproseg/metrics/objects.py
#  Purpose       : Object-level (cell-level) segmentation metrics
#
#  Copyright (c) 2026 Manaz Kaleel
#  Authors       : Manaz Kaleel <manaz.ucd@gmail.com>
#
#  Created on    : 2026-04-29
#  Last modified : 2026-04-29, MK
#  Version       : 0.1.0
#  License       : Apache-2.0
#  Repository    : https://github.com/manazucd/CiproSeg
# =======================================================================
"""
Cell-level metrics for binary segmentation.

Pixel Dice rewards covering the right area but is blind to splits
(one ground-truth cell predicted as several disconnected components)
and merges (several touching cells predicted as one component).
This module provides the complementary view at object granularity.

A predicted component is a true positive if it overlaps a unique
ground-truth component with IoU at least the given threshold (0.5
by default, the standard PASCAL VOC convention). Otherwise it is a
false positive. Unmatched ground-truth components are false negatives.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import cv2
import numpy as np

_CC_CONNECTIVITY = 8
_DEFAULT_MIN_AREA = 16


@dataclass(frozen=True)
class ObjectMetrics:
    """Object-level metrics for one image."""
    n_gt: int
    n_pred: int
    tp: int
    fp: int
    fn: int
    splits: int
    merges: int
    precision: float
    recall: float
    f1: float

    def to_dict(self) -> Dict[str, float]:
        return {
            "obj_n_gt": float(self.n_gt),
            "obj_n_pred": float(self.n_pred),
            "obj_tp": float(self.tp),
            "obj_fp": float(self.fp),
            "obj_fn": float(self.fn),
            "obj_splits": float(self.splits),
            "obj_merges": float(self.merges),
            "obj_precision": float(self.precision),
            "obj_recall": float(self.recall),
            "obj_f1": float(self.f1),
        }


def label_components(mask01: np.ndarray, min_area: int = _DEFAULT_MIN_AREA) -> Tuple[np.ndarray, int]:
    """
    Label 8-connected components in a binary mask.

    Components smaller than min_area pixels are dropped (set to 0).
    Returns (labels, n_components) where labels has 0 for background and
    1..n_components for foreground components, and n_components is the
    count after filtering.
    """
    if mask01.ndim != 2:
        raise ValueError(f"mask01 must be 2D, got shape {mask01.shape}")
    if min_area < 0:
        raise ValueError(f"min_area must be >= 0, got {min_area}")

    bin_u8 = (mask01 > 0).astype(np.uint8)
    n_total, labels, stats, _ = cv2.connectedComponentsWithStats(
        bin_u8, connectivity=_CC_CONNECTIVITY
    )

    keep_mapping = np.zeros(n_total, dtype=np.int32)
    next_label = 1
    for cid in range(1, n_total):
        if stats[cid, cv2.CC_STAT_AREA] >= min_area:
            keep_mapping[cid] = next_label
            next_label += 1

    if next_label == 1:
        return np.zeros_like(labels, dtype=np.int32), 0

    relabelled = keep_mapping[labels]
    return relabelled.astype(np.int32), int(next_label - 1)


def _build_overlap_matrix(
    gt_labels: np.ndarray, pred_labels: np.ndarray, n_gt: int, n_pred: int
) -> np.ndarray:
    """
    Build an (n_gt + 1) x (n_pred + 1) overlap matrix in pixels.
    Row 0 and column 0 are background; valid components are 1..n_gt and
    1..n_pred respectively.
    """
    if n_gt == 0 or n_pred == 0:
        return np.zeros((n_gt + 1, n_pred + 1), dtype=np.int64)

    flat_idx = (
        gt_labels.astype(np.int64) * (n_pred + 1)
        + pred_labels.astype(np.int64)
    ).ravel()
    counts = np.bincount(flat_idx, minlength=(n_gt + 1) * (n_pred + 1))
    return counts.reshape(n_gt + 1, n_pred + 1)


def compute_object_metrics(
    pred_mask01: np.ndarray,
    gt_mask01: np.ndarray,
    iou_threshold: float = 0.5,
    min_area: int = _DEFAULT_MIN_AREA,
) -> ObjectMetrics:
    """
    Object-level precision, recall, F1, and split/merge counts for one image.

    Steps:
        1. Label connected components in prediction and ground truth
        2. Build the pairwise pixel-overlap matrix
        3. Match pred components to gt components by IoU >= threshold
        4. Each gt with >= 2 matches contributes to splits
        5. Each pred with >= 2 matches contributes to merges
        6. TP = matched pairs, FP = unmatched preds, FN = unmatched gts

    Parameters:
        pred_mask01: Predicted binary mask, 2D array
        gt_mask01:   Ground-truth binary mask, 2D array
        iou_threshold: Minimum IoU for a pred-gt pair to be a match
        min_area:    Components smaller than this are discarded as noise

    Returns:
        ObjectMetrics dataclass with all counts and rates.
    """
    if pred_mask01.shape != gt_mask01.shape:
        raise ValueError(
            f"shape mismatch: pred {pred_mask01.shape} vs gt {gt_mask01.shape}"
        )
    if not (0.0 < iou_threshold <= 1.0):
        raise ValueError(f"iou_threshold must be in (0, 1], got {iou_threshold}")

    gt_labels, n_gt = label_components(gt_mask01, min_area=min_area)
    pred_labels, n_pred = label_components(pred_mask01, min_area=min_area)

    if n_gt == 0 and n_pred == 0:
        return ObjectMetrics(0, 0, 0, 0, 0, 0, 0, 1.0, 1.0, 1.0)
    if n_gt == 0:
        return ObjectMetrics(0, n_pred, 0, n_pred, 0, 0, 0, 0.0, 0.0, 0.0)
    if n_pred == 0:
        return ObjectMetrics(n_gt, 0, 0, 0, n_gt, 0, 0, 0.0, 0.0, 0.0)

    overlap = _build_overlap_matrix(gt_labels, pred_labels, n_gt, n_pred)
    overlap_inner = overlap[1:, 1:]  

    gt_areas = overlap[1:, :].sum(axis=1)
    pred_areas = overlap[:, 1:].sum(axis=0)

    union = (
        gt_areas[:, None] + pred_areas[None, :] - overlap_inner
    ).astype(np.float64)
    iou = np.where(union > 0, overlap_inner.astype(np.float64) / union, 0.0)

    matched = iou >= iou_threshold
    matches_per_gt = matched.sum(axis=1)
    matches_per_pred = matched.sum(axis=0)

    tp = int((matches_per_gt > 0).sum())
    fn = int((matches_per_gt == 0).sum())
    fp = int((matches_per_pred == 0).sum())

    splits = int(((matches_per_gt - 1).clip(min=0)).sum())
    merges = int(((matches_per_pred - 1).clip(min=0)).sum())

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (
        2.0 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    return ObjectMetrics(
        n_gt=int(n_gt),
        n_pred=int(n_pred),
        tp=tp,
        fp=fp,
        fn=fn,
        splits=splits,
        merges=merges,
        precision=float(precision),
        recall=float(recall),
        f1=float(f1),
    )


def aggregate_object_metrics(per_image: list) -> Dict[str, float]:
    """
    Aggregate per-image ObjectMetrics into a single mean-style summary.

    TP, FP, FN, splits, merges are summed across images (cell counts
    are additive). Precision, recall, F1 are recomputed from the totals
    (micro average), which is the standard convention.
    """
    if not per_image:
        return {
            "obj_tp": 0.0, "obj_fp": 0.0, "obj_fn": 0.0,
            "obj_splits": 0.0, "obj_merges": 0.0,
            "obj_precision": 0.0, "obj_recall": 0.0, "obj_f1": 0.0,
            "obj_n_gt": 0.0, "obj_n_pred": 0.0,
        }

    tp = sum(m.tp for m in per_image)
    fp = sum(m.fp for m in per_image)
    fn = sum(m.fn for m in per_image)
    splits = sum(m.splits for m in per_image)
    merges = sum(m.merges for m in per_image)
    n_gt_total = sum(m.n_gt for m in per_image)
    n_pred_total = sum(m.n_pred for m in per_image)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (
        2.0 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    return {
        "obj_tp": float(tp),
        "obj_fp": float(fp),
        "obj_fn": float(fn),
        "obj_splits": float(splits),
        "obj_merges": float(merges),
        "obj_precision": float(precision),
        "obj_recall": float(recall),
        "obj_f1": float(f1),
        "obj_n_gt": float(n_gt_total),
        "obj_n_pred": float(n_pred_total),
    }
