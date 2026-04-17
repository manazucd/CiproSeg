#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SPDX-License-Identifier: Apache-2.0
# =======================================================================
#  Project       : CiproSeg - LOSO U-Net Segmentation Framework
#  File          : src/ciproseg/visualization/plot_curves.py
#  Purpose       : Plot
#
#  Copyright (c) 2026 Manaz Kaleel
#  Authors       : Manaz Kaleel <manaz.ucd@gmail.com>
#
#  Created on    : 2025-11-01
#  Last modified : 2026-02-25, MK
#  Version       : 0.1.0
#  License       : Apache-2.0
#  Repository    : https://github.com/manazucd/CiproSeg
# =======================================================================
from __future__ import annotations

import os
import re
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt

@dataclass(frozen=True)
class Curve:
    name: str
    x: List[float]
    y: List[float]


@dataclass(frozen=True)
class FoldCurves:
    fold_name: str
    curves: Dict[str, Curve]


_METRIC_PATTERN = re.compile(r"(?P<key>[a-zA-Z_][a-zA-Z0-9_]*)\s*=\s*(?P<val>[-+0-9.eE]+)")

# Support both "Epoch 12" (old) and "ep 012" (current ProgressBar format).
_EPOCH_PATTERNS = (
    re.compile(r"\bEpoch\s+(?P<epoch>\d+)\b", re.IGNORECASE),
    re.compile(r"\bep\s+(?P<epoch>\d+)\b", re.IGNORECASE),
)


def _safe_float(s: str) -> Optional[float]:
    try:
        return float(s)
    except Exception:
        return None


def parse_metrics_from_line(line: str) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for m in _METRIC_PATTERN.finditer(line):
        v = _safe_float(m.group("val"))
        if v is not None:
            out[m.group("key")] = v
    return out


def parse_epoch_from_line(line: str) -> Optional[int]:
    for pat in _EPOCH_PATTERNS:
        m = pat.search(line)
        if m:
            try:
                return int(m.group("epoch"))
            except Exception:
                return None
    return None


def parse_fold_from_line(line: str) -> Optional[str]:
    """Extracts fold name from common log line patterns (bracket prefix, 'Fold:' label, PB pipe format, inline token)."""
    for pat in (
        r"\[(fold\d+_test_[^\]]+)\]",                               # [fold01_test_X]
        r"\bFold:\s*(fold\d+_test_[A-Za-z0-9_-]+)",                 # Fold: fold01_test_X
        r"\bFold\s+\d+\s*\|\s*(fold\d+_test_[A-Za-z0-9_-]+)\b",     # Fold 01 | fold01_test_X | ...
        r"\b(fold\d+_test_[A-Za-z0-9_-]+)\b",                       # inline token
    ):
        m = re.search(pat, line)
        if m:
            return m.group(1)
    return None

def read_training_log(
    path: str,
    keys: Sequence[str] = ("train_loss", "val_dice", "val_iou", "lr"),
    *,
    strict_epoch: bool = False,
) -> List[FoldCurves]:
    """Parses a text log file and returns per fold metric curves. Works with both 'Epoch N' and 'ep N' formats."""
    if not os.path.exists(path):
        raise FileNotFoundError(path)

    store: Dict[str, Dict[str, Tuple[List[float], List[float]]]] = {}
    last_fold: Optional[str] = None

    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            fold = parse_fold_from_line(line) or last_fold
            if fold is not None:
                last_fold = fold

            epoch = parse_epoch_from_line(line)
            metrics = parse_metrics_from_line(line)

            if strict_epoch and epoch is None:
                continue

            for k in keys:
                if k not in metrics:
                    continue
                fold_key = fold or "fold_unknown"
                store.setdefault(fold_key, {}).setdefault(k, ([], []))
                xs, ys = store[fold_key][k]
                xs.append(float(epoch) if epoch is not None else float(len(xs) + 1))
                ys.append(float(metrics[k]))

    return [
        FoldCurves(fold_name=fn, curves={k: Curve(name=k, x=xs, y=ys) for k, (xs, ys) in mm.items()})
        for fn, mm in sorted(store.items())
    ]


def _ensure_dir(path: str) -> None:
    d = os.path.dirname(os.path.abspath(path))
    if d:
        os.makedirs(d, exist_ok=True)


def _x_label(xs: List[float]) -> str:
    if not xs:
        return "step"
    return "epoch" if all(abs(x - round(x)) < 1e-9 for x in xs) else "step"


def _first_xs_for_metric(folds: Sequence[FoldCurves], metric: str) -> List[float]:
    for fc in folds:
        c = fc.curves.get(metric)
        if c and c.x:
            return c.x
    return [1.0]

def _annotate_best_point(
    ax,
    curve: Curve,
    *,
    mode: str,
    label: str,
    decimals: int = 5,
) -> None:
    """
    Annotate the best point on a curve with a dot and a text label.

    mode:
      - "min" for losses
      - "max" for scores (dice/iou)
    """
    if not curve.x or not curve.y:
        return

    if len(curve.x) != len(curve.y):
        # Defensive: avoid confusing annotations if data malformed
        return

    best_idx = min(range(len(curve.y)), key=lambda i: curve.y[i]) if mode == "min" \
        else max(range(len(curve.y)), key=lambda i: curve.y[i])

    bx = float(curve.x[best_idx])
    by = float(curve.y[best_idx])

    ax.scatter([bx], [by], marker="o", s=40, zorder=5)

    txt = f"{by:.{decimals}f}, {int(round(bx)) if abs(bx-round(bx))<1e-9 else bx}"
    ax.annotate(
        txt,
        xy=(bx, by),
        xytext=(8, 8),
        textcoords="offset points",
        ha="left",
        va="bottom",
        fontsize=8,
    )


def plot_fold_curves(
    fold_curves: FoldCurves,
    metrics: Sequence[str] = ("train_loss", "val_dice", "val_iou", "lr"),
    *,
    title: Optional[str] = None,
    show: bool = True,
    savepath: Optional[str] = None,
    annotate_best: bool = True,
) -> None:
    """Plots selected metrics for a single fold."""
    present = [m for m in metrics if m in fold_curves.curves]
    if not present:
        raise ValueError(
            f"No requested metrics found for {fold_curves.fold_name}. "
            f"Available={list(fold_curves.curves)}"
        )
    fig, ax = plt.subplots()

    for m in present:
        c = fold_curves.curves[m]
        ax.plot(c.x, c.y, label=m)

    if annotate_best:
        for m in ("train_loss", "val_loss"):
            if m in fold_curves.curves and m in present:
                _annotate_best_point(ax, fold_curves.curves[m], mode="min", label=m, decimals=5)

        for m in ("val_dice", "val_iou"):
            if m in fold_curves.curves and m in present:
                _annotate_best_point(ax, fold_curves.curves[m], mode="max", label=m, decimals=4)

    ax.set_xlabel(_x_label(fold_curves.curves[present[0]].x))
    ax.set_ylabel("value")
    ax.set_title(title or fold_curves.fold_name)
    ax.legend()
    fig.tight_layout()

    if savepath:
        Path(savepath).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(savepath, dpi=150)

    if show:
        plt.show()
    plt.close(fig)


def plot_metric_across_folds(
    folds: Sequence[FoldCurves],
    metric: str = "val_dice",
    *,
    title: Optional[str] = None,
    show: bool = True,
    savepath: Optional[str] = None,
) -> None:
    """Plots one metric across multiple folds on a single figure."""
    plt.figure()
    plotted = 0

    for fc in folds:
        c = fc.curves.get(metric)
        if not c:
            continue
        plt.plot(c.x, c.y, label=fc.fold_name)
        plotted += 1

    if plotted == 0:
        raise ValueError(f"Metric '{metric}' not found in any fold.")

    plt.xlabel(_x_label(_first_xs_for_metric(folds, metric)))
    plt.ylabel(metric)
    plt.title(title or f"{metric} across folds")
    plt.legend(fontsize=8, ncol=1)
    plt.tight_layout()

    if savepath:
        _ensure_dir(savepath)
        plt.savefig(savepath, dpi=150)
    plt.show() if show else plt.close()


def summarize_best_per_fold(
    folds: Sequence[FoldCurves],
    metric: str = "val_dice",
    mode: str = "max",
) -> Dict[str, float]:
    """Returns the best metric value per fold. mode='max' for dice/iou, 'min' for loss."""
    fn = min if mode == "min" else max
    return {
        fc.fold_name: float(fn(fc.curves[metric].y))
        for fc in folds
        if metric in fc.curves and fc.curves[metric].y
    }


def plot_from_logfile(
    logfile: str,
    *,
    keys: Sequence[str] = ("train_loss", "val_dice", "val_iou", "lr"),
    fold: Optional[str] = None,
    metric_across_folds: Optional[str] = None,
    show: bool = True,
    outdir: Optional[str] = None,
) -> List[FoldCurves]:
    """One call helper for notebooks and scripts, parses a log file and plots as requested."""
    folds = read_training_log(logfile, keys=keys, strict_epoch=False)

    if outdir:
        os.makedirs(outdir, exist_ok=True)

    if fold is not None:
        fc = next((x for x in folds if x.fold_name == fold), None)
        if fc is None:
            raise KeyError(f"Fold '{fold}' not found. Available: {[x.fold_name for x in folds]}")
        plot_fold_curves(fc, metrics=keys, title=fold, show=show,
                         savepath=os.path.join(outdir, f"{fold}_curves.png") if outdir else None)

    if metric_across_folds is not None:
        plot_metric_across_folds(folds, metric=metric_across_folds, show=show,
                                 savepath=os.path.join(outdir, f"{metric_across_folds}_across_folds.png") if outdir else None)

    return folds
