#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SPDX-License-Identifier: Apache-2.0
# =======================================================================
#  Project       : CiproSeg - LOSO U-Net Segmentation Framework
#  File          : src/ciproseg/core/callbacks/progress_bar.py
#  Purpose       : Progress bar for printing
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

from datetime import datetime

from ..hooks import Hook
from ...utils.dist import is_global_zero


def _fmt(x, nd: int = 4) -> str:
    try:
        return f"{float(x):.{nd}f}"
    except Exception:
        return "nan"


def _ts() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


class ProgressBar(Hook):
    """Prints training progress to stdout. Reads state populated by Trainer/Evaluator."""

    def on_train_start(self, state) -> None:
        if is_global_zero():
            print("[PB] Training started")

    def on_fold_start(self, state) -> None:
        if not is_global_zero():
            return
        fold_id = state.get("fold_id")
        fold_name = state.get("fold_name", "")
        test_series = state.get("test_series", "")
        n_train, n_val, n_test = state.get("n_train"), state.get("n_val"), state.get("n_test")
        patch = state.get("patch")

        print("\n====================")
        fold_label = f"{fold_id:02d} | " if fold_id is not None else ""
        print(f"[PB] Fold {fold_label}{fold_name} | test_series = {test_series}")
        if any(v is not None for v in (n_train, n_val, n_test)):
            print(f"[PB] frames: train = {n_train} | val = {n_val} | test = {n_test}")
        if patch is not None:
            print(f"[PB] patch={patch} stride={state.get('stride')} epochs={state.get('epochs')} "
                  f"batch={state.get('batch_size')} lr={state.get('lr')}")
        print("====================")

    def on_epoch_end(self, state) -> None:
        if not is_global_zero():
            return
        val_metrics = state.get("val_metrics") or {}
        best_val = state.get("best_val_metric")
        msg = (
            f"[{_ts()}] [PB] {state.get('fold_name', '')} | ep {int(state.get('epoch', 0)):03d} | "
            f"train_loss = {_fmt(state.get('train_loss'), nd=5)} | "
            f"val_loss = {_fmt(val_metrics.get('loss'), nd=5)} | "
            f"val_dice = {_fmt(val_metrics.get('dice'))} | "
            f"val_iou = {_fmt(val_metrics.get('iou'))}"
        )
        if best_val is not None:
            msg += f" | best_val_dice = {_fmt(best_val)}"
        print(msg, flush=True)

    def on_fold_end(self, state) -> None:
        if not is_global_zero():
            return
        m = state.get("test_metrics") or {}
        print(
            f"[{_ts()}] {state.get('fold_name', '')} |< TEST >| "
            f"dice = {_fmt(m.get('dice'))} | iou = {_fmt(m.get('iou'))} | "
            f"prec = {_fmt(m.get('precision'))} | rec = {_fmt(m.get('recall'))}"
        )

    def on_train_end(self, state) -> None:
        if is_global_zero():
            print("[PB] Training finished")
            