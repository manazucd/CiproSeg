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
#  Last modified : 2026-04-03, MK
#  Version       : 0.1.0
#  License       : Apache-2.0
#  Repository    : https://github.com/manazucd/CiproSeg
# =======================================================================
from __future__ import annotations

from datetime import datetime

from ..hooks import Hook
from ...utils.dist import is_global_zero
from ...utils.logger import get_logger


def _fmt(x, nd: int = 4) -> str:
    try:
        return f"{float(x):.{nd}f}"
    except Exception:
        return "nan"


def _ts() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


class ProgressBar(Hook):
    """Prints training progress via the ciproseg logger (console + train.log)."""

    def on_train_start(self, state) -> None:
        if not is_global_zero():
            return
        logger = get_logger()
        cfg = state.get("cfg")
        logger.info("[PB] Training started")
        if cfg is not None:
            model = (cfg.get("model") or {}).get("name", "?")
            droot = (cfg.get("dataset") or {}).get("root", "?")
            patch = (cfg.get("dataset") or {}).get("patch", "?")
            stride = (cfg.get("dataset") or {}).get("stride", "?")
            epochs = (cfg.get("trainer") or {}).get("epochs", "?")
            bs = (cfg.get("trainer") or {}).get("batch_size", "?")
            logger.info(
                "[PB] config: model=%s root=%s patch=%s stride=%s epochs=%s batch=%s",
                model, droot, patch, stride, epochs, bs,
            )

    def on_fold_start(self, state) -> None:
        if not is_global_zero():
            return
        logger = get_logger()
        fold_id = state.get("fold_id")
        fold_name = state.get("fold_name", "")
        test_series = state.get("test_series", "")
        n_train, n_val, n_test = state.get("n_train"), state.get("n_val"), state.get("n_test")
        patch = state.get("patch")

        fold_label = f"{fold_id:02d} | " if fold_id is not None else ""
        lines = [f"[PB] Fold {fold_label}{fold_name} | test_series = {test_series}"]
        if any(v is not None for v in (n_train, n_val, n_test)):
            lines.append(f"[PB] frames: train = {n_train} | val = {n_val} | test = {n_test}")
        if patch is not None:
            lines.append(
                f"[PB] patch={patch} stride={state.get('stride')} epochs={state.get('epochs')} "
                f"batch={state.get('batch_size')} lr={state.get('lr')}"
            )

        width = max(len(line) for line in lines)
        border = "+" + "-" * (width + 2) + "+"
        logger.info(border)
        for line in lines:
            logger.info("| %s |", f"{line:<{width}}")
        logger.info(border)

    def on_epoch_end(self, state) -> None:
        if not is_global_zero():
            return
        logger = get_logger()

        ep = int(state.get("epoch", 0))
        fold_name = state.get("fold_name", "")
        train_loss = _fmt(state.get("train_loss"), nd=5)
        is_eval = state.get("is_eval_epoch", False)

        if is_eval:
            val_metrics = state.get("val_metrics") or {}
            best_val = state.get("best_val_metric")
            msg = (
                f"[{_ts()}] [PB] {fold_name} | ep {ep:03d} | "
                f"train_loss = {train_loss} | "
                f"val_loss = {_fmt(val_metrics.get('loss'), nd=5)} | "
                f"val_dice = {_fmt(val_metrics.get('dice'))} | "
                f"val_iou = {_fmt(val_metrics.get('iou'))}"
            )
            if best_val is not None:
                msg += f" | best_val_dice = {_fmt(best_val)}"
            sep = "=" * len(msg)
            logger.info(sep)
            logger.info(msg)
            logger.info(sep)
        else:
            msg = (
                f"[{_ts()}] [PB] {fold_name} | ep {ep:03d} | "
                f"train_loss = {train_loss}"
            )
            logger.info(msg)

    def on_fold_end(self, state) -> None:
        if not is_global_zero():
            return
        logger = get_logger()
        m = state.get("test_metrics") or {}
        logger.info(
            "[%s] %s |< TEST >| dice = %s | iou = %s | prec = %s | rec = %s",
            _ts(),
            state.get("fold_name", ""),
            _fmt(m.get("dice")),
            _fmt(m.get("iou")),
            _fmt(m.get("precision")),
            _fmt(m.get("recall")),
        )

    def on_train_end(self, state) -> None:
        if is_global_zero():
            get_logger().info("[PB] Training finished")
            