#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SPDX-License-Identifier: Apache-2.0
# =======================================================================
#  Project       : CiproSeg - LOSO U-Net Segmentation Framework
#  File          : tests/test_checkpoint.py
#  Purpose       : Test checkpoint -> Unit test: white-box-ish
#
#  Copyright (c) 2026 Manaz Kaleel
#  Authors       : Manaz Kaleel <manaz.ucd@gmail.com>
#
#  Created on    : 2025-11-01
#  Last modified : 2026-10-03, MK
#  Version       : 0.1.0
#  License       : Apache-2.0
#  Repository    : https://github.com/manazucd/CiproSeg
# =======================================================================
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import torch

from ciproseg.utils.checkpoint import Checkpointer


# Minimal trainer stub, exposes only what Checkpointer reads during save/load.
class _DummyTrainer:
    best_metric: float = 0.123
    last_metrics: Dict[str, float] = {
        "dice": 0.9,
        "iou": 0.8,
        "precision": 0.85,
        "recall": 0.88,
    }

def _make_checkpointer(tmp_path: Path) -> Checkpointer:
    cfg: Dict[str, Any] = {"trainer": {"output_dir": str(tmp_path)}}
    return Checkpointer(cfg)


def test_checkpointer_saves_run_summary(tmp_path: Path) -> None:
    ckpt = _make_checkpointer(tmp_path)
    tr = _DummyTrainer()
    ckpt.save_run_summary(tr, metric_value=tr.best_metric)

    pt_path = tmp_path / "run_summary.pt"
    js_path = tmp_path / "run_summary.json"

    assert pt_path.exists(), "Expected run_summary.pt to be written"
    assert js_path.exists(), "Expected run_summary.json to be written"

    state = torch.load(pt_path, map_location="cpu")
    assert "best_validation_dice_across_folds" in state
    assert "mean_test_metrics_across_folds" in state
    assert state["mean_test_metrics_across_folds"]["dice"] == tr.last_metrics["dice"]

    with js_path.open("r", encoding="utf-8") as fh:
        j = json.load(fh)

    assert "best_validation_dice_across_folds" in j
    assert "mean_test_metrics_across_folds" in j
    assert abs(
        float(j["best_validation_dice_across_folds"]) - tr.best_metric
    ) < 1e-12


def test_checkpointer_maybe_load_when_missing(tmp_path: Path) -> None:
    ckpt = _make_checkpointer(tmp_path)
    assert ckpt.maybe_load(_DummyTrainer()) is None