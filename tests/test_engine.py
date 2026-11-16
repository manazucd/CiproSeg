#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SPDX-License-Identifier: Apache-2.0
# =======================================================================
#  Project       : CiproSeg - LOSO U-Net Segmentation Framework
#  File          : tests/test_engine.py
#  Purpose       : Test engine -> Integration / smoke test: black-box-ish
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

from pathlib import Path
from typing import Any, Dict

import pytest

from ciproseg.core.engine import Engine


# Stubs replace the real Trainer and Evaluator via monkeypatch so the smoke test stays fast and free of GPU/data dependencies.
class _DummyTrainer:
    def __init__(self, cfg: Dict[str, Any], datamodule: Any = None, hooks: Any = None):
        self.cfg = cfg
        self.best_metric: float = 0.42
        self.last_metrics: Dict[str, float] = {
            "dice": 0.9,
            "iou": 0.8,
            "precision": 0.85,
            "recall": 0.88,
        }

    def train(self, start_state: Any = None) -> None:
        return


class _DummyEvaluator:
    def __init__(self, cfg: Dict[str, Any], datamodule: Any = None, hooks: Any = None):
        self.cfg = cfg

    def evaluate(self) -> Dict[str, float]:
        return {"num_series": 1.0, "num_frames": 3.0}

    def test(self) -> Dict[str, float]:
        return {"num_series": 1.0, "num_frames": 3.0}


def test_engine_fit_and_evaluate_smoke(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr("ciproseg.core.engine.Trainer", _DummyTrainer)

    cfg: Dict[str, Any] = {
        "trainer": {"output_dir": str(tmp_path)},
        "dataset": {"root": str(tmp_path)},
    }

    eng = Engine(cfg)
    eng.fit()

    assert (tmp_path / "run_summary.pt").exists(), "Checkpointer should write run_summary.pt after fit()"

    metrics = eng.evaluate()
    assert isinstance(metrics, dict)
    assert "num_series" in metrics
    