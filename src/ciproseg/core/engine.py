#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SPDX-License-Identifier: Apache-2.0
# =======================================================================
#  Project       : CiproSeg - LOSO U-Net Segmentation Framework
#  File          : src/ciproseg/core/engine.py
#  Purpose       : Engine
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

from typing import Any, Dict

from .trainer import Trainer
from .evaluator import Evaluator
from .hooks import HookList
from .callbacks.progress_bar import ProgressBar
from ..utils.checkpoint import Checkpointer
from ..utils.dist import is_global_zero
from ..utils.logger import get_logger
from ..data.datamodules import build_datamodule


class Engine:
    """
    Orchestrates training, evaluation, and test runs.
    Contains no task logic, all LOSO splitting, tiling, and checkpointing
    are delegated to Trainer, Evaluator, and Checkpointer respectively.
    """

    def __init__(self, cfg: Dict[str, Any]) -> None:
        self.cfg = cfg
        self.logger = get_logger()
        self.datamodule = build_datamodule(cfg)
        self.hooks = HookList([ProgressBar()])
        self.ckpt = Checkpointer(cfg)
        self.trainer = self._build_trainer()
        self.evaluator = self._build_evaluator()

        if hasattr(self.trainer, "ckpt"):
            self.trainer.ckpt = self.ckpt

    def _build_trainer(self) -> Trainer:
        # Gracefully handle older Trainer signatures that predate the hooks argument.
        try:
            return Trainer(self.cfg, self.datamodule, hooks=self.hooks)
        except TypeError:
            t = Trainer(self.cfg, self.datamodule)
            if hasattr(t, "hooks"):
                t.hooks = self.hooks
            return t

    def _build_evaluator(self) -> Evaluator:
        # Gracefully handle older Evaluator signatures that predate the hooks argument.
        try:
            return Evaluator(self.cfg, self.datamodule, hooks=self.hooks)
        except TypeError:
            e = Evaluator(self.cfg, self.datamodule)
            if hasattr(e, "hooks"):
                e.hooks = self.hooks
            return e

    def fit(self) -> None:
        self.logger.info("Starting training...")
        start_state = self.ckpt.maybe_load(self.trainer)
        self.hooks.on_train_start({"cfg": self.cfg})

        # Trainer owns per fold checkpointing while Engine writes the run summary.
        self.trainer.train(start_state=start_state)
        self.hooks.on_train_end({"cfg": self.cfg})

        if is_global_zero() and hasattr(self.ckpt, "save_run_summary"):
            self.ckpt.save_run_summary(
                self.trainer,
                metric_value=float(getattr(self.trainer, "best_metric", float("-inf"))),
            )

    def evaluate(self) -> Dict[str, float]:
        self.logger.info("Running evaluation...")
        self.hooks.on_eval_start({"cfg": self.cfg})
        metrics = self.evaluator.evaluate()
        self.hooks.on_eval_end({"cfg": self.cfg, "metrics": metrics})
        if is_global_zero():
            self.logger.info(f"Eval metrics: {metrics}")
        return metrics

    def test(self) -> Dict[str, float]:
        self.logger.info("Running test...")
        self.hooks.on_eval_start({"cfg": self.cfg})
        metrics = self.evaluator.test()
        self.hooks.on_eval_end({"cfg": self.cfg, "metrics": metrics})
        if is_global_zero():
            self.logger.info(f"Test metrics: {metrics}")
        return metrics
    