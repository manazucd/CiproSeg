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
#  Last modified : 2026-04-03, MK
#  Version       : 0.1.0
#  License       : Apache-2.0
#  Repository    : https://github.com/manazucd/CiproSeg
# =======================================================================
from __future__ import annotations

from typing import Any, Dict, Optional

from .trainer import Trainer
from .hooks import HookList
from .callbacks.progress_bar import ProgressBar
from ..utils.logger import get_logger
from ..data.datamodules import build_datamodule


class Engine:
    """
    Orchestrates training runs.

    Evaluation is handled separately via 'evaluate_checkpoint()' in
    'ciproseg.core.evaluator' (called by 'ciproseg-eval' CLI).
    """

    def __init__(self, cfg: Dict[str, Any]) -> None:
        self.cfg = cfg
        self.logger = get_logger()
        self.datamodule = build_datamodule(cfg)
        self.hooks = HookList([ProgressBar()])

        self._trainer: Optional[Trainer] = None

    @property
    def trainer(self) -> Trainer:
        """Lazily build the Trainer on first access."""
        if self._trainer is None:
            self._trainer = self._build_trainer()
        return self._trainer

    def _build_trainer(self) -> Trainer:
        try:
            return Trainer(self.cfg, self.datamodule, hooks=self.hooks)
        except TypeError:
            t = Trainer(self.cfg, self.datamodule)
            if hasattr(t, "hooks"):
                t.hooks = self.hooks
            return t

    def fit(self) -> None:
        self.logger.info("Starting training...")
        self.trainer.train()
        self.hooks.on_train_end({"cfg": self.cfg})
