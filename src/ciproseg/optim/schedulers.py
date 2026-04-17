#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SPDX-License-Identifier: Apache-2.0
# =======================================================================
#  Project       : CiproSeg - LOSO U-Net Segmentation Framework
#  File          : src/ciproseg/optim/schedulers.py
#  Purpose       : Scheduler
#
#  Copyright (c) 2026 Manaz Kaleel
#  Authors       : Manaz Kaleel <manaz.ucd@gmail.com>
#
#  Created on    : 2025-11-01
#  Last modified : 2026-04-15, MK
#  Version       : 0.1.0
#  License       : Apache-2.0
#  Repository    : https://github.com/manazucd/CiproSeg
# =======================================================================
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

import torch


@dataclass(frozen=True)
class SchedulerBundle:
    scheduler: torch.optim.lr_scheduler._LRScheduler
    interval: str  # "step" or "epoch"


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    cfg: Dict[str, Any],
    *,
    steps_per_epoch: Optional[int] = None,
    epochs: Optional[int] = None,
) -> Optional[SchedulerBundle]:
    """
    Builds a scheduler from config. Supported names: cosine, step, onecycle.

        sched:
            name: cosine       # or step / onecycle
            tmax: 40
            eta_min: 0.0
            warmup_epochs: 10  # optional linear warmup (cosine and step only)
    """
    scfg = cfg.get("sched") or {}
    name = str(scfg.get("name", "none")).lower()

    if name in ("none", "", "null"):
        return None

    if name == "cosine":
        cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=int(scfg.get("tmax", epochs if epochs is not None else 100)),
            eta_min=float(scfg.get("eta_min", 0.0)),
        )
        warmup_epochs = int(scfg.get("warmup_epochs", 0))
        if warmup_epochs > 0:
            warmup = torch.optim.lr_scheduler.LinearLR(
                optimizer,
                start_factor=1e-3,
                end_factor=1.0,
                total_iters=warmup_epochs,
            )
            sched = torch.optim.lr_scheduler.SequentialLR(
                optimizer,
                schedulers=[warmup, cosine],
                milestones=[warmup_epochs],
            )
        else:
            sched = cosine
        return SchedulerBundle(scheduler=sched, interval="epoch")

    if name == "step":
        sched = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=int(scfg.get("step_size", 10)),
            gamma=float(scfg.get("gamma", 0.1)),
        )
        return SchedulerBundle(scheduler=sched, interval="epoch")

    if name == "onecycle":
        if steps_per_epoch is None or epochs is None:
            raise ValueError("OneCycleLR requires steps_per_epoch and epochs.")
        sched = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=float(scfg.get("max_lr", 1e-3)),
            epochs=int(epochs),
            steps_per_epoch=int(steps_per_epoch),
            pct_start=float(scfg.get("pct_start", 0.1)),
            div_factor=float(scfg.get("div_factor", 25.0)),
            final_div_factor=float(scfg.get("final_div_factor", 1e4)),
        )
        return SchedulerBundle(scheduler=sched, interval="step")

    raise ValueError(f"Unknown scheduler: '{name}'")
