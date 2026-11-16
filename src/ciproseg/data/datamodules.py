#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SPDX-License-Identifier: Apache-2.0
# =======================================================================
#  Project       : CiproSeg - LOSO U-Net Segmentation Framework
#  File          : src/ciproseg/data/datamodules.py
#  Purpose       : Datamodules
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

from dataclasses import dataclass
from typing import Any, Dict, Optional

import torch
from torch.utils.data import DataLoader, Dataset

from .collate import default_collate_fn, segmentation_collate_fn


@dataclass
class DataModule:
    """
    Generic train/val/test datamodule. Not used during LOSO training. Trainer
    builds per fold datasets directly but available for single split workflows.
    """
    train: Dataset
    val: Optional[Dataset]
    test: Optional[Dataset]
    batch_size: int = 32
    num_workers: int = 4
    pin_memory: bool = True
    collate_fn: Any = default_collate_fn

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.train, batch_size=self.batch_size, shuffle=True,
                          num_workers=self.num_workers, pin_memory=self.pin_memory,
                          collate_fn=self.collate_fn)

    def val_dataloader(self) -> Optional[DataLoader]:
        if self.val is None:
            return None
        return DataLoader(self.val, batch_size=self.batch_size, shuffle=False,
                          num_workers=self.num_workers, pin_memory=self.pin_memory,
                          collate_fn=self.collate_fn)

    def test_dataloader(self) -> Optional[DataLoader]:
        if self.test is None:
            return None
        return DataLoader(self.test, batch_size=self.batch_size, shuffle=False,
                          num_workers=self.num_workers, pin_memory=self.pin_memory,
                          collate_fn=self.collate_fn)


class CiproSegDataModule:
    """
    Thin wrapper that centralises DataLoader settings for LOSO training.
    Trainer calls make_train_loader() per fold rather than managing
    batch_size, num_workers, and collate_fn inline.
    """

    def __init__(self, cfg: Dict[str, Any], device: str = "cuda") -> None:
        tcfg = cfg.get("trainer") or {}
        self.batch_size = int(tcfg.get("batch_size", 8))
        self.num_workers = int(tcfg.get("num_workers", 2))
        self.pin_memory = device == "cuda"
        self.collate_fn = segmentation_collate_fn

    def make_train_loader(self, train_ds: Dataset) -> DataLoader:
        return DataLoader(train_ds, batch_size=self.batch_size, shuffle=True,
                          num_workers=self.num_workers, pin_memory=self.pin_memory,
                          collate_fn=self.collate_fn)

    def make_eval_loader(self, ds: Dataset) -> DataLoader:
        # Reserved for patch based validation, full image eval bypasses this.
        return DataLoader(ds, batch_size=self.batch_size, shuffle=False,
                          num_workers=self.num_workers, pin_memory=self.pin_memory,
                          collate_fn=self.collate_fn)


def build_datamodule(cfg: Dict[str, Any]) -> CiproSegDataModule:
    """Factory used by Engine to build a CiproSegDataModule with the correct device."""
    force_cpu = bool((cfg.get("trainer") or {}).get("cpu", False))
    device = "cuda" if (torch.cuda.is_available() and not force_cpu) else "cpu"
    return CiproSegDataModule(cfg, device=device)
