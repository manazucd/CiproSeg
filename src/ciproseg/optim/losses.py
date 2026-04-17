#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SPDX-License-Identifier: Apache-2.0
# =======================================================================
#  Project       : CiproSeg - LOSO U-Net Segmentation Framework
#  File          : src/ciproseg/optim/losses.py
#  Purpose       : Loss functions for binary segmentation
#
#  Copyright (c) 2026 Manaz Kaleel
#  Authors       : Manaz Kaleel <manaz.ucd@gmail.com>
#
#  Created on    : 2026-04-15
#  Last modified : 2026-04-15, MK
#  Version       : 0.1.0
#  License       : Apache-2.0
#  Repository    : https://github.com/manazucd/CiproSeg
# =======================================================================
from __future__ import annotations

from typing import Any, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    """Soft Dice loss operating on raw logits (applies sigmoid internally)."""

    def __init__(self, smooth: float = 1.0) -> None:
        super().__init__()
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits)
        pflat = probs.reshape(-1)
        tflat = targets.reshape(-1)
        intersection = (pflat * tflat).sum()
        return 1.0 - (2.0 * intersection + self.smooth) / (
            pflat.sum() + tflat.sum() + self.smooth
        )


class DiceBCELoss(nn.Module):
    """
    Weighted combination of Dice loss and BCE-with-logits loss.

    Both components receive raw logits; sigmoid is applied internally
    by each component so this plugs directly into the existing training
    loop without changes to the model output.
    """

    def __init__(
        self,
        dice_weight: float = 0.5,
        bce_weight: float = 0.5,
        smooth: float = 1.0,
    ) -> None:
        super().__init__()
        self.dice_weight = dice_weight
        self.bce_weight = bce_weight
        self.dice = DiceLoss(smooth=smooth)
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return (
            self.dice_weight * self.dice(logits, targets)
            + self.bce_weight * self.bce(logits, targets)
        )


def build_criterion(cfg: Dict[str, Any]) -> nn.Module:
    """
    Build a loss function from config.

    Supported values for loss.name:
        bce          - BCEWithLogitsLoss (current default, backward compatible)
        dice         - soft Dice loss only
        dice_bce     - weighted Dice + BCE combination

    Config example:
        loss:
          name: dice_bce
          dice_weight: 0.5
          bce_weight: 0.5

    When no loss section is present, falls back to BCEWithLogitsLoss
    so existing configs continue to work without modification.
    """
    lcfg = cfg.get("loss") or {}
    name = str(lcfg.get("name", "bce")).lower()

    if name == "bce":
        return nn.BCEWithLogitsLoss()

    if name == "dice":
        return DiceLoss(smooth=float(lcfg.get("smooth", 1.0)))

    if name == "dice_bce":
        return DiceBCELoss(
            dice_weight=float(lcfg.get("dice_weight", 0.5)),
            bce_weight=float(lcfg.get("bce_weight", 0.5)),
            smooth=float(lcfg.get("smooth", 1.0)),
        )

    raise ValueError(f"Unknown loss: '{name}'")
