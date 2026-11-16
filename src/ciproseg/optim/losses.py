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
#  Last modified : 2026-04-29, MK
#  Version       : 0.1.0
#  License       : Apache-2.0
#  Repository    : https://github.com/manazucd/CiproSeg
# =======================================================================
from __future__ import annotations

from typing import Any, Dict

import torch
import torch.nn as nn


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


class TverskyLoss(nn.Module):
    """
    Tversky loss with separately weighted FP and FN terms.

    The Tversky index for a binary segmentation is

        TI = TP / (TP + alpha * FP + beta * FN)

    The loss is 1 - TI. Setting alpha < beta penalises false negatives
    more than false positives, which is appropriate when the model is
    under-segmenting (e.g. CiproSeg fold 03 worst slices, where FN
    dominates the error budget).

    Recommended starting points:
        alpha=0.5, beta=0.5  -> equivalent to Dice loss
        alpha=0.3, beta=0.7  -> mild FN penalty
        alpha=0.2, beta=0.8  -> heavy FN penalty
    """

    def __init__(
        self,
        alpha: float = 0.3,
        beta: float = 0.7,
        smooth: float = 1.0,
    ) -> None:
        super().__init__()
        if not (0.0 <= alpha <= 1.0):
            raise ValueError(f"alpha must be in [0, 1], got {alpha}")
        if not (0.0 <= beta <= 1.0):
            raise ValueError(f"beta must be in [0, 1], got {beta}")
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits)
        pflat = probs.reshape(-1)
        tflat = targets.reshape(-1)

        tp = (pflat * tflat).sum()
        fp = (pflat * (1.0 - tflat)).sum()
        fn = ((1.0 - pflat) * tflat).sum()

        tversky_index = (tp + self.smooth) / (
            tp + self.alpha * fp + self.beta * fn + self.smooth
        )
        return 1.0 - tversky_index


class FocalTverskyLoss(nn.Module):
    """
    Focal Tversky loss: (1 - TI) ** gamma.

    Adds a focusing factor gamma > 1 that increases the gradient on
    hard examples (low Tversky index) relative to easy ones. With
    gamma=1.0 this reduces to plain Tversky.

    Recommended starting points:
        alpha=0.3, beta=0.7, gamma=1.33  -> mild focal weighting
        alpha=0.3, beta=0.7, gamma=2.0   -> strong focal weighting
    """

    def __init__(
        self,
        alpha: float = 0.3,
        beta: float = 0.7,
        gamma: float = 1.33,
        smooth: float = 1.0,
    ) -> None:
        super().__init__()
        if not (0.0 <= alpha <= 1.0):
            raise ValueError(f"alpha must be in [0, 1], got {alpha}")
        if not (0.0 <= beta <= 1.0):
            raise ValueError(f"beta must be in [0, 1], got {beta}")
        if gamma <= 0.0:
            raise ValueError(f"gamma must be > 0, got {gamma}")
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits)
        pflat = probs.reshape(-1)
        tflat = targets.reshape(-1)

        tp = (pflat * tflat).sum()
        fp = (pflat * (1.0 - tflat)).sum()
        fn = ((1.0 - pflat) * tflat).sum()

        tversky_index = (tp + self.smooth) / (
            tp + self.alpha * fp + self.beta * fn + self.smooth
        )

        return torch.pow(torch.clamp(1.0 - tversky_index, min=1e-7), self.gamma)


def build_criterion(cfg: Dict[str, Any]) -> nn.Module:
    """
    Build a loss function from config.

    Supported values for loss.name:
        bce            BCEWithLogitsLoss (default, backward compatible)
        dice           Soft Dice loss
        dice_bce       Weighted Dice + BCE combination
        tversky        Tversky loss with alpha/beta weighting
        focal_tversky  Focal Tversky with alpha/beta/gamma

    Config examples:
        loss:
          name: tversky
          alpha: 0.3
          beta: 0.7

        loss:
          name: focal_tversky
          alpha: 0.3
          beta: 0.7
          gamma: 1.33

    When no loss section is present, falls back to BCEWithLogitsLoss.
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

    if name == "tversky":
        return TverskyLoss(
            alpha=float(lcfg.get("alpha", 0.3)),
            beta=float(lcfg.get("beta", 0.7)),
            smooth=float(lcfg.get("smooth", 1.0)),
        )

    if name == "focal_tversky":
        return FocalTverskyLoss(
            alpha=float(lcfg.get("alpha", 0.3)),
            beta=float(lcfg.get("beta", 0.7)),
            gamma=float(lcfg.get("gamma", 1.33)),
            smooth=float(lcfg.get("smooth", 1.0)),
        )

    raise ValueError(f"Unknown loss: '{name}'")
