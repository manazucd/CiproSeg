#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SPDX-License-Identifier: Apache-2.0
# =======================================================================
#  Project       : CiproSeg - LOSO U-Net Segmentation Framework
#  File          : src/ciproseg/optim/optimizers.py
#  Purpose       : Optimiser
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

import torch
import torch.nn as nn


def _split_decay_params(model: nn.Module, weight_decay: float) -> list[dict]:
    """
    Separates parameters into decay and no decay groups.

    Excludes from weight decay:
        - Bias parameters
        - Parameters of normalization layers (BatchNorm, LayerNorm, GroupNorm, etc.)
    """

    norm_classes = (
        nn.BatchNorm1d,
        nn.BatchNorm2d,
        nn.BatchNorm3d,
        nn.LayerNorm,
        nn.GroupNorm,
        nn.InstanceNorm1d,
        nn.InstanceNorm2d,
        nn.InstanceNorm3d,
    )

    decay = []
    no_decay = []

    for module_name, module in model.named_modules():
        for param_name, param in module.named_parameters(recurse=False):

            if not param.requires_grad:
                continue

            full_name = f"{module_name}.{param_name}" if module_name else param_name

            if param_name.endswith("bias"):
                no_decay.append(param)
            elif isinstance(module, norm_classes):
                no_decay.append(param)
            else:
                decay.append(param)

    return [
        {"params": decay,    "weight_decay": float(weight_decay)},
        {"params": no_decay, "weight_decay": 0.0},
    ]


def build_optimizer(params_or_model: Any, cfg: Dict[str, Any]) -> torch.optim.Optimizer:
    """
    Builds an optimizer from config. Supported names: adam, adamw, sgd.

        optim:
            name: adamw
            lr: 1e-4
            weight_decay: 1e-4
            betas: [0.9, 0.999]
            eps: 1e-8
    """
    ocfg = cfg.get("optim") or {}
    name = str(ocfg.get("name", "adam")).lower()
    lr = float(ocfg.get("lr", 1e-3))
    wd = float(ocfg.get("weight_decay", 0.0))
    betas = tuple(ocfg.get("betas", (0.9, 0.999)))
    eps = float(ocfg.get("eps", 1e-8))
    momentum = float(ocfg.get("momentum", 0.9))

    if isinstance(params_or_model, nn.Module):
        params = _split_decay_params(params_or_model, wd) if wd > 0 else params_or_model.parameters()
        wd_fallback = 0.0
    else:
        params = params_or_model
        wd_fallback = wd

    if name == "adam":
        return torch.optim.Adam(params, lr=lr, betas=betas, eps=eps, weight_decay=wd_fallback)

    if name == "adamw":
        return torch.optim.AdamW(params, lr=lr, betas=betas, eps=eps, weight_decay=wd_fallback)

    if name == "sgd":
        return torch.optim.SGD(params, lr=lr, momentum=momentum, nesterov=True, weight_decay=wd_fallback)

    raise ValueError(f"Unknown optimizer: '{name}'")
