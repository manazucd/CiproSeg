#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SPDX-License-Identifier: Apache-2.0
# =======================================================================
#  Project       : CiproSeg - LOSO U-Net Segmentation Framework
#  File          : src/ciproseg/utils/device.py
#  Purpose       : Device selection
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
from typing import Optional

import torch

from .dist import is_dist_initialized, get_local_rank


@dataclass(frozen=True)
class DeviceInfo:
    device: torch.device
    device_str: str
    is_cuda: bool
    local_rank: int
    world_size: int


def _world_size() -> int:
    return torch.distributed.get_world_size() if is_dist_initialized() else 1


def _make(dev: torch.device, local_rank: int, world_size: int) -> DeviceInfo:
    return DeviceInfo(device=dev, device_str=str(dev), is_cuda=(dev.type == "cuda"),
                      local_rank=local_rank, world_size=world_size)


def _resolve_explicit_device(device: str) -> torch.device:
    # Under DDP, a bare "cuda" maps to the rank specific GPU, all other strings honoured as-is.
    if is_dist_initialized() and str(device).strip().lower() == "cuda":
        return torch.device(f"cuda:{get_local_rank()}")
    return torch.device(device)


def resolve_device(
    prefer_cuda: bool = True,
    force_cpu: bool = False,
    device: Optional[str] = None,
) -> DeviceInfo:
    """
    Resolves the torch device in priority order:
      1. force_cpu=True  (overrides everything)
      2. Explicit device argument
      3. DDP active -> cuda:{local_rank} (or cpu if CUDA unavailable)
      4. prefer_cuda and CUDA available -> cuda
      5. cpu
    """
    if force_cpu:
        return _make(torch.device("cpu"), 0, 1)

    if device is not None:
        dev = _resolve_explicit_device(device)
        if dev.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError(f"Requested device '{device}', but CUDA is not available.")
        if is_dist_initialized():
            return _make(dev, get_local_rank(), torch.distributed.get_world_size())
        return _make(dev, 0, 1)

    if is_dist_initialized():
        lr, ws = get_local_rank(), torch.distributed.get_world_size()
        return _make(torch.device(f"cuda:{lr}" if torch.cuda.is_available() else "cpu"), lr, ws)

    if prefer_cuda and torch.cuda.is_available():
        return _make(torch.device("cuda"), 0, 1)

    return _make(torch.device("cpu"), 0, 1)
