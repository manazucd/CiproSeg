#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SPDX-License-Identifier: Apache-2.0
# =======================================================================
#  Project       : CiproSeg - LOSO U-Net Segmentation Framework
#  File          : src/ciproseg/utils/dist.py
#  Purpose       : Dist
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

import os
from datetime import timedelta

import torch
import torch.distributed as dist


def init(backend: str = "nccl", timeout_s: int = 1800) -> bool:
    """Initialises torch.distributed when DDP env vars are present. Returns True if initialised."""
    if "RANK" not in os.environ:
        return False
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
    dist.init_process_group(backend=backend, timeout=timedelta(seconds=timeout_s))
    return True


def is_dist_initialized() -> bool:
    return dist.is_available() and dist.is_initialized()


def get_local_rank() -> int:
    return int(os.environ.get("LOCAL_RANK", 0))


def is_global_zero() -> bool:
    return not is_dist_initialized() or dist.get_rank() == 0
