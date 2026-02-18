#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SPDX-License-Identifier: Apache-2.0
# =======================================================================
#  Project       : CiproSeg - LOSO U-Net Segmentation Framework
#  File          : src/ciproseg/utils/seed.py
#  Purpose       : Seed
#
#  Copyright (c) 2026 Manaz Kaleel
#  Authors       : Manaz Kaleel <manaz.ucd@gmail.com>
#
#  Created on    : 2026-02-15
#  Last modified : 2026-02-17
#  Version       : 0.1.0
#  License       : Apache-2.0
#  Repository    : https://github.com/manazucd/CiproSeg
# =======================================================================
from __future__ import annotations

import os
import random
import warnings

import numpy as np
import torch


def seed_everything(
    seed: int,
    deterministic: bool = False,
    benchmark: bool = True,
    set_env: bool = True,
) -> None:
    """
    Seeds Python, NumPy, and PyTorch (CPU + CUDA) for reproducibility.
    Note: deterministic=True may be slower and can raise on non deterministic ops.
    """
    if set_env:
        os.environ["PYTHONHASHSEED"] = str(seed)

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = bool(deterministic)
    torch.backends.cudnn.benchmark = bool(benchmark)

    if deterministic:
        # Required for CUDA matmul determinism on some configurations.
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        try:
            torch.use_deterministic_algorithms(True)
        except Exception as e:
            warnings.warn(f"Could not enable deterministic algorithms: {e}")
    else:
        try:
            torch.use_deterministic_algorithms(False)
        except Exception:
            pass
        