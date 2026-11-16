#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SPDX-License-Identifier: Apache-2.0
# =======================================================================
#  Project       : CiproSeg - LOSO U-Net Segmentation Framework
#  File          : src/ciproseg/__init__.py
#  Purpose       : __init__.py
#
#  Copyright (c) 2026 Manaz Kaleel
#  Authors       : Manaz Kaleel <manaz.ucd@gmail.com>
#
#  Created on    : 2026-02-15
#  Last modified : 2026-02-18
#  Version       : 0.1.0
#  License       : Apache-2.0
#  Repository    : https://github.com/manazucd/CiproSeg
# =======================================================================
"""
CiproSeg: LOSO U-Net segmentation framework for microscopy image analysis.
PyTorch based single GPU research framework with Leave-One-Series-Out cross validation.
The basic infrastructure for DDP is in built for future upgrades.
"""

from __future__ import annotations

__version__ = "0.1.0"
__author__ = "Manaz Kaleel"
__license__ = "Apache-2.0"

from .utils.seed import seed_everything
from .utils.config import load_config

__all__ = [
    "__version__",
    "__author__",
    "__license__",
    "seed_everything",
    "load_config",
]
