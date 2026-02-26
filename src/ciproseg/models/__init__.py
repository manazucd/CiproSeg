#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SPDX-License-Identifier: Apache-2.0
# =======================================================================
#  Project       : CiproSeg - LOSO U-Net Segmentation Framework
#  File          : src/ciproseg/models/__init__.py
#  Purpose       : Model package init, ensures all @register_model
#                  decorators are executed on first import of this package.
#
#  Copyright (c) 2026 Manaz Kaleel
#  Authors       : Manaz Kaleel <manaz.ucd@gmail.com>
#
#  Created on    : 2026-02-21
#  Last modified : 2026-02-26, MK
#  Version       : 0.1.0
#  License       : Apache-2.0
#  Repository    : https://github.com/manazucd/CiproSeg
# =======================================================================
from __future__ import annotations
from . import unet4 as _unet4
from . import unet2 as _unet2
from .registry import build_model_from_cfg, create_model, register_model

__all__ = [
    "build_model_from_cfg",
    "create_model",
    "register_model",
]
