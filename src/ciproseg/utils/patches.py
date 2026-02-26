#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SPDX-License-Identifier: Apache-2.0
# =======================================================================
#  Project       : CiproSeg - LOSO U-Net Segmentation Framework
#  File          : src/ciproseg/utils/patches.py
#  Purpose       : Patches
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

from typing import Tuple

import cv2
import numpy as np


def pad_to_patch(img: np.ndarray, mask01: np.ndarray, patch: int) -> Tuple[np.ndarray, np.ndarray]:
    """Pads img/mask to at least (patch, patch). image uses reflect padding, mask uses zero padding."""
    if img.ndim != 2 or mask01.ndim != 2:
        raise ValueError(f"pad_to_patch expects 2D arrays, got img={img.shape}, mask={mask01.shape}")

    h, w = img.shape
    pad_h, pad_w = max(0, patch - h), max(0, patch - w)
    if pad_h == 0 and pad_w == 0:
        return img, mask01

    img = cv2.copyMakeBorder(img, 0, pad_h, 0, pad_w, borderType=cv2.BORDER_REFLECT_101)
    mask01 = cv2.copyMakeBorder(mask01, 0, pad_h, 0, pad_w, borderType=cv2.BORDER_CONSTANT, value=0)
    return img[:patch, :patch], mask01[:patch, :patch]
