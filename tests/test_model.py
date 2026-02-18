#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SPDX-License-Identifier: Apache-2.0
# =======================================================================
#  Project       : CiproSeg - LOSO U-Net Segmentation Framework
#  File          : tests/test_model.py
#  Purpose       : Test model -> Unit test: white-box-ish
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

import torch

from ciproseg.models.unet import UNet

_MODEL = UNet(in_channels=1, out_channels=1, base_channels=16)


def test_unet_forward_shape_logits() -> None:
    x = torch.randn(2, 1, 128, 128)
    logits = _MODEL(x)

    assert logits.shape == (2, 1, 128, 128)
    # Raw logits must not be sigmoid-squashed: at least one value should fall outside [0, 1].
    assert logits.min() < 0.0 or logits.max() > 1.0


def test_unet_num_params_positive() -> None:
    n = _MODEL.get_num_params()
    assert isinstance(n, int)
    assert n > 0