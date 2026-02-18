#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SPDX-License-Identifier: Apache-2.0
# =======================================================================
#  Project       : CiproSeg - LOSO U-Net Segmentation Framework
#  File          : tests/test_loss.py
#  Purpose       : Test loss -> Unit test: white-box-ish
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
import torch.nn as nn


def test_bce_with_logits_loss_backward() -> None:
    criterion = nn.BCEWithLogitsLoss()
    logits = torch.randn(4, 1, 64, 64, requires_grad=True)
    target = (torch.rand(4, 1, 64, 64) > 0.5).float()

    loss = criterion(logits, target)
    assert loss.ndim == 0  # scalar loss
    loss.backward()
    assert logits.grad is not None
    assert torch.isfinite(logits.grad).all()