#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SPDX-License-Identifier: Apache-2.0
# =======================================================================
#  Project       : CiproSeg - LOSO U-Net Segmentation Framework
#  File          : src/ciproseg/models/unet_small2.py
#  Purpose       : Small 2-level U-Net (reduced capacity) for microscopy segmentation
#
#  Copyright (c) 2026 Manaz Kaleel
#  Authors       : Manaz Kaleel <manaz.ucd@gmail.com>
#
#  Created on    : 2026-02-25, MK
#  Last modified : 2026-02-25, MK
#  Version       : 0.1.0
#  License       : Apache-2.0
#  Repository    : https://github.com/manazucd/CiproSeg
# =======================================================================
from __future__ import annotations

import torch
import torch.nn as nn

from .registry import register_model


class DoubleConv(nn.Module):
    """Conv2d -> BN -> ReLU applied twice with the same channel width."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class UNet2(nn.Module):
    """
    2 level (2 down / 2 up) U-Net for binary segmentation.
    Returns raw logits; apply torch.sigmoid() externally.

    This is intentionally much smaller than the default 4-level U-Net.
    """

    def __init__(self, in_channels: int = 1, out_channels: int = 1, base_channels: int = 32) -> None:
        super().__init__()
        b = base_channels

        self.down1, self.pool1 = DoubleConv(in_channels, b), nn.MaxPool2d(2)
        self.down2, self.pool2 = DoubleConv(b, b * 2), nn.MaxPool2d(2)

        self.bottleneck = DoubleConv(b * 2, b * 4)

        self.upconv2 = nn.ConvTranspose2d(b * 4, b * 2, kernel_size=2, stride=2)
        self.up2 = DoubleConv(b * 4, b * 2)  

        self.upconv1 = nn.ConvTranspose2d(b * 2, b, kernel_size=2, stride=2)
        self.up1 = DoubleConv(b * 2, b)     

        self.final_conv = nn.Conv2d(b, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        d1 = self.down1(x)
        p1 = self.pool1(d1)

        d2 = self.down2(p1)
        p2 = self.pool2(d2)

        bn = self.bottleneck(p2)

        up2 = self.up2(torch.cat([self.upconv2(bn), d2], dim=1))
        up1 = self.up1(torch.cat([self.upconv1(up2), d1], dim=1))

        return self.final_conv(up1)

    def get_num_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


@register_model("unet2")
def build_unet2_from_registry(**kwargs) -> UNet2:
    return UNet2(**kwargs)
