#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SPDX-License-Identifier: Apache-2.0
# =======================================================================
#  Project       : CiproSeg - LOSO U-Net Segmentation Framework
#  File          : src/ciproseg/models/unet4.py
#  Purpose       : U-Net architecture for microscopy segmentation
#
#  Copyright (c) 2026 Manaz Kaleel
#  Authors       : Manaz Kaleel <manaz.ucd@gmail.com>
#
#  Created on    : 2026-02-18
#  Last modified : 2026-04-11, MK
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


class UNet4(nn.Module):
    """
    4 level U-Net for binary segmentation. Returns raw logits,
    apply torch.sigmoid() externally to obtain probabilities.
    """

    def __init__(self, in_channels: int = 1, out_channels: int = 1, base_channels: int = 64) -> None:
        super().__init__()
        b = base_channels

        self.down1, self.pool1 = DoubleConv(in_channels, b),      nn.MaxPool2d(2)
        self.down2, self.pool2 = DoubleConv(b,      b * 2),        nn.MaxPool2d(2)
        self.down3, self.pool3 = DoubleConv(b * 2,  b * 4),        nn.MaxPool2d(2)
        self.down4, self.pool4 = DoubleConv(b * 4,  b * 8),        nn.MaxPool2d(2)

        self.bottleneck = DoubleConv(b * 8, b * 16)

        self.upconv4 = nn.ConvTranspose2d(b * 16, b * 8, kernel_size=2, stride=2)
        self.up4     = DoubleConv(b * 16, b * 8)
        self.upconv3 = nn.ConvTranspose2d(b * 8,  b * 4, kernel_size=2, stride=2)
        self.up3     = DoubleConv(b * 8,  b * 4)
        self.upconv2 = nn.ConvTranspose2d(b * 4,  b * 2, kernel_size=2, stride=2)
        self.up2     = DoubleConv(b * 4,  b * 2)
        self.upconv1 = nn.ConvTranspose2d(b * 2,  b,     kernel_size=2, stride=2)
        self.up1     = DoubleConv(b * 2,  b)

        self.final_conv = nn.Conv2d(b, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        d1 = self.down1(x);  p1 = self.pool1(d1)
        d2 = self.down2(p1); p2 = self.pool2(d2)
        d3 = self.down3(p2); p3 = self.pool3(d3)
        d4 = self.down4(p3); p4 = self.pool4(d4)

        bn = self.bottleneck(p4)

        up4 = self.up4(torch.cat([self.upconv4(bn),  d4], dim=1))
        up3 = self.up3(torch.cat([self.upconv3(up4), d3], dim=1))
        up2 = self.up2(torch.cat([self.upconv2(up3), d2], dim=1))
        up1 = self.up1(torch.cat([self.upconv1(up2), d1], dim=1))

        return self.final_conv(up1)

    def get_num_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


@register_model("unet4")
def build_unet4_from_registry(**kwargs) -> UNet4:
    return UNet4(**kwargs)