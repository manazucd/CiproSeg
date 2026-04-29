#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SPDX-License-Identifier: Apache-2.0
# =======================================================================
#  Project       : CiproSeg - LOSO U-Net Segmentation Framework
#  File          : src/ciproseg/models/unet2.py
#  Purpose       : Small 2-level U-Net (reduced capacity) for microscopy segmentation
#
#  Copyright (c) 2026 Manaz Kaleel
#  Authors       : Manaz Kaleel <manaz.ucd@gmail.com>
#
#  Created on    : 2026-02-25
#  Last modified : 2026-04-29, MK
#  Version       : 0.1.0
#  License       : Apache-2.0
#  Repository    : https://github.com/manazucd/CiproSeg
# =======================================================================
from __future__ import annotations

import torch
import torch.nn as nn

from .registry import register_model


# Default GroupNorm group count. With LOSO and a small batch (e.g. 4)
# BatchNorm running statistics are unreliable, so GroupNorm or
# InstanceNorm are preferred when norm != "batch".
_DEFAULT_GN_GROUPS = 8


def _make_norm(num_features: int, kind: str) -> nn.Module:
    """
    Build a normalisation layer.

    Supported kinds:
        batch     BatchNorm2d (default, backward compatible)
        group     GroupNorm with up to _DEFAULT_GN_GROUPS groups, reduced
                  if the channel count does not divide evenly
        instance  InstanceNorm2d with affine parameters
    """
    if kind == "batch":
        return nn.BatchNorm2d(num_features)

    if kind == "group":
        n_groups = min(_DEFAULT_GN_GROUPS, num_features)
        while num_features % n_groups != 0 and n_groups > 1:
            n_groups -= 1
        return nn.GroupNorm(num_groups=n_groups, num_channels=num_features)

    if kind == "instance":
        return nn.InstanceNorm2d(num_features, affine=True, track_running_stats=False)

    raise ValueError(
        f"Unknown norm kind: '{kind}'. Expected one of: 'batch', 'group', 'instance'."
    )


class DoubleConv(nn.Module):
    """
    Conv2d -> Norm -> ReLU applied twice with the same channel width.

    When dropout_p > 0, an nn.Dropout2d layer is appended after the
    second ReLU. Channel dropout is preferred over per-pixel dropout
    in convolutional layers because it preserves spatial structure.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        dropout_p: float = 0.0,
        norm: str = "batch",
    ) -> None:
        super().__init__()
        if not (0.0 <= dropout_p < 1.0):
            raise ValueError(
                f"dropout_p must be in [0.0, 1.0), got {dropout_p}"
            )

        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            _make_norm(out_channels, norm),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            _make_norm(out_channels, norm),
            nn.ReLU(inplace=True),
        ]
        if dropout_p > 0.0:
            layers.append(nn.Dropout2d(p=dropout_p))

        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class UNet2(nn.Module):
    """
    2 level (2 down / 2 up) U-Net for binary segmentation.
    Returns raw logits; apply torch.sigmoid() externally.

    This is intentionally much smaller than the default 4-level U-Net.

    Optional dropout (dropout_p > 0) is applied only in the bottleneck
    and decoder DoubleConv blocks. Encoder dropout is intentionally
    disabled because it tends to degrade low-level feature extraction.

    The norm parameter selects the normalisation layer used inside
    every DoubleConv.
    """

    def __init__(
        self,
        in_channels: int = 1,
        out_channels: int = 1,
        base_channels: int = 32,
        dropout_p: float = 0.0,
        norm: str = "batch",
    ) -> None:
        super().__init__()
        b = base_channels

        # Encoder: no dropout (preserves low and mid level features).
        self.down1, self.pool1 = DoubleConv(in_channels, b, norm=norm), nn.MaxPool2d(2)
        self.down2, self.pool2 = DoubleConv(b, b * 2, norm=norm), nn.MaxPool2d(2)

        # Bottleneck: highest channel count, most prone to memorisation.
        self.bottleneck = DoubleConv(b * 2, b * 4, dropout_p=dropout_p, norm=norm)

        # Decoder: dropout applied here to fight overfitting on the
        # task-specific reconstruction path.
        self.upconv2 = nn.ConvTranspose2d(b * 4, b * 2, kernel_size=2, stride=2)
        self.up2 = DoubleConv(b * 4, b * 2, dropout_p=dropout_p, norm=norm)

        self.upconv1 = nn.ConvTranspose2d(b * 2, b, kernel_size=2, stride=2)
        self.up1 = DoubleConv(b * 2, b, dropout_p=dropout_p, norm=norm)

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
