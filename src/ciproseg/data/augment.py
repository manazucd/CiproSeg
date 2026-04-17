#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SPDX-License-Identifier: Apache-2.0
# =======================================================================
#  Project       : CiproSeg - LOSO U-Net Segmentation Framework
#  File          : src/ciproseg/data/augment.py
#  Purpose       : Augmentation utilities
#
#  Copyright (c) 2026 Manaz Kaleel
#  Authors       : Manaz Kaleel <manaz.ucd@gmail.com>
#
#  Created on    : 2025-11-01
#  Last modified : 2026-04-15, MK
#  Version       : 0.1.0
#  License       : Apache-2.0
#  Repository    : https://github.com/manazucd/CiproSeg
# =======================================================================
from __future__ import annotations

import random
from typing import Tuple

import cv2
import numpy as np


def compute_boundary_map(mask01: np.ndarray, ksize: int = 3) -> np.ndarray:
    """Morphological gradient boundary map, returns a uint8 array with values in {0, 1}."""
    if mask01.ndim != 2:
        raise ValueError(f"compute_boundary_map expects 2D mask, got {mask01.shape}")
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (ksize, ksize))
    grad = cv2.morphologyEx((mask01 > 0).astype(np.uint8) * 255, cv2.MORPH_GRADIENT, kernel)
    return (grad > 0).astype(np.uint8)


def _elastic_deform(
    img: np.ndarray,
    mask01: np.ndarray,
    alpha: float = 80.0,
    sigma: float = 10.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Apply random elastic deformation to an image/mask pair."""
    h, w = img.shape[:2]
    dx = cv2.GaussianBlur(
        (np.random.rand(h, w).astype(np.float32) * 2 - 1),
        ksize=(0, 0),
        sigmaX=sigma,
    ) * alpha
    dy = cv2.GaussianBlur(
        (np.random.rand(h, w).astype(np.float32) * 2 - 1),
        ksize=(0, 0),
        sigmaX=sigma,
    ) * alpha

    x_coords, y_coords = np.meshgrid(np.arange(w), np.arange(h))
    map_x = (x_coords + dx).astype(np.float32)
    map_y = (y_coords + dy).astype(np.float32)

    img = cv2.remap(img, map_x, map_y, interpolation=cv2.INTER_LINEAR,
                    borderMode=cv2.BORDER_REFLECT_101)
    mask01 = cv2.remap(mask01, map_x, map_y, interpolation=cv2.INTER_NEAREST,
                       borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    return img, mask01


def augment_pair(img: np.ndarray, mask01: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Geometric and photometric augmentation applied consistently to an image/mask pair."""
    if random.random() < 0.5:
        img, mask01 = np.fliplr(img).copy(), np.fliplr(mask01).copy()
    if random.random() < 0.5:
        img, mask01 = np.flipud(img).copy(), np.flipud(mask01).copy()

    if random.random() < 0.5:
        h, w = img.shape
        M = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), random.uniform(-15.0, 15.0), 1.0)
        img = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
        mask01 = cv2.warpAffine(mask01, M, (w, h), flags=cv2.INTER_NEAREST,
                                borderMode=cv2.BORDER_CONSTANT, borderValue=0)

    if random.random() < 0.3:
        img, mask01 = _elastic_deform(img, mask01)

    if random.random() < 0.7:
        img = np.clip(random.uniform(0.9, 1.1) * img + random.uniform(-0.05, 0.05), 0.0, 1.0)

    if random.random() < 0.5:
        sigma = random.uniform(0.0, 0.02)
        if sigma > 0:
            img = np.clip(img + np.random.normal(0.0, sigma, size=img.shape).astype(np.float32), 0.0, 1.0)

    return img, mask01


def sample_patch_xy(
    mask01: np.ndarray,
    patch: int,
    mode_probs: Tuple[float, float, float] = (0.6, 0.3, 0.1),
    min_fg_frac: float = 0.01,
    max_tries: int = 50,
) -> Tuple[int, int, str]:
    """
    Sample a patch top left (x, y) using a weighted mixture of foreground biased,
    boundary biased, and random strategies.
    """
    h, w = mask01.shape[:2]
    if h <= patch or w <= patch:
        return 0, 0, "fallback"

    p_fg, p_bd, _ = mode_probs
    r = random.random()
    mode = "foreground" if r < p_fg else ("boundary" if r < p_fg + p_bd else "random")

    if mode == "boundary":
        ys, xs = np.where(compute_boundary_map(mask01, ksize=3) > 0)
        if len(xs) < 10:
            mode = "foreground"

    if mode == "foreground":
        ys, xs = np.where(mask01 > 0)
        if len(xs) < 10:
            mode = "random"

    for _ in range(max_tries):
        if mode == "random":
            return int(random.randint(0, w - patch)), int(random.randint(0, h - patch)), "random"

        idx = int(random.randint(0, len(xs) - 1))
        x = int(np.clip(int(xs[idx]) - patch // 2, 0, w - patch))
        y = int(np.clip(int(ys[idx]) - patch // 2, 0, h - patch))

        if mode != "foreground" or float((mask01[y:y + patch, x:x + patch] > 0).mean()) >= min_fg_frac:
            return x, y, mode

    return int(random.randint(0, w - patch)), int(random.randint(0, h - patch)), "fallback"
