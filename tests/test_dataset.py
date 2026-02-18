#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SPDX-License-Identifier: Apache-2.0
# =======================================================================
#  Project       : CiproSeg - LOSO U-Net Segmentation Framework
#  File          : tests/test_dataset.py
#  Purpose       : Test Dataset -> Unit test: white-box-ish
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

from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np

from ciproseg.data.datasets.ciproseg_ecoli import (
    build_index,
    make_loso_folds,
    build_fold_splits,
    read_image_gray,
    read_mask_01,
    RandomCropSegDataset,
    default_crops_per_frame,
)


def _write_png(path: Path, arr: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ok = cv2.imwrite(str(path), arr)
    assert ok, f"Failed to write image: {path}"


def _make_tiny_dataset(
    root: Path,
    series: List[str],
    n_frames: int = 3,
    hw: Tuple[int, int] = (64, 64),
) -> None:
    h, w = hw
    for s in series:
        for i in range(n_frames):
            img = np.random.randint(0, 255, size=(h, w), dtype=np.uint8)

            # Offset the foreground square per frame to simulate cell motion.
            mask = np.zeros((h, w), dtype=np.uint8)
            mask[10 + i : 22 + i, 12 + i : 24 + i] = 255

            _write_png(root / "images" / s / f"frame_{i:03d}.png", img)
            _write_png(root / "masks" / s / f"frame_{i:03d}.png", mask)


def test_build_index_and_readers(tmp_path: Path) -> None:
    _make_tiny_dataset(tmp_path, series=["S1", "S2"], n_frames=2)

    index = build_index(str(tmp_path), exts=(".png",))
    assert set(index.keys()) == {"S1", "S2"}
    assert len(index["S1"]) == 2

    item = index["S1"][0]
    img = read_image_gray(item.img_path)
    msk = read_mask_01(item.mask_path)

    assert img.dtype == np.float32
    assert img.min() >= 0.0 and img.max() <= 1.0
    assert msk.dtype == np.uint8
    assert set(np.unique(msk)).issubset({0, 1})


def test_loso_folds_and_splits(tmp_path: Path) -> None:
    _make_tiny_dataset(tmp_path, series=["S1", "S2", "S3"], n_frames=5)

    index = build_index(str(tmp_path), exts=(".png",))
    folds = make_loso_folds(index)
    assert len(folds) == 3

    test_s, train_s = folds[0]
    assert test_s in {"S1", "S2", "S3"}
    assert test_s not in train_s

    train_frames, val_frames, test_frames = build_fold_splits(
        index=index,
        test_series=test_s,
        train_series=train_s,
        train_frac=0.8,
    )
    assert len(test_frames) == 5
    assert len(train_frames) > 0
    assert len(val_frames) > 0


def test_random_crop_dataset_returns_tensors(tmp_path: Path) -> None:
    _make_tiny_dataset(tmp_path, series=["S1"], n_frames=4)
    index = build_index(str(tmp_path), exts=(".png",))
    frames = index["S1"]

    patch = 32
    K = default_crops_per_frame(num_train_frames=len(frames), target_patches_per_epoch=64, kmin=1, kmax=32)
    ds = RandomCropSegDataset(frames=frames, patch=patch, crops_per_frame=K, do_augment=False)

    sample = ds[0]
    if isinstance(sample, dict):
        x_t, y_t = sample["x"], sample["y"]
    else:
        x_t, y_t = sample

    assert x_t.shape == (1, patch, patch)
    assert y_t.shape == (1, patch, patch)
    assert x_t.dtype.is_floating_point
    assert y_t.dtype.is_floating_point