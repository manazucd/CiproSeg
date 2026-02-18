#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SPDX-License-Identifier: Apache-2.0
# =======================================================================
#  Project       : CiproSeg - LOSO U-Net Segmentation Framework
#  File          : src/ciproseg/data/datasets/ciproseg_ecoli.py
#  Purpose       : Dataset
#
#  Copyright (c) 2026 Manaz Kaleel, Team NexoraLab
#  Authors       : Manaz Kaleel <manaz.ucd@gmail.com>, Team NexoraLab
#
#  Created on    : 2025-11-01
#  Last modified : 2026-02-18, MK
#  Version       : 0.1.0
#  License       : Apache-2.0
#  Repository    : https://github.com/manazucd/CiproSeg
# =======================================================================

from __future__ import annotations

import os
import math
from dataclasses import dataclass
from typing import Dict, List, Tuple, Any

import numpy as np
import torch
from torch.utils.data import Dataset

from ...utils.io import read_image_gray, read_mask_01
from ...utils.patches import pad_to_patch
from ..augment import augment_pair, sample_patch_xy


@dataclass(frozen=True)
class FrameItem:
    series: str
    fname: str
    img_path: str
    mask_path: str


# Indexing / splits (LOSO)
def list_series(dataset_root: str) -> List[str]:
    images_root = os.path.join(dataset_root, "images")
    if not os.path.isdir(images_root):
        raise FileNotFoundError(f"Missing folder: {images_root}")
    series = sorted([d for d in os.listdir(images_root) if os.path.isdir(os.path.join(images_root, d))])
    if not series:
        raise RuntimeError(f"No series folders found under {images_root}")
    return series


def build_index(
    dataset_root: str,
    exts: Tuple[str, ...] = (".tif", ".tiff", ".png", ".jpg", ".jpeg"),
) -> Dict[str, List[FrameItem]]:
    images_root = os.path.join(dataset_root, "images")
    masks_root = os.path.join(dataset_root, "masks")
    if not os.path.isdir(images_root):
        raise FileNotFoundError(f"Missing folder: {images_root}")
    if not os.path.isdir(masks_root):
        raise FileNotFoundError(f"Missing folder: {masks_root}")

    series_images = set([d for d in os.listdir(images_root) if os.path.isdir(os.path.join(images_root, d))])
    series_masks = set([d for d in os.listdir(masks_root) if os.path.isdir(os.path.join(masks_root, d))])
    if series_images != series_masks:
        raise RuntimeError(
            "Series folders mismatch between images/ and masks/.\n"
            f"Missing in masks/: {sorted(series_images - series_masks)}\n"
            f"Missing in images/: {sorted(series_masks - series_images)}"
        )

    index: Dict[str, List[FrameItem]] = {}
    for s in sorted(series_images):
        img_dir = os.path.join(images_root, s)
        msk_dir = os.path.join(masks_root, s)

        img_files = sorted([f for f in os.listdir(img_dir) if f.lower().endswith(exts)])
        msk_files = sorted([f for f in os.listdir(msk_dir) if f.lower().endswith(exts)])

        if img_files != msk_files:
            imgs_set, msks_set = set(img_files), set(msk_files)
            raise RuntimeError(
                f"Filename mismatch in series '{s}'.\n"
                f"Only in images/: {sorted(list(imgs_set - msks_set))[:10]}\n"
                f"Only in masks/: {sorted(list(msks_set - imgs_set))[:10]}"
            )

        items: List[FrameItem] = []
        for f in img_files:
            items.append(
                FrameItem(
                    series=s,
                    fname=f,
                    img_path=os.path.join(img_dir, f),
                    mask_path=os.path.join(msk_dir, f),
                )
            )
        index[s] = items

    return index


def split_time_series(items: List[FrameItem], train_frac: float = 0.8) -> Tuple[List[FrameItem], List[FrameItem]]:
    n = len(items)
    if n < 2:
        return items, []
    cut = max(1, int(math.floor(n * train_frac)))
    cut = min(cut, n - 1)
    return items[:cut], items[cut:]


def make_loso_folds(index: Dict[str, List[FrameItem]]) -> List[Tuple[str, List[str]]]:
    series = sorted(index.keys())
    return [(test_s, [s for s in series if s != test_s]) for test_s in series]


def build_fold_splits(
    index: Dict[str, List[FrameItem]],
    test_series: str,
    train_series: List[str],
    train_frac: float = 0.8,
) -> Tuple[List[FrameItem], List[FrameItem], List[FrameItem]]:
    train_frames: List[FrameItem] = []
    val_frames: List[FrameItem] = []
    for s in train_series:
        tr, va = split_time_series(index[s], train_frac=train_frac)
        train_frames.extend(tr)
        val_frames.extend(va)
    test_frames = index[test_series]
    return train_frames, val_frames, test_frames


class RandomCropSegDataset(Dataset):
    """
    On the fly random crop dataset.
    length = len(frames) * crops_per_frame

    Returns a dict:
      {"x": (1,H,W) float32, "y": (1,H,W) float32, "meta": {...}} (optional)
    """

    def __init__(
        self,
        frames: List[FrameItem],
        patch: int = 256,
        crops_per_frame: int = 8,
        fg_frac: float = 0.01,
        mode_probs: Tuple[float, float, float] = (0.6, 0.3, 0.1),
        do_augment: bool = True,
        return_meta: bool = True,
    ):
        self.frames = frames
        self.patch = patch
        self.crops_per_frame = crops_per_frame
        self.fg_frac = fg_frac
        self.mode_probs = mode_probs
        self.do_augment = do_augment
        self.return_meta = return_meta

    def __len__(self) -> int:
        return max(1, len(self.frames) * self.crops_per_frame)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        frame = self.frames[idx // self.crops_per_frame]
        img = read_image_gray(frame.img_path)
        mask01 = read_mask_01(frame.mask_path)

        x, y, mode = sample_patch_xy(
            mask01=mask01,
            patch=self.patch,
            mode_probs=self.mode_probs,
            min_fg_frac=self.fg_frac,
            max_tries=60,
        )

        img_c = img[y : y + self.patch, x : x + self.patch]
        m_c = mask01[y : y + self.patch, x : x + self.patch]

        if img_c.shape[0] != self.patch or img_c.shape[1] != self.patch:
            img_c, m_c = pad_to_patch(img_c, m_c, self.patch)

        if self.do_augment:
            img_c, m_c = augment_pair(img_c, m_c)

        x_t = torch.from_numpy(img_c).unsqueeze(0).float()                   # (1,H,W)
        y_t = torch.from_numpy(m_c.astype(np.float32)).unsqueeze(0).float()  # (1,H,W)

        out: Dict[str, Any] = {"x": x_t, "y": y_t}

        if self.return_meta:
            out["meta"] = {
                "series": frame.series,
                "fname": frame.fname,
                "mode": mode,
                "xy": (int(x), int(y)),
                "patch": int(self.patch),
                "index": int(idx),
            }

        return out


def default_crops_per_frame(
    num_train_frames: int,
    target_patches_per_epoch: int = 2000,
    kmin: int = 4,
    kmax: int = 32,
) -> int:
    if num_train_frames <= 0:
        return kmin
    k = int(math.ceil(target_patches_per_epoch / num_train_frames))
    return max(kmin, min(kmax, k))
