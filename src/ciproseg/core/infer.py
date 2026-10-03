#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SPDX-License-Identifier: Apache-2.0
# =======================================================================
#  Project       : CiproSeg - LOSO U-Net Segmentation Framework
#  File          : src/ciproseg/core/infer.py
#  Purpose       : Standalone directory-level inference (no dataset index needed)
#
#  Copyright (c) 2026 Manaz Kaleel
#  Authors       : Manaz Kaleel <manaz.ucd@gmail.com>
#
#  Created on    : 2026-02-21
#  Last modified : 2026-10-03, MK
#  Version       : 0.1.0
#  License       : Apache-2.0
#  Repository    : https://github.com/manazucd/CiproSeg
# =======================================================================
from __future__ import annotations

import logging
import os
import tempfile
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
import torch

from ..models import build_model_from_cfg
from ..utils.device import resolve_device
from ..utils.io import read_image_gray, read_mask_01
from .evaluator import predict_full_mask_tiled

logger = logging.getLogger("ciproseg")

_IMAGE_EXTS: Tuple[str, ...] = (".tif", ".tiff", ".png", ".jpg", ".jpeg")

# Suffixes used when writing output artefacts (relative to stem of input file)
_SUFFIX_PROB_NPY = "_prob.npy"      # float32 probability map [0.0, 1.0]
_SUFFIX_PROB = "_prob.png"          # uint16 visualisation [0, 65535]
_SUFFIX_BINARY = "_binary.png"      # uint8 binary mask {0, 255}
_SUFFIX_DIFF = "_diff.png"          # TP/FP/FN colour map                                [optional]

_SUFFIX_PRED_OVERLAY = "_pred_overlay.png"   # original (as BGR) + predicted boundary in green
_SUFFIX_GT_OVERLAY = "_gt_overlay.png"       # original (as BGR) + GT boundary in cyan   [optional]

def load_model_from_checkpoint(
    ckpt_path: str,
    device: str = "cpu",
) -> torch.nn.Module:
    """
    Loads a CiproSeg checkpoint and reconstructs the model.
    """
    ckpt_path = str(ckpt_path)
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    state = torch.load(ckpt_path, map_location="cpu")

    missing = [k for k in ("model", "cfg") if k not in state]
    if missing:
        raise KeyError(
            f"Checkpoint {ckpt_path!r} is missing required keys: {missing}. "
            "Expected a CiproSeg checkpoint saved by Checkpointer."
        )

    cfg = state["cfg"]
    model = build_model_from_cfg(cfg)
    model.load_state_dict(state["model"])
    model.to(device)
    model.eval()

    logger.info(
        f"[infer] Loaded checkpoint: {ckpt_path}  "
        f"fold={state.get('fold_name', '?')}  "
        f"best_val_metric={state.get('best_val_metric', '?'):.4f}"
        if isinstance(state.get("best_val_metric"), float)
        else f"[infer] Loaded checkpoint: {ckpt_path}"
    )
    return model

def _to_uint8_gray(img_gray01: np.ndarray) -> np.ndarray:
    """Float [0,1] 2-D array → uint8 [0, 255] 2-D array."""
    return np.clip(img_gray01 * 255.0, 0, 255).astype(np.uint8)


def _prob_to_uint16(prob: np.ndarray) -> np.ndarray:
    """Float [0,1] probability map → uint16 [0, 65535]."""
    return np.clip(prob * 65535.0, 0, 65535).astype(np.uint16)


def _draw_boundary_on_bgr(
    bgr: np.ndarray,
    mask_binary_255: np.ndarray,
    colour: Tuple[int, int, int] = (0, 255, 0),
    thickness: int = 1,
) -> np.ndarray:
    """
    Overlays the boundary of 'mask_binary_255' onto 'bgr' (in-place copy).
    """
    out = bgr.copy()
    contours, _ = cv2.findContours(mask_binary_255, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(out, contours, -1, colour, thickness)
    return out


def _make_diff_map(
    pred_binary_255: np.ndarray,
    gt_binary_255: np.ndarray,
) -> np.ndarray:
    """
    Produces a colour-coded mask difference map (H, W, 3) in BGR uint8:
        TP  -- green      (0, 200, 0)    predicted foreground and GT foreground
        FP  -- red        (0, 0, 220)    predicted foreground where GT is background
        FN  -- blue       (220, 0, 0)    GT foreground missed by prediction
        TN  -- near-black (20, 20, 20)   background in both
    """
    pred = pred_binary_255 > 0
    gt = gt_binary_255 > 0

    h, w = pred.shape
    diff = np.full((h, w, 3), fill_value=20, dtype=np.uint8)  # TN default

    diff[gt & pred] = (0, 200, 0)    # TP  green
    diff[~gt & pred] = (0, 0, 220)   # FP  red  (BGR)
    diff[gt & ~pred] = (220, 0, 0)   # FN  blue (BGR)

    return diff


def _atomic_imwrite(path: str, img: np.ndarray) -> None:
    """Write image atomically via tmp file + os.replace (mirrors Checkpointer pattern)."""
    dirpath = os.path.dirname(path) or "."
    os.makedirs(dirpath, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=dirpath, suffix=os.path.splitext(path)[1])
    os.close(fd)
    ok = cv2.imwrite(tmp, img)
    if not ok:
        os.unlink(tmp)
        raise RuntimeError(f"cv2.imwrite failed for {path!r}")
    os.replace(tmp, path)


def save_inference_outputs(
    stem: str,
    out_dir: str,
    img_gray01: np.ndarray,
    prob: np.ndarray,
    threshold: float = 0.5,
    gt_mask01: Optional[np.ndarray] = None,
) -> List[str]:
    """
    Saves all output artefacts for a single image.
    """
    written: List[str] = []

    # Save the original floating point probability map [0.0, 1.0]
    prob_npy_path = os.path.join(out_dir, stem + _SUFFIX_PROB_NPY)
    np.save(prob_npy_path, prob.astype(np.float32, copy=False))
    written.append(prob_npy_path)

    binary_255 = ((prob >= threshold) * 255).astype(np.uint8)
    prob_u16 = _prob_to_uint16(prob)

    gray_u8 = _to_uint8_gray(img_gray01)                                 # uint8
    bgr = cv2.cvtColor(gray_u8, cv2.COLOR_GRAY2BGR)                      # (H,W,3) uint8

    prob_path = os.path.join(out_dir, stem + _SUFFIX_PROB)
    _atomic_imwrite(prob_path, prob_u16)
    written.append(prob_path)

    bin_path = os.path.join(out_dir, stem + _SUFFIX_BINARY)
    _atomic_imwrite(bin_path, binary_255)
    written.append(bin_path)

    pred_overlay = _draw_boundary_on_bgr(bgr, binary_255, colour=(0, 255, 0))
    pred_ov_path = os.path.join(out_dir, stem + _SUFFIX_PRED_OVERLAY)
    _atomic_imwrite(pred_ov_path, pred_overlay)
    written.append(pred_ov_path)

    if gt_mask01 is not None:
        gt_255 = (gt_mask01 > 0).astype(np.uint8) * 255

        gt_overlay = _draw_boundary_on_bgr(bgr, gt_255, colour=(255, 255, 0))  # cyan in BGR
        gt_ov_path = os.path.join(out_dir, stem + _SUFFIX_GT_OVERLAY)
        _atomic_imwrite(gt_ov_path, gt_overlay)
        written.append(gt_ov_path)

        diff = _make_diff_map(binary_255, gt_255)
        diff_path = os.path.join(out_dir, stem + _SUFFIX_DIFF)
        _atomic_imwrite(diff_path, diff)
        written.append(diff_path)

    return written


def _discover_images(input_dir: str) -> List[Path]:
    """
    Returns a sorted list of image Paths found directly inside 'input_dir'.
    Only files whose extension matches '_IMAGE_EXTS' are included.
    Does NOT recurse into subdirectories (mirrors dataset convention).
    """
    p = Path(input_dir)
    if not p.is_dir():
        raise NotADirectoryError(f"Input directory does not exist: {input_dir}")

    found = sorted(f for f in p.iterdir() if f.is_file() and f.suffix.lower() in _IMAGE_EXTS)
    if not found:
        raise FileNotFoundError(
            f"No images with extensions {_IMAGE_EXTS} found in: {input_dir}"
        )
    return found


def _find_gt_mask(img_path: Path, masks_dir: Optional[str]) -> Optional[str]:
    """
    Resolves a ground-truth mask path for 'img_path' when 'masks_dir' is given.

    Strategy: look for a file in 'masks_dir' whose stem matches the image stem
    (any supported extension). Returns None if not found (silent; GT is optional).
    """
    if masks_dir is None:
        return None

    md = Path(masks_dir)
    if not md.is_dir():
        logger.warning(f"[infer] --masks directory not found, skipping GT outputs: {masks_dir}")
        return None

    stem = img_path.stem
    for ext in _IMAGE_EXTS:
        candidate = md / (stem + ext)
        if candidate.is_file():
            return str(candidate)

    exact = md / img_path.name
    if exact.is_file():
        return str(exact)

    return None


def run_inference_on_dir(
    ckpt_path: str,
    input_dir: str,
    output_dir: str,
    threshold: float = 0.5,
    patch: int = 256,
    stride: Optional[int] = None,
    device_str: Optional[str] = None,
    force_cpu: bool = False,
    masks_dir: Optional[str] = None,
) -> None:
    """
    Runs tiled inference on every image in 'input_dir' and writes result artefacts
    to 'output_dir' (created if necessary) preserving original filenames as stems.
    """
    if stride is None:
        stride = patch // 2

    dev_info = resolve_device(
        prefer_cuda=True,
        force_cpu=force_cpu,
        device=device_str,
    )
    device = dev_info.device_str
    logger.info(f"[infer] Device: {device}")

    model = load_model_from_checkpoint(ckpt_path, device=device)

    images = _discover_images(input_dir)
    os.makedirs(output_dir, exist_ok=True)
    logger.info(f"[infer] Found {len(images)} image(s) in {input_dir!r}")
    logger.info(f"[infer] Outputs -> {output_dir!r}")
    logger.info(f"[infer] patch={patch}  stride={stride}  threshold={threshold:.3f}")

    for i, img_path in enumerate(images, start=1):
        logger.info(f"[infer] [{i}/{len(images)}] {img_path.name}")

        img_gray01 = read_image_gray(str(img_path))

        prob = predict_full_mask_tiled(
            model=model,
            img_gray01=img_gray01,
            patch=patch,
            stride=stride,
            device=device,
        )

        gt_path = _find_gt_mask(img_path, masks_dir)
        gt_mask01 = read_mask_01(gt_path) if gt_path is not None else None
        if gt_path is not None:
            logger.debug(f"[infer]   GT mask: {gt_path}")

        written = save_inference_outputs(
            stem=img_path.stem,
            out_dir=output_dir,
            img_gray01=img_gray01,
            prob=prob,
            threshold=threshold,
            gt_mask01=gt_mask01,
        )
        logger.debug(f"[infer]   Wrote {len(written)} file(s)")

    logger.info(f"[infer] Done. Processed {len(images)} image(s).")
