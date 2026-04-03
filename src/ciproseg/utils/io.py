#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SPDX-License-Identifier: Apache-2.0
# =======================================================================
#  Project       : CiproSeg - LOSO U-Net Segmentation Framework
#  File          : src/ciproseg/utils/io.py
#  Purpose       : I/O
#
#  Copyright (c) 2026 Manaz Kaleel
#  Authors       : Manaz Kaleel <manaz.ucd@gmail.com>
#
#  Created on    : 2026-02-15
#  Last modified : 2026-04-03, MK
#  Version       : 0.1.0
#  License       : Apache-2.0
#  Repository    : https://github.com/manazucd/CiproSeg
# =======================================================================
from __future__ import annotations

import contextlib
import io
import logging
import warnings
from pathlib import Path

import cv2
import numpy as np
import tifffile as tiff

# Suppress tifffile's noisy ImageJ metadata messages without swallowing real errors.
_NOISY_SUBSTR = "ImageJ series metadata invalid or corrupted file"


@contextlib.contextmanager
def _suppress_tifffile_noise():
    """Silences known tifffile ImageJ metadata noise via warnings, logger, and stderr."""
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=f".*{_NOISY_SUBSTR}.*")
        tif_logger = logging.getLogger("tifffile")
        prev_level = tif_logger.level
        tif_logger.setLevel(logging.ERROR)
        try:
            with contextlib.redirect_stderr(io.StringIO()):
                yield
        finally:
            tif_logger.setLevel(prev_level)


def _imread_gray_any(path: str) -> np.ndarray:
    """Reads any supported image format and returns a 2D grayscale array."""
    ext = Path(str(path)).suffix.lower()

    if ext in {".tif", ".tiff"}:
        try:
            with _suppress_tifffile_noise():
                img = tiff.imread(path)
        except ValueError as e:
            if "requires the 'imagecodecs' package" in str(e):
                raise ValueError(
                    f"\nTIFF read failed for: {path}\n"
                    "This TIFF uses LZW (or similar) compression.\n"
                    "Fix: pip install imagecodecs\n"
                    "  or conda install -c conda-forge imagecodecs\n"
                ) from e
            raise
        except Exception as e:
            raise RuntimeError(f"Failed to read image: {path} ({type(e).__name__}: {e})") from e
    else:
        img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise RuntimeError(f"Failed to read image (cv2): {path}")

    if img.ndim == 3:
        if img.shape[-1] in (1, 3, 4):
            img = img[..., 0]
        elif img.shape[0] in (1, 3, 4):
            img = img[0, ...]
        else:
            img = img[..., 0]

    if img.ndim != 2:
        raise ValueError(f"Expected 2D grayscale after conversion, got shape {img.shape} for {path}")

    return img


def _dtype_max(img: np.ndarray) -> float:
    """Return the theoretical maximum for the image's dtype."""
    if img.dtype == np.uint8:
        return 255.0
    if img.dtype == np.uint16:
        return 65535.0
    if img.dtype == np.uint32:
        return 4294967295.0
    if img.dtype.kind == "f":
        return 1.0

    if np.issubdtype(img.dtype, np.integer):
        return float(np.iinfo(img.dtype).max)
    return 1.0


def read_image_gray(path: str) -> np.ndarray:
    """Reads an image as float32 normalised to [0, 1]."""
    img = _imread_gray_any(path)
    if img.size == 0:
        return img.astype(np.float32)

    divisor = _dtype_max(img)
    img = img.astype(np.float32)
    if divisor > 1.0:
        img /= divisor

    return img


def read_mask_01(path: str) -> np.ndarray:
    """Reads a mask as uint8 with values in {0, 1}; any non zero pixel becomes 1."""
    return (_imread_gray_any(path) > 0).astype(np.uint8)
