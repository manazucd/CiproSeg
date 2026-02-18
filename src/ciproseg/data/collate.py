#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SPDX-License-Identifier: Apache-2.0
# =======================================================================
#  Project       : CiproSeg - LOSO U-Net Segmentation Framework
#  File          : src/ciproseg/data/collate.py
#  Purpose       : Collate functions for segmentation dataloaders
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

from typing import Any, Dict, List, Tuple, Union

import torch

# Dataset items may be a (x, y) tuple or a {"x", "y", "meta"?} dict.
BatchItem = Union[Dict[str, Any], Tuple[torch.Tensor, torch.Tensor]]


def default_collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
    """Minimal collate for dict style datasets with 'x' and 'y' keys only."""
    if not batch:
        raise ValueError("Empty batch")
    return {
        "x": torch.stack([b["x"] for b in batch]),
        "y": torch.stack([b["y"] for b in batch]),
    }


def segmentation_collate_fn(batch: List[BatchItem]) -> Dict[str, Any]:
    """
    Collate for segmentation tasks, handles both (x, y) tuples and dicts.
    Meta values, when present, are collected as a list rather than stacked.
    """
    if not batch:
        raise ValueError("Empty batch")

    first = batch[0]

    if isinstance(first, tuple):
        xs = torch.stack([x for x, _ in batch])  
        ys = torch.stack([y for _, y in batch])  
        return {"x": xs, "y": ys}

    if isinstance(first, dict):
        out: Dict[str, Any] = {
            "x": torch.stack([b["x"] for b in batch]),  
            "y": torch.stack([b["y"] for b in batch]),  
        }
        if "meta" in first:
            out["meta"] = [b.get("meta", {}) for b in batch]
        return out

    raise TypeError(f"Unsupported batch item type: {type(first)}")
