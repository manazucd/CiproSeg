#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SPDX-License-Identifier: Apache-2.0
# =======================================================================
#  Project       : CiproSeg - LOSO U-Net Segmentation Framework
#  File          : tools/check_dataset_pairs.py
#  Purpose       : Verify image/mask filename parity across all series
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
import sys
from pathlib import Path

EXTS = (".tif", ".tiff", ".png", ".jpg", ".jpeg")


def list_files(p: Path) -> list[str]:
    return sorted(f.name for f in p.iterdir() if f.is_file() and f.name.lower().endswith(EXTS))


def main(root: str) -> None:
    root = Path(root)
    img_root = Path(root) / "images"
    msk_root = Path(root) / "masks"

    series = sorted(d.name for d in img_root.iterdir() if d.is_dir())
    bad = 0

    for s in series:
        imgs = set(list_files(img_root / s))
        msks = set(list_files(msk_root / s))
        if imgs == msks:
            continue

        bad += 1
        # Cap output at 20 filenames per side to keep the report readable.
        print(f"\n=== Series: {s} ===")
        print("Only in images:", sorted(imgs - msks)[:20])
        print("Only in masks :", sorted(msks - imgs)[:20])

    if bad == 0:
        print("All series have matching filenames.")
    else:
        print(f"\n{bad} series have mismatches.")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("Usage: python tools/check_dataset_pairs.py <dataset_root>")
    main(sys.argv[1])
