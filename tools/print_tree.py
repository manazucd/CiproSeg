#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SPDX-License-Identifier: Apache-2.0
# =======================================================================
#  Project       : CiproSeg - LOSO U-Net Segmentation Framework
#  File          : tools/print_tree.py
#  Purpose       : Utility to print repository tree
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

from pathlib import Path


def print_tree(root: Path, prefix: str = "") -> None:
    entries = sorted(
        (p for p in root.iterdir() if not p.name.startswith(".")),
        key=lambda p: (p.is_file(), p.name),
    )
    for i, p in enumerate(entries):
        is_last = i == len(entries) - 1
        print(prefix + ("└─ " if is_last else "├─ ") + p.name)
        if p.is_dir():
            print_tree(p, prefix + ("   " if is_last else "│  "))


if __name__ == "__main__":
    print_tree(Path("."))