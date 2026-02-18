#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SPDX-License-Identifier: Apache-2.0
# =======================================================================
#  Project       : CiproSeg - LOSO U-Net Segmentation Framework
#  File          : examples/train_from_python.py
#  Purpose       : Example
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
"""
Demonstrates invoking the CiproSeg training pipeline directly from Python
rather than through the shell CLI. Useful for embedding training runs inside
notebooks, test harnesses, or higher-level orchestration scripts.
"""

import sys
from pathlib import Path

from ciproseg.cli.train_cli import main as train_main


def run() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    config_path = repo_root / "configs" / "experiment.yaml"

    old_argv = sys.argv[:]
    try:
        sys.argv = ["ciproseg-train", "--config", str(config_path)]
        train_main()
    finally:
        sys.argv = old_argv

if __name__ == "__main__":
    run()
    