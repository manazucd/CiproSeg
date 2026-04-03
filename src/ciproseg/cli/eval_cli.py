#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SPDX-License-Identifier: Apache-2.0
# =======================================================================
#  Project       : CiproSeg - LOSO U-Net Segmentation Framework
#  File          : src/ciproseg/cli/eval_cli.py
#  Purpose       : Evaluation CLI
#
#  Copyright (c) 2026 Manaz Kaleel
#  Authors       : Manaz Kaleel <manaz.ucd@gmail.com>
#
#  Created on    : 2025-11-01
#  Last modified : 2026-04-03, MK
#  Version       : 0.1.0
#  License       : Apache-2.0
#  Repository    : https://github.com/manazucd/CiproSeg
# =======================================================================
from __future__ import annotations

import argparse
import json
import logging
import sys


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="ciproseg-eval",
        description=(
            "Evaluate a trained CiproSeg checkpoint on its LOSO held-out test series.\n\n"
            "Example:\n"
            "  ciproseg-eval --ckpt runs/unet2/2026-04-03_120254/fold_01_test_controlnonsilica/best.pt"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--ckpt",
        type=str,
        required=True,
        metavar="PATH",
        help="Path to a CiproSeg .pt checkpoint (e.g. runs/.../fold_01_.../best.pt).",
    )
    p.add_argument(
        "--dataset-root",
        type=str,
        default=None,
        metavar="DIR",
        help="Override the dataset root stored in the checkpoint (if data has moved).",
    )
    p.add_argument(
        "--patch", type=int, default=None,
        help="Override patch size for tiled inference.",
    )
    p.add_argument(
        "--stride", type=int, default=None,
        help="Override stride for tiled inference.",
    )
    p.add_argument(
        "--thr", type=float, default=0.5,
        help="Probability threshold for binarisation (default: 0.5).",
    )
    p.add_argument(
        "--cpu", action="store_true",
        help="Force CPU evaluation.",
    )
    p.add_argument(
        "--json", type=str, default=None, metavar="PATH",
        help="Optional path to save metrics as JSON.",
    )
    return p


def main() -> int:
    args = build_argparser().parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="[%(levelname)s] %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    logging.info(f"[CiproSeg] Evaluating checkpoint: {args.ckpt}")

    from ..core.evaluator import evaluate_checkpoint

    device = "cpu" if args.cpu else None

    try:
        metrics = evaluate_checkpoint(
            args.ckpt,
            dataset_root=args.dataset_root,
            patch=args.patch,
            stride=args.stride,
            thr=args.thr,
            device=device,
        )
    except (FileNotFoundError, KeyError, ValueError, RuntimeError) as exc:
        logging.error(f"[CiproSeg] {exc}")
        return 1

    print(f"\n{'─' * 60}")
    print(f"  Checkpoint : {metrics.get('ckpt_path', args.ckpt)}")
    print(f"  Fold       : {metrics.get('fold_name', '?')}")
    print(f"  Test series: {metrics.get('test_series', '?')}")
    print(f"{'─' * 60}")
    print(f"  Dice       : {metrics['dice']:.4f}")
    print(f"  IoU        : {metrics['iou']:.4f}")
    print(f"  Precision  : {metrics['precision']:.4f}")
    print(f"  Recall     : {metrics['recall']:.4f}")
    if "loss" in metrics:
        print(f"  Loss (BCE) : {metrics['loss']:.5f}")
    print(f"{'─' * 60}\n")

    if args.json:
        try:
            with open(args.json, "w", encoding="utf-8") as f:
                json.dump(metrics, f, indent=2, default=str)
            logging.info(f"[CiproSeg] Wrote metrics JSON: {args.json}")
        except Exception as e:
            logging.warning(f"[CiproSeg] Failed to write JSON ({args.json}): {e}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
