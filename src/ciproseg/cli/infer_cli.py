#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SPDX-License-Identifier: Apache-2.0
# =======================================================================
#  Project       : CiproSeg - LOSO U-Net Segmentation Framework
#  File          : src/ciproseg/cli/infer_cli.py
#  Purpose       : Inference CLI (thin shell over core/infer.py)
#
#  Copyright (c) 2026 Manaz Kaleel
#  Authors       : Manaz Kaleel <manaz.ucd@gmail.com>
#
#  Created on    : 2026-02-21
#  Last modified : 2026-04-03, MK
#  Version       : 0.1.0
#  License       : Apache-2.0
#  Repository    : https://github.com/manazucd/CiproSeg
# =======================================================================
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Optional

from ..core.infer import run_inference_on_dir

def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="ciproseg-infer",
        description=(
            "CiproSeg inference: run a trained checkpoint on a directory of images.\n"
            "Outputs per-image probability maps, binary masks, and overlay visuals."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    p.add_argument(
        "--ckpt",
        type=str,
        required=True,
        metavar="PATH",
        help="Path to a CiproSeg .pt checkpoint (e.g. runs/fold_00/best.pt).",
    )
    p.add_argument(
        "--input",
        type=str,
        required=True,
        metavar="DIR",
        help="Directory containing input images (.tif/.tiff/.png/.jpg/.jpeg).",
    )

    p.add_argument(
        "--output",
        type=str,
        default=None,
        metavar="DIR",
        help=(
            "Output directory for artefacts. "
            "Defaults to '<input>_predictions' placed beside --input."
        ),
    )
    p.add_argument(
        "--masks",
        type=str,
        default=None,
        metavar="DIR",
        help=(
            "Optional directory of ground-truth masks (same filenames as images). "
            "When supplied, GT overlay and TP/FP/FN diff map are also saved."
        ),
    )
    p.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        metavar="FLOAT",
        help="Probability threshold for binarisation [0, 1]. Default: 0.5.",
    )
    p.add_argument(
        "--patch",
        type=int,
        default=256,
        metavar="INT",
        help="Patch (tile) size in pixels used during tiled inference. Default: 256.",
    )
    p.add_argument(
        "--stride",
        type=int,
        default=None,
        metavar="INT",
        help=(
            "Tile stride in pixels. Smaller strides give smoother overlap averaging "
            "at the cost of more computation. Defaults to patch // 2."
        ),
    )
    p.add_argument(
        "--cpu",
        action="store_true",
        default=False,
        help="Force CPU inference (overrides CUDA auto-detection).",
    )
    p.add_argument(
        "--device",
        type=str,
        default=None,
        metavar="STR",
        help="Explicit device string, e.g. 'cuda:1'. Ignored when --cpu is set.",
    )
    p.add_argument(
        "--verbose",
        action="store_true",
        default=False,
        help="Enable DEBUG-level logging for per-file details.",
    )
    return p


def _resolve_output_dir(input_dir: str, output_arg: Optional[str]) -> str:
    """
    If --output was given, use it as-is.
    Otherwise place <input_name>_predictions one directory above
    the parent of the input directory (safely).
    """
    if output_arg:
        return output_arg

    p = Path(input_dir).resolve()
    base = p.parents[1] if len(p.parents) > 1 else p.parent
    return str(base / f"{p.name}_predictions")


def _validate_args(args: argparse.Namespace) -> None:
    """Validate argument constraints before touching any heavy imports."""
    if not 0.0 <= args.threshold <= 1.0:
        raise ValueError(f"--threshold must be in [0, 1], got {args.threshold}")
    if args.patch < 1:
        raise ValueError(f"--patch must be >= 1, got {args.patch}")
    if args.stride is not None and args.stride < 1:
        raise ValueError(f"--stride must be >= 1, got {args.stride}")
    if args.stride is not None and args.stride > args.patch:
        raise ValueError(
            f"--stride ({args.stride}) must be <= --patch ({args.patch}). "
            "A stride larger than the patch would leave gaps in coverage."
        )


def main() -> int:
    args = build_argparser().parse_args()

    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="[%(levelname)s] %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    logger = logging.getLogger("ciproseg")
    logger.setLevel(log_level)

    try:
        _validate_args(args)
    except ValueError as exc:
        logger.error(f"[infer] Argument error: {exc}")
        return 2

    output_dir = _resolve_output_dir(args.input, args.output)

    logger.info("[infer] ============================================================")
    logger.info("[infer] CiproSeg inference")
    logger.info(f"[infer]   ckpt      : {args.ckpt}")
    logger.info(f"[infer]   input     : {args.input}")
    logger.info(f"[infer]   output    : {output_dir}")
    logger.info(f"[infer]   masks     : {args.masks or '(none)'}")
    logger.info(f"[infer]   threshold : {args.threshold}")
    logger.info(f"[infer]   patch     : {args.patch}")
    logger.info(f"[infer]   stride    : {args.stride or args.patch // 2}  {'(default)' if args.stride is None else ''}")
    logger.info(f"[infer]   device    : {'cpu (forced)' if args.cpu else args.device or 'auto'}")
    logger.info("[infer] ============================================================")

    try:
        run_inference_on_dir(
            ckpt_path=args.ckpt,
            input_dir=args.input,
            output_dir=output_dir,
            threshold=args.threshold,
            patch=args.patch,
            stride=args.stride,         
            device_str=args.device,
            force_cpu=args.cpu,
            masks_dir=args.masks,
        )
    except (FileNotFoundError, NotADirectoryError) as exc:
        logger.error(f"[infer] {exc}")
        return 1
    except KeyboardInterrupt:
        logger.warning("[infer] Interrupted.")
        return 130
    except Exception as exc:
        logger.exception(f"[infer] Unexpected error: {exc}")
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
