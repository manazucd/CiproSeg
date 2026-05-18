#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SPDX-License-Identifier: Apache-2.0
# =======================================================================
#  Project       : CiproSeg - LOSO U-Net Segmentation Framework
#  File          : src/ciproseg/cli/gui_cli.py
#  Purpose       : Launch the CiproSeg web GUI on localhost
#
#  Copyright (c) 2026 Manaz Kaleel
#  Authors       : Manaz Kaleel <manaz.ucd@gmail.com>
#
#  Created on    : 2026-04-11
#  Last modified : 2026-04-11, MK
#  Version       : 0.1.0
#  License       : Apache-2.0
#  Repository    : https://github.com/manazucd/CiproSeg
# =======================================================================
"""Launch the CiproSeg web GUI (localhost FastAPI server)."""
from __future__ import annotations

import argparse
import sys


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="ciproseg-gui",
        description="Launch the CiproSeg web GUI on localhost.",
    )
    p.add_argument("--host", type=str, default="127.0.0.1", help="Bind host (default: 127.0.0.1).")
    p.add_argument("--port", type=int, default=7860, help="Port (default: 7860).")
    p.add_argument("--no-browser", action="store_true", help="Do not auto-open the browser.")
    return p


def main() -> int:
    args = build_argparser().parse_args()

    from ..gui.server import run_server

    print(f"\n  CiproSeg GUI starting on http://{args.host}:{args.port}\n")
    run_server(host=args.host, port=args.port, open_browser=not args.no_browser)
    return 0


if __name__ == "__main__":
    sys.exit(main())
