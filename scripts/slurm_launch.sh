#!/usr/bin/env bash
# -*- coding: utf-8 -*-
# SPDX-License-Identifier: Apache-2.0
# =======================================================================
#  Project       : CiproSeg - LOSO U-Net Segmentation Framework
#  File          : scripts/slurm_launch.sh
#  Purpose       : Run the script on a cluster
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

set -euo pipefail

CFG_PATH="${1:-configs/experiment.yaml}"
NPROC_PER_NODE="${NPROC_PER_NODE:-1}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"

echo "[INFO] Launching torchrun with ${NPROC_PER_NODE} processes on this node."

if [[ ! -f "${CFG_PATH}" ]]; then
  echo "[ERROR] Config not found: ${CFG_PATH}" >&2
  exit 2
fi

torchrun --standalone --nproc_per_node="${NPROC_PER_NODE}" \
  -m ciproseg.cli.train_cli --config "${CFG_PATH}"
  