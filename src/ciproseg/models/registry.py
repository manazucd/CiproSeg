#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# SPDX-License-Identifier: Apache-2.0
# =======================================================================
#  Project       : CiproSeg - LOSO U-Net Segmentation Framework
#  File          : src/ciproseg/models/registry.py
#  Purpose       : Model registry
#
#  Copyright (c) 2026 Manaz Kaleel
#  Authors       : Manaz Kaleel <manaz.ucd@gmail.com>
#
#  Created on    : 2025-11-01
#  Last modified : 2026-02-26, MK
#  Version       : 0.1.0
#  License       : Apache-2.0
#  Repository    : https://github.com/manazucd/CiproSeg
# =======================================================================
from __future__ import annotations

from typing import Any, Callable, Dict

_MODEL_REGISTRY: Dict[str, Callable[..., Any]] = {}


def register_model(name: str) -> Callable:
    """Decorator that registers a model builder function under the given name."""
    def decorator(fn: Callable[..., Any]) -> Callable[..., Any]:
        if name in _MODEL_REGISTRY:
            raise KeyError(f"Model '{name}' is already registered.")
        _MODEL_REGISTRY[name] = fn
        return fn
    return decorator


def create_model(name: str, **kwargs) -> Any:
    if name not in _MODEL_REGISTRY:
        raise KeyError(f"Model '{name}' not registered. Available: {list(_MODEL_REGISTRY.keys())}")
    return _MODEL_REGISTRY[name](**kwargs)


def build_model_from_cfg(cfg: Dict[str, Any]) -> Any:
    model_cfg = cfg.get("model") or {}
    name = model_cfg.get("name")
    if name is None:
        raise ValueError("Model name missing in config under 'model.name'.")

    params = model_cfg.get("params")
    if params is None:
        params = {k: v for k, v in model_cfg.items() if k != "name"}

    return create_model(name, **params)
