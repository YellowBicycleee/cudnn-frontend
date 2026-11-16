# Copyright 2026 Cursor Research
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Derived from Mixture-of-Kittens (https://github.com/cursor/mixture-of-kittens, commit 22fc95ae);
# ported to CuTe DSL and modified by NVIDIA.

"""CuTe MoK: an expert-parallel MoE layer as CuTe DSL megakernels, derived from
Mixture-of-Kittens.  ``mok_{prepare, forward, backward}`` (``.api``, imported on first access,
exported as ``cudnn.ops.*``) are the operator functions over a prepared plan; ``.functional`` is the layer interface
they call (workspace, schedule, forward, backward) and ``.ops`` holds the operators around the megakernels.

This package imports torch, so it is reached only through the lazy table of ``cudnn.ops``, never from
``import cudnn``."""

import importlib

import torch  # noqa: F401

from . import ops as ops
from . import functional as functional

# The operator API resolves on first attribute access: importing this package does not import ``.api``.
_LAZY_EXPORTS = {
    "mok_prepare": (".api", "mok_prepare"),
    "mok_forward": (".api", "mok_forward"),
    "mok_backward": (".api", "mok_backward"),
    "CuteMoKConfig": (".api", "CuteMoKConfig"),
    "CuteMoKPlan": (".api", "CuteMoKPlan"),
}

__all__ = [
    "functional",
    "ops",
    *_LAZY_EXPORTS,
]


def __getattr__(name):
    target = _LAZY_EXPORTS.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, attribute = target
    value = getattr(importlib.import_module(module_name, __name__), attribute)
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | set(__all__))
