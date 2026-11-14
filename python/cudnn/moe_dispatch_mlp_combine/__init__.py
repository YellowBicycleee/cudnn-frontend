# Copyright 2026 Cursor Research
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Derived from Mixture-of-Kittens (https://github.com/cursor/mixture-of-kittens, commit 22fc95ae);
# ported to CuTe DSL and modified by NVIDIA.

"""MoE dispatch, expert MLP and combine: an expert-parallel MoE layer as CuTe DSL megakernels, derived from
Mixture-of-Kittens.  ``moe_dispatch_mlp_combine_{prepare, forward, backward}`` (``.api``, imported on first access,
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
    "moe_dispatch_mlp_combine_prepare": (".api", "moe_dispatch_mlp_combine_prepare"),
    "moe_dispatch_mlp_combine_forward": (".api", "moe_dispatch_mlp_combine_forward"),
    "moe_dispatch_mlp_combine_backward": (".api", "moe_dispatch_mlp_combine_backward"),
    "MoeDispatchMlpCombineConfig": (".api", "MoeDispatchMlpCombineConfig"),
    "MoeDispatchMlpCombinePlan": (".api", "MoeDispatchMlpCombinePlan"),
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
