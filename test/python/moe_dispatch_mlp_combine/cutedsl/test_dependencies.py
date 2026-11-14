# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU: what the operator needs of its dependencies is checked on first use, not when its package is imported.

PyTorch: the workspace probes the distributed features it calls (``Work.block_current_stream`` and, for more than
one rank, symmetric memory) and declines a PyTorch without them, naming what is missing.  Triton (the optional
``nvidia-cudnn-frontend[triton]``): only the fused kernels of a gated backward import it, and without it they decline."""

from __future__ import annotations

import subprocess
import sys
import types
import unittest
from unittest import mock

import torch

from cudnn.moe_dispatch_mlp_combine import functional


class TorchFeatures(unittest.TestCase):
    def test_installed_torch_has_every_feature(self):
        self.assertIsNone(functional._torch_distributed_features(1))
        self.assertEqual(functional._torch_distributed_features(8).__name__, "torch.distributed._symmetric_memory")

    def test_missing_features_are_named(self):
        bare = types.SimpleNamespace(Work=object, _SymmetricMemory=object)
        with mock.patch.object(torch._C, "_distributed_c10d", bare):
            with self.assertRaisesRegex(RuntimeError, "Work.block_current_stream") as one_rank:
                functional._torch_distributed_features(1)
            self.assertNotIn("_symmetric_memory", str(one_rank.exception))  # one rank needs no symmetric memory
            with self.assertRaisesRegex(RuntimeError, "multicast_ptr"):
                functional._torch_distributed_features(8)
        with mock.patch.dict(sys.modules, {"torch.distributed._symmetric_memory": None}):
            with self.assertRaisesRegex(RuntimeError, r"torch\.distributed\._symmetric_memory"):
                functional._torch_distributed_features(8)


def _run(code):
    """Runs ``code`` in a fresh interpreter, where nothing is imported yet."""
    return subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=600)


_BLOCK = "import sys; sys.modules.update(dict.fromkeys(('triton', 'torch.distributed._symmetric_memory')));"


class OptionalImports(unittest.TestCase):
    def test_import_needs_neither_triton_nor_symmetric_memory(self):
        result = _run(_BLOCK + "import cudnn.moe_dispatch_mlp_combine")
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_fused_gate_kernels_without_triton_decline(self):
        result = _run(_BLOCK + "from cudnn.moe_dispatch_mlp_combine import functional; functional._gate_fused_module()")
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("RuntimeError", result.stderr)
        self.assertIn("nvidia-cudnn-frontend[triton]", result.stderr)
