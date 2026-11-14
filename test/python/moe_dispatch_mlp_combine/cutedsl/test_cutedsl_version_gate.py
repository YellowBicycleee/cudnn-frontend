# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU: the operator's CuTe DSL gate (``cudnn/moe_dispatch_mlp_combine/cutedsl/_version_gate.py``) accepts a CuTe DSL
at or above its floor (4.8.0) in the cu13 build and declines anything else with the installed and the required
version, through cuDNN frontend's gate (``cudnn.frost.buffers``, ``python/cudnn/AGENTS.md`` Rule 7)."""

from __future__ import annotations

import sys
import types
import unittest
from unittest import mock

from cudnn.frost import buffers

from ._cutedsl_by_path import load_cutedsl_module

gate = load_cutedsl_module("_version_gate")


class Floor(unittest.TestCase):
    def test_floor_is_at_least_the_library_floor(self):
        self.assertEqual(gate.CUTEDSL_FLOOR, (4, 8, 0))
        self.assertGreaterEqual(gate.CUTEDSL_FLOOR, tuple(buffers.CUTEDSL_MIN_VERSION))

    def test_cu13_or_unknown_flavor_passes(self):
        self.assertIsNone(gate.flavor_error(13))
        self.assertIsNone(gate.flavor_error(None))  # flavor not known

    def test_cu12_build_is_declined_with_the_reason(self):
        message = gate.flavor_error(12)
        self.assertIn("cu12", message)
        self.assertIn("nvidia-cutlass-dsl[cu13]>=4.8.0", message)


class RequireCutedsl(unittest.TestCase):
    def _require(self, dsl_version, cuda_major, distribution="nvidia-cutlass-dsl"):
        version_info = types.ModuleType("cutlass.base_dsl.version_info")
        version_info.CUDA_VERSION = types.SimpleNamespace(major=cuda_major, minor=0)
        modules = {
            "cutlass": types.ModuleType("cutlass"),
            "cutlass.base_dsl": types.ModuleType("cutlass.base_dsl"),
            "cutlass.base_dsl.version_info": version_info,
        }
        state = (dsl_version is not None, None if dsl_version is None else (distribution, dsl_version))
        with (
            mock.patch.object(gate, "_VERSION", None),
            mock.patch.object(buffers, "_DSL_STATE", state),
            mock.patch.dict(sys.modules, modules),
        ):
            return gate.require_cutedsl()

    def test_supported_install_returns_the_version(self):
        for dsl in ("4.8.0", "4.8.1", "4.9.0", "4.10.0", "5.0.0", "4.8.0.dev1", "4.8.0+local"):
            self.assertEqual(self._require(dsl, 13), dsl)

    def test_older_release_names_both_versions(self):
        for dsl in ("4.5.2", "4.7.0", "4.7.9", "3.0.0"):
            with self.assertRaises(gate.CuteDslRequirementError) as raised:
                self._require(dsl, 13)
            message = str(raised.exception)
            self.assertIn("moe_dispatch_mlp_combine requires nvidia-cutlass-dsl >= 4.8.0", message)
            self.assertIn(f"found {dsl}", message)

    def test_missing_dsl_is_declined(self):
        with self.assertRaisesRegex(gate.CuteDslRequirementError, "not installed"):
            self._require(None, 13)

    def test_cu12_build_raises_before_any_jit(self):
        with self.assertRaisesRegex(gate.CuteDslRequirementError, "cu12"):
            self._require("4.8.0", 12)


if __name__ == "__main__":
    unittest.main()
