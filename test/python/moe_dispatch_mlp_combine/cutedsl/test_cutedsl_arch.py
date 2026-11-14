# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU: the Blackwell target of the CuTe DSL kernels (``cudnn/moe_dispatch_mlp_combine/cutedsl/_arch.py``): the device's compute capability
picks sm_100a / sm_103a, a ``CUTE_DSL_ARCH`` naming another part is declined, and the target reaches CuTe DSL through
the compile options, not through the environment."""

import unittest
from unittest import mock

from cudnn.moe_dispatch_mlp_combine.cutedsl import _arch


class ArchResolution(unittest.TestCase):
    def test_device_picks_the_part(self):
        self.assertEqual(_arch.resolve(None, (10, 0)), "sm_100a")
        self.assertEqual(_arch.resolve(None, (10, 3)), "sm_103a")

    def test_no_or_unsupported_device_falls_back_to_the_default(self):
        self.assertEqual(_arch.resolve(None, None), _arch.DEFAULT_ARCH)
        self.assertEqual(_arch.resolve(None, (9, 0)), _arch.DEFAULT_ARCH)
        self.assertEqual(_arch.resolve(None, (12, 0)), _arch.DEFAULT_ARCH)
        self.assertEqual(_arch.resolve("sm_100a", None), _arch.DEFAULT_ARCH)  # nothing to launch on: not used

    def test_matching_request_is_accepted(self):
        self.assertEqual(_arch.resolve("sm_100a", (10, 0)), "sm_100a")
        self.assertEqual(_arch.resolve("sm_103a", (10, 3)), "sm_103a")

    def test_request_for_another_part_is_declined_with_the_reason(self):
        for requested, capability in (
            ("sm_100a", (10, 3)),
            ("sm_103a", (10, 0)),
            ("sm_103", (10, 3)),
            ("sm_100f", (10, 3)),
            ("sm_90a", (10, 3)),
            ("", (10, 0)),
        ):
            with self.assertRaises(RuntimeError) as ctx:
                _arch.resolve(requested, capability)
            message = str(ctx.exception)
            self.assertIn(f"{_arch.ARCH_ENV}={requested}", message)
            self.assertIn(_arch.SUPPORTED_CAPABILITIES[capability], message)

    def test_target_and_tvm_ffi_are_compile_options(self):
        self.assertIn(_arch.TARGET_ARCH, _arch.CAPABILITY_BY_ARCH)
        self.assertEqual(_arch.SUPPORTED_CAPABILITIES[_arch.TARGET_CAPABILITY], _arch.TARGET_ARCH)
        self.assertEqual(set(_arch.SUPPORTED_CAPABILITIES.values()), {"sm_100a", "sm_103a"})
        options = _arch.COMPILE_OPTIONS.split()
        self.assertEqual(options[options.index("--gpu-arch") + 1], _arch.TARGET_ARCH)
        self.assertIn("--enable-tvm-ffi", options)
        self.assertEqual(options[options.index("--opt-level") + 1], "2")


class DeviceCheck(unittest.TestCase):
    def test_check_device_rejects_another_part(self):
        with mock.patch.object(_arch.torch.cuda, "get_device_capability", return_value=(9, 0)):
            with self.assertRaises(NotImplementedError) as ctx:
                _arch.check_device(0, "the test kernel")
        message = str(ctx.exception)
        self.assertIn(_arch.TARGET_ARCH, message)
        self.assertIn("(9, 0)", message)
        with mock.patch.object(_arch.torch.cuda, "get_device_capability", return_value=_arch.TARGET_CAPABILITY):
            _arch.check_device(0, "the test kernel")  # no raise


if __name__ == "__main__":
    unittest.main()
