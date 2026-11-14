# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CuTe DSL BF16 backward epilogue: the PyTorch reference, and the unused route slots.

The backward sums the routed gradients with no weight, so a slot a token does not use cannot be
cancelled by a zero -- it has to be skipped.  Its row of ``d_x_routed_buffer`` was never written
and the symmetric buffer is allocated rather than zeroed, so the rows are poisoned with NaN here:
an implementation that reads them fails.

One GPU, no distributed setup: ``python -m pytest moe_dispatch_mlp_combine/cutedsl/test_cutedsl_bwd_epilogue.py``.
"""

from __future__ import annotations

import unittest

import torch

from cudnn.moe_dispatch_mlp_combine.cutedsl._arch import TARGET_ARCH, TARGET_CAPABILITY


def _supported() -> bool:
    return torch.cuda.is_available() and torch.cuda.get_device_capability() == TARGET_CAPABILITY


def _inputs(tokens, hidden, topk, seed, missing_route_fraction=0.0):
    generator = torch.Generator(device="cuda").manual_seed(seed)
    d_x_shared = torch.randn(tokens, hidden, device="cuda", generator=generator).to(torch.bfloat16)
    routed = torch.randn(tokens * topk, hidden, device="cuda", generator=generator).to(torch.bfloat16)
    routes = torch.zeros(tokens, topk, device="cuda", dtype=torch.int32)
    if missing_route_fraction:
        unused = torch.rand(routes.shape, generator=generator, device="cuda") < missing_route_fraction
        routes = routes.masked_fill(unused, -1)
        routed.view(tokens, topk, hidden)[unused] = float("nan")
    return d_x_shared, routed, routes


@unittest.skipUnless(_supported(), f"the CuTe DSL backward epilogue is compiled for {TARGET_ARCH}")
class BwdEpilogueMatchesReference(unittest.TestCase):

    def assert_close(self, expected, actual, what=""):
        from .utils import BF16_TOLERANCE

        max_abs, max_relative_l1 = BF16_TOLERANCE
        difference = (expected.float() - actual.float()).abs()
        reference_sum = expected.float().abs().sum()
        relative_l1 = 0.0 if reference_sum == 0 else (difference.sum() / reference_sum).item()
        self.assertLessEqual(difference.max().item(), max_abs, f"{what}max abs error")
        self.assertLessEqual(relative_l1, max_relative_l1, f"{what}relative L1 error")

    def test_random(self):
        from cudnn.moe_dispatch_mlp_combine.cutedsl.bwd_epilogue import bwd_epilogue_bf16
        from .utils import run_bwd_epilogue_reference

        for tokens, hidden, topk in [(512, 1024, 8), (512, 4096, 10), (1024, 7168, 6), (512, 256, 1)]:
            with self.subTest(tokens=tokens, hidden=hidden, topk=topk):
                d_x_shared, routed, routes = _inputs(tokens, hidden, topk, tokens + hidden + topk)
                self.assert_close(run_bwd_epilogue_reference(d_x_shared, routed, routes), bwd_epilogue_bf16(d_x_shared, routed, routes))

    def test_unused_route_slots_are_skipped(self):
        from cudnn.moe_dispatch_mlp_combine.cutedsl.bwd_epilogue import bwd_epilogue_bf16
        from .utils import run_bwd_epilogue_reference

        for tokens, hidden, topk in [(512, 1024, 8), (512, 4096, 10)]:
            with self.subTest(tokens=tokens, hidden=hidden, topk=topk):
                d_x_shared, routed, routes = _inputs(tokens, hidden, topk, tokens + hidden, missing_route_fraction=0.4)
                self.assertTrue((routes < 0).any(), "the case has no unused slot")
                actual = bwd_epilogue_bf16(d_x_shared, routed, routes)
                self.assertTrue(torch.isfinite(actual.float()).all(), "a poisoned row of an unused slot reached the output")
                self.assert_close(run_bwd_epilogue_reference(d_x_shared, routed, routes), actual)

    def test_token_with_no_routes_returns_the_shared_term(self):
        from cudnn.moe_dispatch_mlp_combine.cutedsl.bwd_epilogue import bwd_epilogue_bf16

        tokens, hidden, topk = 512, 1024, 6
        d_x_shared, routed, _ = _inputs(tokens, hidden, topk, seed=7)
        routed.fill_(float("nan"))
        routes = torch.full((tokens, topk), -1, device="cuda", dtype=torch.int32)
        actual = bwd_epilogue_bf16(d_x_shared, routed, routes)
        self.assertTrue(torch.equal(d_x_shared.view(torch.int16), actual.view(torch.int16)))

    def test_routed_only_is_the_zero_shared_sum(self):
        """No shared gradient (a routed-only layer, C++ ``bwd_epilogue_kernel<false>``): the
        accumulator starts at zero, the sum with an all-zero shared term bit for bit; a token
        without routes gives exact zeros, and unused slots' NaN rows are skipped."""
        from cudnn.moe_dispatch_mlp_combine.cutedsl.bwd_epilogue import bwd_epilogue_bf16
        from .utils import run_bwd_epilogue_reference

        for tokens, hidden, topk, missing in [(512, 1024, 8, 0.0), (512, 4096, 10, 0.4), (1024, 7168, 6, 0.0), (512, 256, 1, 0.0)]:
            with self.subTest(tokens=tokens, hidden=hidden, topk=topk, missing=missing):
                d_x_shared, routed, routes = _inputs(tokens, hidden, topk, tokens + hidden + topk, missing_route_fraction=missing)
                zero = torch.zeros_like(d_x_shared)
                routed_only = bwd_epilogue_bf16(None, routed, routes)
                self.assertTrue(torch.equal(bwd_epilogue_bf16(zero, routed, routes).view(torch.int16), routed_only.view(torch.int16)))
                self.assert_close(run_bwd_epilogue_reference(zero, routed, routes), routed_only)
        tokens, hidden, topk = 512, 1024, 6
        _, routed, _ = _inputs(tokens, hidden, topk, seed=9)
        routed.fill_(float("nan"))
        routes = torch.full((tokens, topk), -1, device="cuda", dtype=torch.int32)
        actual = bwd_epilogue_bf16(None, routed, routes)
        self.assertTrue(torch.equal(actual.view(torch.int16), torch.zeros_like(actual).view(torch.int16)))


if __name__ == "__main__":
    unittest.main()
