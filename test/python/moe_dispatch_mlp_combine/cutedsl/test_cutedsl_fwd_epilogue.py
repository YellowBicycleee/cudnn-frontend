# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CuTe DSL BF16 forward epilogue against the PyTorch reference, bit for bit.

``mok::fwd_epilogue`` is the reference: y_shared + sum_k w_k * combine_k, FP32 accumulation in route
order and one BF16 rounding.  Geometries cover hidden sizes whose last 1024-column tile holds 256, 768 or
1024 columns, top-k 1, 2, 4, 6, 7, 8, 9, 10, 12, 13, 16, 17, 18, 24, 26, 27 and 40 (both sides of every resident-CTA
band edge on GB300, six down to one CTA per SM, and the two load rounds of top-k + 1 > 32), tile counts that are not
a multiple of the persistent grid, CTAs with a single tile, and CTAs that reuse both slots many times.  Random normal data: subnormal results are not
covered (the CUDA build flushes them to zero, the CuTe DSL kernel does not).

One GPU, no distributed setup: ``python -m pytest moe_dispatch_mlp_combine/cutedsl/test_cutedsl_fwd_epilogue.py``.
"""

import unittest

import torch
from cudnn.moe_dispatch_mlp_combine.cutedsl._arch import SUPPORTED_CAPABILITIES


def _blackwell() -> bool:
    return torch.cuda.is_available() and torch.cuda.get_device_capability() in SUPPORTED_CAPABILITIES


def _inputs(tokens, hidden, topk, seed, missing_route_fraction=0.0):
    """Inputs of one epilogue call.

    ``missing_route_fraction`` marks that share of the route slots unused (-1) and poisons their
    combine rows with NaN: the kernel must skip them, and weighting by zero would not -- the
    symmetric buffer those rows come from is allocated, not zeroed.
    """
    generator = torch.Generator(device="cuda").manual_seed(seed)
    y_shared = torch.randn(tokens, hidden, device="cuda", generator=generator).to(torch.bfloat16)
    combine = torch.randn(tokens * topk, hidden, device="cuda", generator=generator).to(torch.bfloat16)
    weights = torch.rand(tokens, topk, device="cuda", generator=generator)
    routes = torch.zeros(tokens, topk, device="cuda", dtype=torch.int32)
    if missing_route_fraction:
        unused = torch.rand(routes.shape, generator=generator, device="cuda") < missing_route_fraction
        routes = routes.masked_fill(unused, -1)
        combine.view(tokens, topk, hidden)[unused] = float("nan")
    return y_shared, combine, weights, routes


@unittest.skipUnless(_blackwell(), "the CuTe DSL forward epilogue is SM100/SM103 only")
class FwdEpilogueMatchesReference(unittest.TestCase):

    def assert_close(self, expected, actual, what=""):
        """Within the suite's BF16 tolerance of the reference: max abs error and relative L1.

        The kernel and the PyTorch model both accumulate the top-k terms in FP32 and round once,
        but not in the same order, and FP32 addition is not associative.  A per-element relative
        bound is the wrong model here: the terms are signed and cancel, so an output can sit five
        orders of magnitude below the terms that produced it and carry a large relative error from
        an absolute one that is tiny.  BF16_TOLERANCE bounds exactly that pair.  (utils.py's
        get_error_stats all-reduces its statistics; this test is single-GPU, so the same two
        statistics are computed locally.)

        Byte equality is asserted where it does hold, between repeated calls of the kernel
        (assert_bitwise) -- the determinism MCore's replay test needs.
        """
        from .utils import BF16_TOLERANCE

        max_abs, max_relative_l1 = BF16_TOLERANCE
        difference = (expected.float() - actual.float()).abs()
        abs_error_max = difference.max().item()
        reference_sum = expected.float().abs().sum()
        relative_l1 = 0.0 if reference_sum == 0 else (difference.sum() / reference_sum).item()
        self.assertLessEqual(abs_error_max, max_abs, f"{what}max abs error")
        self.assertLessEqual(relative_l1, max_relative_l1, f"{what}relative L1 error")

    def assert_bitwise(self, expected, actual, what=""):
        differing = (expected.view(torch.int16) != actual.view(torch.int16)).nonzero()
        if differing.numel():
            t, c = (int(v) for v in differing[0])
            self.fail(f"{what}{differing.shape[0]} elements differ; first at ({t}, {c}): " f"{float(expected[t, c])} vs {float(actual[t, c])}")

    def test_random(self):
        from cudnn.moe_dispatch_mlp_combine.cutedsl.fwd_epilogue import fwd_epilogue_bf16
        from .utils import run_fwd_epilogue_reference

        # Tiles per CTA on a 152-SM GB300 (grid = min(tiles, SMs x CTAs per SM); top-1 to top-7 run 6 CTAs per SM,
        # top-8 5, top-10 4, top-16 3, top-18 and top-24 2, top-40 1): (512, 256, *) 512 tiles, one per CTA;
        # (512, 1280, 10) 1024 tiles, 1-2 per CTA; (2560, 4096, 10) 10240 tiles, 16-17 per CTA, both slots reused
        # many times; (1024, 7168, 6) 7168 tiles, 7-8 per CTA at 6 CTAs per SM; the 768 / 1280 / 2560-token cases
        # have tile counts that are not a multiple of the grid.  (512, 1792, 16): three CTAs per SM, 2-3 tiles per
        # CTA; (512, 4096, 18) and (768, 3072, 24): two CTAs per SM, 6-8 tiles per CTA; (512, 1280, 40): one CTA per
        # SM, two load rounds per tile (41 vectors), 6-7 tiles per CTA.
        for tokens, hidden, topk in [
            (512, 4096, 10),
            (1024, 7168, 6),
            (768, 7168, 4),
            (512, 2304, 1),
            (1280, 1280, 8),
            (512, 1792, 16),
            (768, 3328, 2),
            (512, 256, 10),
            (512, 256, 1),
            (512, 1280, 10),
            (2560, 4096, 10),
            (512, 4096, 18),
            (768, 3072, 24),
            (512, 1280, 40),
            # both sides of every resident-CTA band edge on GB300 (6/5 at top-k 7/8, 5/4 at
            # 9/10, 4/3 at 12/13, 3/2 at 17/18, 2/1 at 26/27): 2048 tiles, 2-14 per CTA
            (512, 4096, 7),
            (512, 4096, 8),
            (512, 4096, 9),
            (512, 4096, 12),
            (512, 4096, 13),
            (512, 4096, 17),
            (512, 4096, 26),
            (512, 4096, 27),
        ]:
            with self.subTest(tokens=tokens, hidden=hidden, topk=topk):
                y_shared, combine, weights, routes = _inputs(tokens, hidden, topk, tokens + hidden + topk)
                expected = run_fwd_epilogue_reference(y_shared, combine, weights, routes)
                actual = fwd_epilogue_bf16(y_shared, combine, weights, routes)
                self.assert_close(expected, actual)
                del expected, actual, y_shared, combine, weights, routes

    def test_unused_route_slots_are_skipped(self):
        """A slot marked -1 contributes nothing, even when its combine row is NaN.

        This is the property that makes weighting by zero insufficient: the dispatch never writes
        the combine rows of a token's unused slots, and the symmetric buffer they live in is
        allocated rather than zeroed, so whatever is there may not be finite.
        """
        from cudnn.moe_dispatch_mlp_combine.cutedsl.fwd_epilogue import fwd_epilogue_bf16
        from .utils import run_fwd_epilogue_reference

        for tokens, hidden, topk in [(512, 1024, 8), (512, 4096, 10), (512, 256, 1)]:
            with self.subTest(tokens=tokens, hidden=hidden, topk=topk):
                y_shared, combine, weights, routes = _inputs(tokens, hidden, topk, tokens + hidden + topk, missing_route_fraction=0.4)
                self.assertTrue((routes < 0).any(), "the case has no unused slot")
                self.assertTrue(torch.isnan(combine.float()).any(), "no poisoned row to skip")
                actual = fwd_epilogue_bf16(y_shared, combine, weights, routes)
                self.assertTrue(torch.isfinite(actual.float()).all(), "a poisoned row of an unused slot reached the output")
                self.assert_close(run_fwd_epilogue_reference(y_shared, combine, weights, routes), actual)

    def test_token_with_no_routes_returns_the_shared_term(self):
        """Every slot of a token may be unused; its output is the shared term alone."""
        from cudnn.moe_dispatch_mlp_combine.cutedsl.fwd_epilogue import fwd_epilogue_bf16

        tokens, hidden, topk = 512, 1024, 6
        y_shared, combine, weights, _ = _inputs(tokens, hidden, topk, seed=7)
        combine.fill_(float("nan"))
        routes = torch.full((tokens, topk), -1, device="cuda", dtype=torch.int32)
        self.assert_bitwise(y_shared, fwd_epilogue_bf16(y_shared, combine, weights, routes))

    def test_repeated_calls_same_shape(self):
        """Two calls of one shape with different data: the second result must not carry anything from the first
        (stale ring slots, weights or a reused output buffer)."""
        from cudnn.moe_dispatch_mlp_combine.cutedsl.fwd_epilogue import fwd_epilogue_bf16
        from .utils import run_fwd_epilogue_reference

        tokens, hidden, topk = 1024, 4096, 6
        for seed in (11, 12):
            y_shared, combine, weights, routes = _inputs(tokens, hidden, topk, seed)
            expected = run_fwd_epilogue_reference(y_shared, combine, weights, routes)
            actual = fwd_epilogue_bf16(y_shared, combine, weights, routes)
            self.assert_close(expected, actual, what=f"seed {seed}: ")
            # The same call twice must give the same bytes: nothing carried over from the first.
            self.assert_bitwise(actual, fwd_epilogue_bf16(y_shared, combine, weights, routes), what=f"seed {seed} replay: ")
            del expected, actual, y_shared, combine, weights, routes

    def test_routed_only_is_the_zero_shared_sum(self):
        """No shared output (a routed-only layer, C++ ``fwd_epilogue_routed_kernel``): the
        accumulator starts at zero, which is the sum with an all-zero shared term bit for bit --
        0 + x is x in FP32 and the route order is the same.  A token without routes gives exact
        zeros, and an unused slot's NaN row is still skipped."""
        from cudnn.moe_dispatch_mlp_combine.cutedsl.fwd_epilogue import fwd_epilogue_bf16
        from .utils import run_fwd_epilogue_reference

        for tokens, hidden, topk, missing in [(512, 4096, 10, 0.0), (768, 7168, 4, 0.3), (512, 256, 1, 0.0), (512, 1280, 40, 0.2)]:
            with self.subTest(tokens=tokens, hidden=hidden, topk=topk, missing=missing):
                y_shared, combine, weights, routes = _inputs(tokens, hidden, topk, tokens + hidden + topk, missing_route_fraction=missing)
                zero = torch.zeros_like(y_shared)
                routed_only = fwd_epilogue_bf16(None, combine, weights, routes)
                self.assert_bitwise(fwd_epilogue_bf16(zero, combine, weights, routes), routed_only)
                self.assert_close(run_fwd_epilogue_reference(zero, combine, weights, routes), routed_only)
        tokens, hidden, topk = 512, 1024, 6
        _, combine, weights, _ = _inputs(tokens, hidden, topk, seed=8)
        combine.fill_(float("nan"))
        routes = torch.full((tokens, topk), -1, device="cuda", dtype=torch.int32)
        self.assert_bitwise(torch.zeros(tokens, hidden, dtype=torch.bfloat16, device="cuda"), fwd_epilogue_bf16(None, combine, weights, routes))
        gate = torch.ones(tokens, 1, dtype=torch.bfloat16, device="cuda")
        with self.assertRaisesRegex(ValueError, "requires a shared output"):
            fwd_epilogue_bf16(None, combine, weights, routes, gate)

    def test_routed_only_refill_of_a_token_without_routes(self):
        """Routed-only, a ring slot refilled with a tile that loads nothing (its token has no used route slot):
        the slot's barrier then expects zero bytes, which must not complete its next phase while other warps
        still wait for the current one (the kernel would not return).  (2560, 4096, 4): 11-12 tiles per CTA on
        GB300, every other token without routes, so half of the refills are such tiles; repeated, because the
        ordering that hangs is rare."""
        from cudnn.moe_dispatch_mlp_combine.cutedsl.fwd_epilogue import fwd_epilogue_bf16

        tokens, hidden, topk = 2560, 4096, 4
        y_shared, combine, weights, routes = _inputs(tokens, hidden, topk, seed=9)
        routes[::2] = -1
        combine.view(tokens, topk, hidden)[::2] = float("nan")
        expected = fwd_epilogue_bf16(torch.zeros_like(y_shared), combine, weights, routes)
        for _ in range(20):
            routed_only = fwd_epilogue_bf16(None, combine, weights, routes)
            torch.cuda.synchronize()
        self.assert_bitwise(expected, routed_only)
        self.assertFalse(routed_only[::2].float().abs().sum().item(), "a token without routes is not zero")

    def test_one_executor_serves_every_shape(self):
        """T, H, top-k, tile and slot counts are runtime parameters: alternating shapes compile one executor."""
        from cudnn.moe_dispatch_mlp_combine.cutedsl import fwd_epilogue as module

        module.clear_fwd_epilogue_executor_cache()
        for tokens, hidden, topk in [(512, 1024, 2), (768, 2048, 6), (512, 1024, 2), (1024, 4096, 10), (512, 1024, 24)]:
            y_shared, combine, weights, routes = _inputs(tokens, hidden, topk, tokens + topk)
            module.fwd_epilogue_bf16(y_shared, combine, weights, routes)
        torch.cuda.synchronize()
        self.assertEqual(len(module._EXECUTOR_CACHE), 1)


if __name__ == "__main__":
    unittest.main()
