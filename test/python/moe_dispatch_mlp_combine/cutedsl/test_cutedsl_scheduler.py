# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CuTe DSL local scheduler against the PyTorch reference, entry for entry.

``utils.run_schedule_reference`` is the reference: the same gathered top-k must give the same padded
expert counts, total, peer ranks (every capacity entry, -1 on padding rows and the tail) and, for
the routed entries, peer token indices in the thread-major order.  Geometries cover route
counts that are not a multiple of the 1024-thread stride, more routes per CUDA thread than one
register batch of the schedule kernel holds, local expert counts that are not a multiple of the
kernel's expert group and above 256, EP 1 / 4 / 8 / 16 / 64, first and last rank, skewed and empty
experts, top-k 1, a single token, the capacity exactly at the padded total, and repeated calls whose
outputs reuse the freed allocations of the previous call.

One GPU, no distributed setup: ``python -m pytest moe_dispatch_mlp_combine/cutedsl/test_cutedsl_scheduler.py``.
"""

import unittest

import torch
from cudnn.moe_dispatch_mlp_combine.cutedsl._arch import SUPPORTED_CAPABILITIES


def _blackwell() -> bool:
    return torch.cuda.is_available() and torch.cuda.get_device_capability() in SUPPORTED_CAPABILITIES


def _uniform(world, tokens, experts, topk, seed):
    generator = torch.Generator(device="cuda").manual_seed(seed)
    scores = torch.rand(world, tokens, experts, device="cuda", generator=generator)
    return scores.topk(topk, dim=-1).indices.to(torch.int32).contiguous()


def _reference(topk_all, num_local_experts, capacity, rank):
    """The PyTorch model of the schedule, which has no geometry restrictions of its own."""
    from .utils import run_schedule_reference

    return run_schedule_reference(topk_all, num_local_experts, capacity, rank)


@unittest.skipUnless(_blackwell(), "the CuTe DSL scheduler is SM100/SM103 only")
class SchedulerMatchesReference(unittest.TestCase):

    @staticmethod
    def _capacity(topk_all, num_local_experts, capacity):
        # Default: enough for every route plus a full padding tile per local expert (the scheduler traps otherwise).
        world, tokens, topk = topk_all.shape
        return world * tokens * topk + num_local_experts * 256 if capacity is None else capacity

    def _case(self, topk_all: torch.Tensor, num_local_experts: int, rank: int, capacity: int | None = None) -> None:
        from cudnn.moe_dispatch_mlp_combine.cutedsl.scheduler import scheduler

        capacity = self._capacity(topk_all, num_local_experts, capacity)
        self._compare(topk_all, num_local_experts, rank, capacity, scheduler(topk_all, num_local_experts, capacity, rank))

    def _compare(self, topk_all, num_local_experts, rank, capacity, actual) -> None:
        expected = _reference(topk_all, num_local_experts, capacity, rank)
        torch.cuda.synchronize()
        routed = expected[0] >= 0
        # The fifth output: the unpadded rows of each local expert, i.e. the routed rows (peer rank
        # >= 0) of its padded segment in the CUDA schedule.
        device = topk_all.device
        expert_of_row = torch.repeat_interleave(torch.arange(num_local_experts, device=device), expected[3].to(torch.int64))
        valid_rows = torch.zeros(num_local_experts, dtype=torch.int32, device=device).index_add_(0, expert_of_row, routed[: int(expected[2])].to(torch.int32))
        self.assertEqual(tuple(actual[0].shape), (capacity,))
        for name, want, got in (
            ("peer_rank", expected[0], actual[0]),  # every capacity entry: routed, padding (-1), tail (-1)
            ("peer_token_idx", expected[1][routed], actual[1][routed]),
            ("num_tokens", expected[2], actual[2]),
            ("tokens_per_expert", expected[3], actual[3]),
            ("valid_tokens_per_expert", valid_rows, actual[4]),
        ):
            differing = (want != got).nonzero()
            if differing.numel():
                first = int(differing[0, 0])
                self.fail(f"{name}: {differing.shape[0]}/{want.numel()} entries differ; first at {first}: " f"CUDA {int(want[first])} vs DSL {int(got[first])}")

    def test_uniform(self):
        """Uniform random routing: Qwen / DeepSeek-like expert counts and others (9 and 20 local experts), EP 1 / 4 / 8 / 16 / 64;
        (4, 65536, 256, 10) has 640 routes per CUDA thread, more than one register batch of the schedule kernel."""
        for world, tokens, experts, topk in [
            (8, 8192, 512, 10),
            (8, 3072, 384, 6),
            (8, 1024, 64, 8),
            (4, 4352, 36, 3),
            (1, 768, 20, 2),
            (8, 20480, 384, 6),
            (16, 1024, 256, 6),
            (64, 512, 128, 4),
            (4, 65536, 256, 10),
        ]:
            topk_all = _uniform(world, tokens, experts, topk, seed=tokens)
            for rank in sorted({0, world - 1}):
                with self.subTest(world=world, tokens=tokens, experts=experts, topk=topk, rank=rank):
                    self._case(topk_all, experts // world, rank)

    def test_more_than_256_local_experts(self):
        """E per rank above 256 (the mega kernels' grouped decode): EP1 with 512 local experts, EP4 with 288, EP8 with 288 (the
        multi-node shape on one GPU), and EP1 with 1024 (more local experts than the schedule kernel's block has threads, so its
        block scan over the padded counts runs a second round).  Each needs more than one pass of the schedule kernel's expert
        tables, which are refilled per group."""
        for world, tokens, experts, topk in [(1, 2048, 512, 8), (4, 1024, 1152, 6), (8, 512, 2304, 4), (1, 1024, 1024, 4)]:
            topk_all = _uniform(world, tokens, experts, topk, seed=experts)
            with self.subTest(world=world, experts=experts):
                self._case(topk_all, experts // world, world - 1)

    def test_many_rows_and_many_experts(self):
        """128 local experts (several expert groups of the schedule kernel) with 320 routes per CUDA thread (several
        register batches): the per-column state must carry across both loops."""
        topk_all = _uniform(4, 32768, 512, 10, seed=7)
        self._case(topk_all, 128, 1)

    def test_op_matches_reference(self):
        """``cudnn.moe_dispatch_mlp_combine.ops.schedule`` equals the reference entry for entry, and its outputs do not alias."""
        from cudnn.moe_dispatch_mlp_combine.ops import schedule

        world, tokens, experts, topk = 8, 1024, 256, 6
        topk_all = _uniform(world, tokens, experts, topk, seed=11)
        capacity = world * tokens * topk + (experts // world) * 256
        expected = _reference(topk_all, experts // world, capacity, 2)
        actual = schedule(topk_all, experts // world, capacity, 2)
        torch.cuda.synchronize()
        routed = expected[0] >= 0
        for name, want, got in (
            ("peer_rank", expected[0], actual[0]),
            ("peer_token_idx", expected[1][routed], actual[1][routed]),
            ("num_tokens", expected[2], actual[2]),
            ("tokens_per_expert", expected[3], actual[3]),
        ):
            self.assertTrue(torch.equal(want, got), name)
        # The two per-capacity-entry outputs must be separate buffers; the three small per-expert
        # counts may share one allocation.  (The stricter "no output aliases another" was a
        # torch.library custom-op requirement, and schedule is a plain function now.)
        peer_rank, peer_token_idx = actual[0], actual[1]
        self.assertNotEqual(
            int(peer_rank.untyped_storage().data_ptr()), int(peer_token_idx.untyped_storage().data_ptr()), "peer_rank and peer_token_idx share a buffer"
        )

    def test_skewed_and_empty_experts(self):
        """Every route of every peer on one expert of rank 0; ranks without routes; a single local expert;
        every route on one rank spread over its experts; the capacity exactly at the padded total."""
        world, tokens, topk = 8, 2304, 1
        topk_all = torch.full((world, tokens, topk), 3, dtype=torch.int32, device="cuda")
        for rank in (0, 5):
            with self.subTest(rank=rank):
                self._case(topk_all, 8, rank)
        with self.subTest(case="capacity exactly at the padded total"):
            # Rank 0 receives all 8 * 2304 routes on expert 3: 18432 rows, already a multiple of 256.
            self._case(topk_all, 8, 0, capacity=world * tokens * topk)
        pattern = torch.arange(tokens, dtype=torch.int32, device="cuda").remainder(world)
        with self.subTest(case="one local expert, round robin"):
            self._case(pattern.view(1, tokens, 1).expand(world, -1, -1).contiguous(), 1, 2)
        with self.subTest(case="all routes to rank 6, spread over its 12 experts"):
            experts = 12
            routes = (torch.arange(world * tokens * 4, dtype=torch.int32, device="cuda") * 7).remainder(experts) + 6 * experts
            topk_all = routes.view(world, tokens, 4)
            for rank in (6, 0):
                self._case(topk_all, experts, rank)

    def test_single_token_topk_one(self):
        """T = 1 and top-k 1 (below the CUDA op's host minimum; the raw binding is the reference)."""
        world, experts = 8, 8
        topk_all = torch.tensor([[[5]], [[13]], [[5]], [[7]], [[0]], [[5]], [[63]], [[6]]], dtype=torch.int32, device="cuda")
        # Rank 0 receives experts 0, 5, 6, 7 (4 padded tiles = 1024 rows), rank 7 one route on expert 63.
        for rank, capacity in ((0, 1024), (7, 256)):
            with self.subTest(rank=rank):
                self._case(topk_all, experts, rank, capacity=capacity)

    def test_repeated_calls_reuse_freed_allocations(self):
        """Two calls of the same geometry in one process with different routings: the second call's outputs come from the
        caching allocator's reuse of the first call's freed buffers, so any entry the kernels do not write (a count, a
        padding row, the tail) would carry the first routing's values."""
        from cudnn.moe_dispatch_mlp_combine.cutedsl.scheduler import scheduler

        world, tokens, experts, topk = 8, 4096, 128, 6
        dense = _uniform(world, tokens, experts, topk, seed=1)
        capacity = world * tokens * topk + (experts // world) * 256
        # Sparse: only peer 2's first 100 tokens route to rank 3 (expert 50 of its 16), everything else elsewhere.
        sparse = torch.full((world, tokens, topk), 7, dtype=torch.int32, device="cuda")  # expert 7 is on rank 0
        sparse[2, :100, 0] = 50
        first = scheduler(dense, experts // world, capacity, 3)
        torch.cuda.synchronize()
        pointers = tuple(int(t.data_ptr()) for t in first)
        del first
        second = scheduler(sparse, experts // world, capacity, 3)
        torch.cuda.synchronize()
        self.assertEqual(
            tuple(int(t.data_ptr()) for t in second), pointers, "the second call did not reuse the first call's buffers; the test no longer covers stale data"
        )
        self._compare(sparse, experts // world, 3, capacity, second)


class SchedulerHost(unittest.TestCase):
    def test_output_and_scratch_sections_are_16_byte_aligned(self):
        """The consumers require 16-byte aligned schedule tensors; the output and scratch sections keep that for every E / EP."""
        from cudnn.moe_dispatch_mlp_combine.cutedsl.scheduler import output_sections, scratch_sections

        for experts, world in [(1, 1), (3, 8), (48, 8), (64, 8), (512, 1), (288, 4), (9, 64)]:
            total, padded, valid, output_size = output_sections(experts)
            min_counts, counts, chunk_counts, column_counts, scratch_size = scratch_sections(experts, world)
            with self.subTest(experts=experts, world=world):
                self.assertEqual((total, min_counts), (0, 0))
                for offset in (padded, valid, output_size, counts, chunk_counts, column_counts):
                    self.assertEqual(offset % 4, 0)
                self.assertGreaterEqual(valid - padded, experts)
                self.assertGreaterEqual(output_size - valid, experts)
                self.assertGreaterEqual(counts - min_counts, experts)
                self.assertGreaterEqual(chunk_counts - counts, experts * world)
                self.assertGreaterEqual(column_counts - chunk_counts, experts * world * 32)
                self.assertGreaterEqual(scratch_size - column_counts, experts * world * 1024)


@unittest.skipUnless(torch.cuda.is_available(), "the limits are read from the current device")
class SchedulerLimits(unittest.TestCase):
    def test_shared_memory_limits_at_their_exact_boundaries(self):
        """The host validation (no launch) accepts the largest E and EP the two shared-memory formulas allow on this device and
        rejects the next value with a ValueError naming the limit: 128 * E <= the opt-in shared memory per block (the count
        histogram; 1816 on B300 / GB300), and 256 * EP + 4 * E <= the bytes the schedule kernel's static block leaves for its
        runtime tables (EP 134 at E = 8)."""
        from cudnn.moe_dispatch_mlp_combine.cutedsl import scheduler as module
        from cudnn.moe_dispatch_mlp_combine.cutedsl.scheduler_schedule import _FIXED_BYTES, _GROUP, _SMEM_BYTES

        budget = torch.cuda.get_device_properties(torch.cuda.current_device()).shared_memory_per_block_optin
        max_experts = budget // 128
        routes = torch.zeros((1, 512, 2), dtype=torch.int32, device="cuda")
        module._validate_inputs(routes, max_experts, 1024, 0)  # accepted: returns the validated geometry
        with self.assertRaisesRegex(ValueError, "count histogram"):
            module._validate_inputs(routes, max_experts + 1, 1024, 0)
        experts = 8
        max_ranks = (_SMEM_BYTES - _FIXED_BYTES - 4 * experts) // (4 * _GROUP)
        module._validate_inputs(torch.zeros((max_ranks, 8, 1), dtype=torch.int32, device="cuda"), experts, 2048, 0)
        with self.assertRaisesRegex(ValueError, "schedule kernel's tables"):
            module._validate_inputs(torch.zeros((max_ranks + 1, 8, 1), dtype=torch.int32, device="cuda"), experts, 2048, 0)
        self.assertEqual((max_experts, max_ranks), (budget // 128, 134))


if __name__ == "__main__":
    unittest.main()
