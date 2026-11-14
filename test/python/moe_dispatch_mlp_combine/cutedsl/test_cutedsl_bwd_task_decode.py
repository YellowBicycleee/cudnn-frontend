# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The three BWD task->expert decode variants (TaskDecode) give the same backward.

Run under torchrun on SM100 / SM103 (any world size; EP1 covers 512 local experts), e.g.
``torchrun --standalone --nproc-per-node=1 -m pytest -s moe_dispatch_mlp_combine/cutedsl/test_cutedsl_bwd_task_decode.py``.
For every shape the auto-selected variant is checked against the PyTorch BF16 reference
(``run_reference_bf16``) with the BF16 tolerance -- the CUDA backend it was once compared with is
gone, and comparing the port with a second run of itself checked nothing -- and every valid forced
variant (SMEM_BINARY only when the local experts fit the
shared-memory table) must reproduce the auto variant's gradients bit for bit.  Every backward runs
on a fresh forward context of the same CuTe DSL forward (the routed Down dgrad epilogue overwrites
the context's hidden_routed with w * Hidden, so a context serves exactly one backward); the forward's
own decode is not forced here.

``test_bwd_routed_dgrad_partial_blocks`` routes a fixed number of rows to every local expert so that
the experts' last 256-row blocks hold 1, 15, 17, 59, 64, 100, 127, 129, 255 and 1 (257 rows) valid rows: the
swap-AB routed Gate/Up dgrad's N (align16 of the valid rows, at least 16, below and above the 128-row
cta_group::2 split, exactly 256) and the Wgrad K skip at a non-multiple of 64 are all exercised, and a
wrong N, a wrong non-leader B offset or an unstored valid dX row shows up against the reference.
"""

from unittest import mock

import pytest
import torch
import torch.distributed as dist

from cudnn.moe_dispatch_mlp_combine import functional
from cudnn.moe_dispatch_mlp_combine.cutedsl import persistent_bf16_bwd
from cudnn.moe_dispatch_mlp_combine.cutedsl.persistent_bf16_bwd_contract import BWD_EXPERT_TABLE_CAP, TaskDecode
from cudnn.moe_dispatch_mlp_combine.functional import get_workspace
from cudnn.moe_dispatch_mlp_combine.cutedsl._arch import SUPPORTED_CAPABILITIES

from .utils import ungated, BF16_TOLERANCE, check_correctness, generate_inputs, run_reference_bf16

GRADIENT_NAMES = ("d_x", "d_router_weights", "d_w_routed_gate", "d_w_routed_up", "d_w_routed_down", "d_w_shared_gate", "d_w_shared_up", "d_w_shared_down")
# (name, local experts, experts that receive tokens (per rank), hidden, intermediate, top-k, local tokens,
#  macrobatch).  Minibatch 2048 = 8 row blocks.  Coverage of the BWD decode:
#   stride 1 (E <= 512 = BWD_EXPERT_TABLE_CAP, SMEM_BINARY auto): 512 (one entry per expert at the capacity),
#   383 and 257 (above the forward's 256), 256, 128 and 96, 64 (~2 row blocks per expert) and 16 (~8 row blocks
#   per expert, so experts span minibatch boundaries).  All of the larger shapes use H = 4096 (16 column blocks for
#   the Gate/Up dgrad) or I = 2048 (8 column blocks for Down dgrad; 16 for the 6144/2048 shape): the tile swizzle
#   (supergroups of 8 columns) only differs from row-major order with more than 8 column blocks, and only then does
#   a wrong row-block origin or row count inside a minibatch move tiles.  16 experts with 1024 tokens fit one
#   macrobatch of 16384 rows, which takes the kernel's macro fallback (dense per-expert Wgrad grid, generation
#   decode without the owner buckets); every other case has several macrobatches (minibatch release, replay).
#   stride 2 with tokens for the first 100 experts only and empty groups (600), and a last group of one (769);
#   stride 3 with empty groups (1100, tokens for the first 200 only).  Empty experts at stride 1: 128 with tokens
#   for the first 100 only and 383 with tokens for the first 300 only (EP1 only, local expert indices).  The decode
#   variant is the only compile-time input: after every case each variant has compiled at most one executor (the
#   expert count and the sizes are runtime values).
CASES = (
    ("512 local experts (Qwen3.5 shape, stride 1 at the BWD table capacity)", 512, 512, 4096, 1024, 10, 4096, 8192),
    ("256 local experts (hidden 6144, stride 1)", 256, 256, 6144, 2048, 8, 2048, 8192),
    ("128 local experts, 16 column blocks (stride 1, above the old BWD cap)", 128, 128, 4096, 2048, 8, 4096, 8192),
    ("96 local experts, 16 column blocks (stride 1, above the old BWD cap)", 96, 96, 4096, 2048, 8, 4096, 8192),
    ("64 local experts, ~2 row blocks each, 16 column blocks (stride 1)", 64, 64, 4096, 2048, 8, 4096, 8192),
    ("128 local experts, experts 100..127 empty, 16 column blocks (stride 1)", 128, 100, 4096, 2048, 8, 4096, 8192),
    ("16 local experts, ~8 row blocks each, spanning minibatches, 16 column blocks (stride 1)", 16, 16, 4096, 2048, 8, 4096, 8192),
    ("16 local experts, one macrobatch (macro fallback: dense Wgrad grid), 16 column blocks", 16, 16, 4096, 2048, 8, 1024, 16384),
    ("257 local experts (stride 1, above the forward's table)", 257, 257, 256, 256, 8, 4096, 8192),
    ("383 local experts, experts 300..382 empty (stride 1)", 383, 300, 256, 256, 8, 4096, 8192),
    ("600 local experts, tokens for the first 100 only (stride 2, multi-block experts, empty groups)", 600, 100, 256, 256, 8, 4096, 8192),
    ("769 local experts (stride 2, last group of one)", 769, 769, 256, 256, 8, 4096, 8192),
    ("1100 local experts, tokens for the first 200 only (stride 3, empty groups)", 1100, 200, 256, 256, 8, 4096, 8192),
)


def _backward(config, workspace, schedule, forward_context, inputs, task_decode):
    """functional.backward with the BWD task->expert decode forced to ``task_decode`` (None = the
    host's choice) by patching the BWD module's selector, the only selection point of backward_bf16."""
    x, _, router_weights, w_shared_gate, w_shared_up, w_shared_down, w_routed_gate, w_routed_up, w_routed_down, d_output = inputs
    select = persistent_bf16_bwd.select_task_decode
    forced = select if task_decode is None else (lambda num_local_experts, requested=None: select(num_local_experts, task_decode))
    with mock.patch.object(persistent_bf16_bwd, "select_task_decode", forced):
        grads = ungated(
            functional.backward(
                config,
                workspace,
                schedule,
                forward_context,
                d_output,
                x,
                router_weights,
                w_shared_gate,
                w_shared_up,
                w_shared_down,
                w_routed_gate,
                w_routed_up,
                w_routed_down,
                None,
            )
        )
    torch.cuda.synchronize(x.device)  # a device fault is reported against this call, not a later one
    return tuple(grads)


def test_shared_storage_layout(context: tuple[int, int, torch.device]) -> None:
    """The expert table lies in the arena tail, after every compute-role staging area (GEMM A/B/D)
    and inside the arena; the storage itself is barriers + arena (the module asserts both at
    import).  Print the layout so the placement is visible."""
    rank, _, device = context
    if torch.cuda.get_device_capability(device) not in SUPPORTED_CAPABILITIES:
        pytest.skip("CuTe DSL backends are SM100/SM103 only")
    from cudnn.moe_dispatch_mlp_combine.cutedsl import _persistent_bf16_bwd_mega as mega

    layout = mega.shared_storage_layout()
    by_name = {name: (offset, size) for name, offset, size in layout}
    arena_offset = by_name["arena"][0]
    table_begin = mega.EXPERT_TABLE_ARENA_OFFSET
    table_end = table_begin + BWD_EXPERT_TABLE_CAP * 4
    assert "expert_valid_end" not in by_name
    assert table_begin >= mega.GEMM_ARENA_BYTES
    assert table_end <= mega.REUSABLE_ARENA_BYTES and arena_offset % 1024 == 0
    assert mega.DYNAMIC_SHARED_MEMORY_BYTES == arena_offset + mega.REUSABLE_ARENA_BYTES < 232448
    # The decoded-task record ring (CLC warp -> other warps): every field has one Int32 word, the
    # stages are 16-byte aligned and lie with the barriers before the arena; the ten geometry words
    # are the _gemm_geometry tuple in its order.
    record_offset, record_bytes = by_name["task_info"]
    assert len(set(mega.TASK_INFO_FIELDS)) == len(mega.TASK_INFO_FIELDS) <= mega.TASK_INFO_WORDS
    assert record_bytes == mega.TASK_INFO_STAGES * mega.TASK_INFO_WORDS * 4 and record_offset % 16 == 0
    assert record_offset + record_bytes <= arena_offset
    assert mega.TASK_INFO_GEOMETRY_FIELDS == (
        "valid",
        "expert",
        "row",
        "column",
        "k_begin",
        "k_tiles",
        "wgrad_first",
        "wgrad_has_next",
        "wgrad_empty",
        "tile_valid",
    )
    if rank == 0:
        for name, offset, size in layout:
            print(f"SMEM {name}: offset {offset} size {size} end {offset + size}")
        print(
            f"SMEM expert table: arena [{table_begin}, {table_end}) of {mega.REUSABLE_ARENA_BYTES}; "
            f"dynamic shared memory {mega.DYNAMIC_SHARED_MEMORY_BYTES} B"
        )


def test_bwd_task_decode_variants_agree(context: tuple[int, int, torch.device]) -> None:
    rank, world_size, device = context
    if torch.cuda.get_device_capability(device) not in SUPPORTED_CAPABILITIES:
        pytest.skip("CuTe DSL backends are SM100/SM103 only")
    # The count below is over the whole executor cache, so this test has to own it: another test
    # file in the same session leaves backward executors of its own behind.
    persistent_bf16_bwd.clear_persistent_bf16_bwd_executor_cache()
    failures = []  # every case runs; the mismatches are reported together at the end
    for name, num_local_experts, routed_local_experts, hidden_dim, intermediate_dim, topk, num_local_tokens, macrobatch_size in CASES:
        if routed_local_experts != num_local_experts and world_size != 1:
            continue  # restricted routing is expressed in local expert indices
        num_experts = num_local_experts * world_size
        # Schedule capacity: every local expert is padded to a 256-row block, so the padded row count
        # can reach E_local * 256 + routed rows (the CUDA scheduler traps above the capacity).  The
        # capacity factor is max(2, ceil(world * multiplier)) times T * top-k.
        routed_rows = num_local_tokens * topk
        capacity_factor = -(-(num_local_experts * 256 + routed_rows * world_size) // routed_rows) + 1
        geometry = dict(fwd_num_comm_sms=30, bwd_num_comm_sms=14, macrobatch_size=macrobatch_size, schedule_capacity_multiplier=capacity_factor / world_size)
        config = functional.CuteMoKConfig(minibatch_size=2048, **geometry)
        workspace = get_workspace(config, dist.group.WORLD, device=device, num_local_tokens=num_local_tokens, hidden_size=hidden_dim, topk=topk)
        # Routing over the first routed_local_experts * world_size experts only; the weights
        # cover all num_local_experts, so the remaining experts receive no rows.
        inputs = generate_inputs(rank, device, routed_local_experts * world_size, num_local_experts, topk, num_local_tokens, hidden_dim, intermediate_dim)
        x, topk_experts, router_weights, w_shared_gate, w_shared_up, w_shared_down, w_routed_gate, w_routed_up, w_routed_down, _ = inputs
        dsl_schedule = functional.build_schedule(workspace, config, topk_experts, num_local_experts=num_local_experts)

        auto = persistent_bf16_bwd.select_task_decode(num_local_experts)
        expected_auto = TaskDecode.SMEM_BINARY if num_local_experts <= BWD_EXPERT_TABLE_CAP else TaskDecode.SMEM_GROUP
        assert auto is expected_auto, (name, num_local_experts, auto)
        stride = -(-num_local_experts // BWD_EXPERT_TABLE_CAP)
        forward_args = (x, router_weights, w_shared_gate, w_shared_up, w_shared_down, w_routed_gate, w_routed_up, w_routed_down, None)
        baseline_grads = run_reference_bf16(*inputs)[1:]

        def dsl_backward(task_decode):
            _, dsl_context = functional.forward(config, workspace, dsl_schedule, *forward_args)
            torch.cuda.synchronize(device)
            return [t.clone() for t in _backward(config, workspace, dsl_schedule, dsl_context, inputs, task_decode)]

        reference = dsl_backward(None)
        for grad_name, want, got in zip(GRADIENT_NAMES, baseline_grads, reference, strict=True):
            check_correctness(f"{name}/auto={auto.name} {grad_name} vs reference", want, got, BF16_TOLERANCE, print_stats=rank == 0)
        for variant in TaskDecode:
            if variant is TaskDecode.SMEM_BINARY and num_local_experts > BWD_EXPERT_TABLE_CAP:
                with pytest.raises(ValueError):
                    persistent_bf16_bwd.select_task_decode(num_local_experts, variant)
                continue
            result = dsl_backward(variant)
            for grad_name, want, got, baseline_grad in zip(GRADIENT_NAMES, reference, result, baseline_grads, strict=True):
                if not torch.equal(want, got):
                    # Which side is off: the variant's distance to the reference next to the auto variant's.
                    differing = int((want != got).sum())
                    err = lambda t: float((t.float() - baseline_grad.float()).abs().max())
                    failures.append(
                        f"{name}/{variant.name}: {grad_name} differs from {auto.name} in {differing} of "
                        f"{got.numel()} elements (max |diff| {float((want.float() - got.float()).abs().max()):.3e}); "
                        f"max |err| vs reference: {auto.name} {err(want):.3e}, {variant.name} {err(got):.3e}"
                    )
            if rank == 0:
                print(f"{name}: BWD {variant.name} vs {auto.name} checked (E_local={num_local_experts}, stride {stride})")
        del workspace, inputs, reference, result, baseline_grads
        torch.cuda.empty_cache()
    # Grouped executors only: the routed-weight layout is another compile-time axis, so a split
    # test earlier in the session would otherwise make every variant look compiled twice.
    compiled = [
        part[1]
        for key in persistent_bf16_bwd._EXECUTOR_CACHE
        if ("routed_weights", "grouped") in key
        for part in key
        if isinstance(part, tuple) and len(part) == 2 and part[0] == "task_decode"
    ]
    assert compiled and all(compiled.count(variant) == 1 for variant in set(compiled)), compiled
    assert not failures, "\n".join(failures)


# Routed rows per local expert of the partial-block case (sum = 2048 tokens x top-k 8; each count is at most
# the token count so a token never routes to the same expert twice).  Valid rows of each expert's last
# 256-row block: 1, 15, 17, 100, 127, 129, 255, 1 (257), 59 (1595), 64 (1600) and 256 (2048, no padding).
PARTIAL_BLOCK_COUNTS = (1, 15, 17, 100, 127, 129, 255, 257, 1595, 1600, 2048, 2048, 2048, 2048, 2048, 2048)
# (name, macrobatch): three macrobatches (18176 padded rows = 71 row blocks, 9 minibatches of 2048: replay,
# ring slot reuse) and one macrobatch (CUDA's macro fallback).
PARTIAL_BLOCK_CASES = (("partial last blocks, 3 macrobatches", 8192), ("partial last blocks, 1 macrobatch", 32768))


def _routing_with_counts(rank: int, device: torch.device, num_local_experts: int, counts, topk: int, num_local_tokens: int) -> torch.Tensor:
    """top-k expert indices (num_local_tokens, topk) that route exactly ``counts[e]`` rows of this rank's
    tokens to this rank's local expert ``e`` (global index rank * E_local + e; no row goes to another rank, so
    every rank's local counts are known).  The sorted expert list is laid out so that token t takes
    positions t, t + T, ...: a run of one expert is at most T long, so a token never repeats an expert."""
    counts = torch.tensor(counts, dtype=torch.int64)
    assert len(counts) == num_local_experts and int(counts.sum()) == topk * num_local_tokens
    assert int(counts.max()) <= num_local_tokens
    flat = torch.repeat_interleave(torch.arange(num_local_experts, dtype=torch.int64), counts)
    topk_experts = flat.view(topk, num_local_tokens).t().contiguous() + rank * num_local_experts
    return topk_experts.to(device)


def test_bwd_routed_dgrad_partial_blocks(context: tuple[int, int, torch.device]) -> None:
    rank, world_size, device = context
    if torch.cuda.get_device_capability(device) not in SUPPORTED_CAPABILITIES:
        pytest.skip("CuTe DSL backends are SM100/SM103 only")
    num_local_experts, topk, num_local_tokens, hidden_dim, intermediate_dim = len(PARTIAL_BLOCK_COUNTS), 8, 2048, 512, 512
    num_experts = num_local_experts * world_size
    for name, macrobatch_size in PARTIAL_BLOCK_CASES:
        routed_rows = num_local_tokens * topk
        capacity_factor = -(-(num_local_experts * 256 + routed_rows * world_size) // routed_rows) + 1
        geometry = dict(fwd_num_comm_sms=30, bwd_num_comm_sms=14, macrobatch_size=macrobatch_size, schedule_capacity_multiplier=capacity_factor / world_size)
        config = functional.CuteMoKConfig(minibatch_size=2048, **geometry)
        workspace = get_workspace(config, dist.group.WORLD, device=device, num_local_tokens=num_local_tokens, hidden_size=hidden_dim, topk=topk)
        inputs = list(generate_inputs(rank, device, num_experts, num_local_experts, topk, num_local_tokens, hidden_dim, intermediate_dim))
        inputs[1] = _routing_with_counts(rank, device, num_local_experts, PARTIAL_BLOCK_COUNTS, topk, num_local_tokens)
        inputs = tuple(inputs)
        x, topk_experts, router_weights, w_shared_gate, w_shared_up, w_shared_down, w_routed_gate, w_routed_up, w_routed_down, _ = inputs
        dsl_schedule = functional.build_schedule(workspace, config, topk_experts, num_local_experts=num_local_experts)
        # The routing landed as intended: the padded and the unpadded counts.
        assert tuple(int(v) for v in dsl_schedule.tokens_per_expert.tolist()) == tuple(-(-c // 256) * 256 for c in PARTIAL_BLOCK_COUNTS), name
        assert tuple(int(v) for v in dsl_schedule.valid_tokens_per_expert.tolist()) == PARTIAL_BLOCK_COUNTS, name
        forward_args = (x, router_weights, w_shared_gate, w_shared_up, w_shared_down, w_routed_gate, w_routed_up, w_routed_down, None)
        baseline_grads = run_reference_bf16(*inputs)[1:]
        _, dsl_context = functional.forward(config, workspace, dsl_schedule, *forward_args)
        torch.cuda.synchronize(device)
        dsl_grads = _backward(config, workspace, dsl_schedule, dsl_context, inputs, None)
        for grad_name, want, got in zip(GRADIENT_NAMES, baseline_grads, dsl_grads, strict=True):
            assert torch.isfinite(got.float()).all(), f"{name}: {grad_name} has non-finite values"
            check_correctness(f"{name} {grad_name} vs reference", want, got, BF16_TOLERANCE, print_stats=rank == 0)
        del workspace, inputs, baseline_grads, dsl_grads, dsl_context
        torch.cuda.empty_cache()
