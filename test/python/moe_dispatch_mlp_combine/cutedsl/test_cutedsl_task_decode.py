# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The three FWD task->expert decode variants (TaskDecode) give the same forward.

Run under torchrun on SM100 / SM103 (any world size; EP1 covers 512 local experts), e.g.
``torchrun --standalone --nproc-per-node=1 -m pytest -s moe_dispatch_mlp_combine/cutedsl/test_cutedsl_task_decode.py``.
For every shape the auto-selected variant's output is checked against the PyTorch BF16 reference
(``run_reference_bf16``) with the BF16 tolerance -- the CUDA backend it was once compared with is
gone, and a second run of the port is no check -- and every valid forced variant (SMEM_BINARY only when the local experts fit
the shared-memory prefix) must reproduce the auto variant's outputs bit for bit.
"""

from unittest import mock

import pytest
import torch
import torch.distributed as dist

from cudnn.moe_dispatch_mlp_combine import functional
from cudnn.moe_dispatch_mlp_combine.cutedsl import persistent_bf16_fwd
from cudnn.moe_dispatch_mlp_combine.cutedsl.persistent_bf16_fwd_contract import EXPERT_TABLE_CAP, TaskDecode
from cudnn.moe_dispatch_mlp_combine.functional import get_workspace
from cudnn.moe_dispatch_mlp_combine.cutedsl._arch import SUPPORTED_CAPABILITIES

from .utils import BF16_TOLERANCE, check_correctness, generate_inputs, run_reference_bf16

# (name, local experts, experts that receive tokens (per rank), hidden, intermediate, top-k, local tokens).
# With 4096 tokens x top-k 8 = 32768 routed rows per rank the mean rows per receiving expert are
# 32768 / routed experts; a 2048-row minibatch holds 8 row blocks.  Coverage:
#   stride 1 (E <= 256, SMEM_BINARY auto) with one block per expert (256) and with multi-block experts
#   (64: ~2 blocks; 16: ~8 blocks, so experts span minibatch boundaries).  These two use H = 4096 and
#   I = 2048, i.e. 16 column blocks for FC1 and Down: the tile swizzle (supergroups of 8 columns) only
#   differs from row-major order with more than 8 column blocks, and only then does a wrong row-block
#   origin or row count inside a minibatch move tiles;
#   stride 2 with a full last group (512), a short last group (257, 383) and empty experts including a
#   whole empty last group (383 with tokens for the first 300 only; EP1 only, local expert indices);
#   stride 3 / 4 (600 / 769) with a short last group, and 600 with multi-block experts and empty groups.
CASES = (
    ("512 local experts (Qwen3.5 shape, stride 2)", 512, 512, 4096, 1024, 10, 4096),
    ("256 local experts (hidden 6144, stride 1)", 256, 256, 6144, 2048, 8, 2048),
    ("64 local experts, ~2 row blocks each, 16 column blocks (stride 1)", 64, 64, 4096, 2048, 8, 4096),
    ("16 local experts, ~8 row blocks each, spanning minibatches, 16 column blocks (stride 1)", 16, 16, 4096, 2048, 8, 4096),
    ("257 local experts (stride 2, last group of one)", 257, 257, 256, 256, 8, 4096),
    ("383 local experts (stride 2, last group of one)", 383, 383, 256, 256, 8, 4096),
    ("383 local experts, experts 300..382 empty (empty last group)", 383, 300, 256, 256, 8, 4096),
    ("600 local experts (stride 3, groups of 3, last group of three)", 600, 600, 256, 256, 8, 4096),
    ("600 local experts, tokens for the first 100 only (stride 3, multi-block experts, empty groups)", 600, 100, 256, 256, 8, 4096),
    ("769 local experts (stride 4, last group of one)", 769, 769, 256, 256, 8, 4096),
)


def _forward(config, workspace, schedule, inputs, task_decode):
    """functional.forward with the task->expert decode forced to ``task_decode`` (None = the host's
    choice) by patching the internal selector, the only selection point of forward_bf16."""
    x, _, router_weights, w_shared_gate, w_shared_up, w_shared_down, w_routed_gate, w_routed_up, w_routed_down, _ = inputs
    select = persistent_bf16_fwd.select_task_decode
    forced = select if task_decode is None else (lambda num_local_experts, requested=None: select(num_local_experts, task_decode))
    with mock.patch.object(persistent_bf16_fwd, "select_task_decode", forced):
        output, context = functional.forward(
            config, workspace, schedule, x, router_weights, w_shared_gate, w_shared_up, w_shared_down, w_routed_gate, w_routed_up, w_routed_down, None
        )
    torch.cuda.synchronize(x.device)  # a device fault is reported against this call, not a later one
    return (
        output,
        context.x_routed,
        context.gate_shared,
        context.gate_routed,
        context.up_shared,
        context.up_routed,
        context.hidden_shared,
        context.hidden_routed,
    )


def test_shared_storage_layout(context: tuple[int, int, torch.device]) -> None:
    """The expert table lies after every mbarrier and before the 1024-aligned arena (the module
    asserts it at import); print the layout so the padding after the table is visible."""
    rank, _, device = context
    if torch.cuda.get_device_capability(device) not in SUPPORTED_CAPABILITIES:
        pytest.skip("CuTe DSL backends are SM100/SM103 only")
    from cudnn.moe_dispatch_mlp_combine.cutedsl import _persistent_bf16_fwd_mega as mega

    layout = mega.shared_storage_layout()
    by_name = {name: (offset, size) for name, offset, size in layout}
    table_offset, table_bytes = by_name["expert_cu_end"]
    arena_offset = by_name["arena"][0]
    assert table_bytes == EXPERT_TABLE_CAP * 4
    assert all(offset + size <= table_offset for name, (offset, size) in by_name.items() if "mbarrier" in name)
    assert table_offset + table_bytes <= arena_offset and arena_offset % 1024 == 0
    # The decoded-task record ring (CLC warp -> other warps): every field has one Int32 word, the
    # stages are 16-byte aligned and lie with the barriers before the table.
    record_offset, record_bytes = by_name["task_info"]
    assert len(set(mega.TASK_INFO_FIELDS)) == len(mega.TASK_INFO_FIELDS) <= mega.TASK_INFO_WORDS
    assert record_bytes == mega.TASK_INFO_STAGES * mega.TASK_INFO_WORDS * 4 and record_offset % 16 == 0
    assert record_offset + record_bytes <= table_offset
    if rank == 0:
        for name, offset, size in layout:
            print(f"SMEM {name}: offset {offset} size {size} end {offset + size}")
        print(f"SMEM padding after expert_cu_end: {arena_offset - table_offset - table_bytes} B before the arena at {arena_offset}")


def test_task_decode_variants_agree(context: tuple[int, int, torch.device]) -> None:
    rank, world_size, device = context
    if torch.cuda.get_device_capability(device) not in SUPPORTED_CAPABILITIES:
        pytest.skip("CuTe DSL backends are SM100/SM103 only")
    for name, num_local_experts, routed_local_experts, hidden_dim, intermediate_dim, topk, num_local_tokens in CASES:
        if routed_local_experts != num_local_experts and world_size != 1:
            continue  # restricted routing is expressed in local expert indices
        num_experts = num_local_experts * world_size
        # Schedule capacity: every local expert is padded to a 256-row block, so the padded row count
        # can reach E_local * 256 + routed rows (the CUDA scheduler traps above the capacity).  The
        # capacity factor is max(2, ceil(world * multiplier)) times T * top-k.
        routed_rows = num_local_tokens * topk
        capacity_factor = -(-(num_local_experts * 256 + routed_rows * world_size) // routed_rows) + 1
        geometry = dict(fwd_num_comm_sms=30, bwd_num_comm_sms=14, macrobatch_size=8192, schedule_capacity_multiplier=capacity_factor / world_size)
        dsl_config = functional.CuteMoKConfig(fwd_minibatch_size=2048, bwd_minibatch_size=2048, **geometry)
        workspace = get_workspace(dsl_config, dist.group.WORLD, device=device, num_local_tokens=num_local_tokens, hidden_size=hidden_dim, topk=topk)
        # Routing over the first routed_local_experts * world_size experts only; the weights
        # cover all num_local_experts, so the remaining experts receive no rows.
        inputs = generate_inputs(rank, device, routed_local_experts * world_size, num_local_experts, topk, num_local_tokens, hidden_dim, intermediate_dim)
        topk_experts = inputs[1]
        dsl_schedule = functional.build_schedule(workspace, dsl_config, topk_experts, num_local_experts=num_local_experts)

        auto = persistent_bf16_fwd.select_task_decode(num_local_experts)
        expected_auto = TaskDecode.SMEM_BINARY if num_local_experts <= EXPERT_TABLE_CAP else TaskDecode.SMEM_GROUP
        assert auto is expected_auto, (name, num_local_experts, auto)
        stride = -(-num_local_experts // EXPERT_TABLE_CAP)
        expected_output = run_reference_bf16(*inputs)[0]
        reference = [t.clone() for t in _forward(dsl_config, workspace, dsl_schedule, inputs, None)]
        check_correctness(f"{name}/auto={auto.name} vs reference", expected_output, reference[0], BF16_TOLERANCE, print_stats=rank == 0)
        for variant in TaskDecode:
            if variant is TaskDecode.SMEM_BINARY and num_local_experts > EXPERT_TABLE_CAP:
                with pytest.raises(ValueError):
                    persistent_bf16_fwd.select_task_decode(num_local_experts, variant)
                continue
            result = _forward(dsl_config, workspace, dsl_schedule, inputs, variant)
            for index, (want, got) in enumerate(zip(reference, result, strict=True)):
                assert torch.equal(want, got), f"{name}/{variant.name}: output {index} differs from {auto.name}"
            if rank == 0:
                print(f"{name}: {variant.name} == {auto.name} (E_local={num_local_experts}, stride {stride})")
        del workspace, inputs, reference, result
        torch.cuda.empty_cache()
