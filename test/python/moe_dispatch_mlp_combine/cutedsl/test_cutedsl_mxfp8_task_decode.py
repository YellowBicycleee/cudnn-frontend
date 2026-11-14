# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The MXFP8 forward's task -> expert decode variants (TaskDecode) give the same bytes.

Run under torchrun on SM100 / SM103 (8 GPUs), like ``test_cutedsl_mxfp8_fwd.py`` whose entries it
reuses.  The MXFP8 forward compiles one kernel per decode variant (the BF16 forward's
``TaskDecode``: the shared expert table searched per expert for E <= EXPERT_TABLE_CAP local experts,
per expert group above it, or the linear scan).  For every geometry the host's choice is checked, and
every forced variant must reproduce its thirteen outputs (the BF16 Gate / Up context quantized by
``mxfp8_quantize``, as ``test_cutedsl_mxfp8_fwd.py`` does), the combine buffer and the epilogue
output byte for byte.  Both Gate / Up context precisions run.  A variant is forced by patching the
host's selector, the only selection point of ``forward_mxfp8``.  Each geometry's executors are
cleared afterwards, so the other MXFP8 tests' executor counts do not depend on test order.
"""

from unittest import mock

import pytest
import torch
import torch.distributed as dist

from cudnn.moe_dispatch_mlp_combine import functional as f
from cudnn.moe_dispatch_mlp_combine.cutedsl import persistent_mxfp8_fwd as mxfp8_host
from cudnn.moe_dispatch_mlp_combine.cutedsl.persistent_bf16_fwd_contract import EXPERT_TABLE_CAP, TaskDecode
from cudnn.moe_dispatch_mlp_combine.ops import mxfp8_quantize
from cudnn.moe_dispatch_mlp_combine.cutedsl._arch import SUPPORTED_CAPABILITIES
from .test_cutedsl_mxfp8_fwd import OUTPUTS, _bytes, _run, _written
from .utils import generate_inputs

# (name, experts, hidden, intermediate, top-k, local tokens, comm SMs, minibatch, macrobatch); the
# local expert count is experts / world size (8): 16 and 64 with several row blocks per expert, 256 at
# the table's capacity (stride 1), 257 and 384 above it (stride 2, a short and a full last group).
CASES = (
    ("16 local experts, several row blocks each", 128, 1024, 768, 4, 4096, 16, 2048, 8192),
    ("64 local experts, 9 column blocks (the tile swizzle differs from row-major order)", 512, 2304, 2304, 4, 2048, 16, 1024, 4096),
    ("64 local experts (Qwen-sized)", 512, 1024, 512, 4, 2048, 16, 1024, 4096),
    ("256 local experts (the table's capacity, stride 1)", 2048, 512, 512, 2, 4096, 16, 1024, 8192),
    ("257 local experts (stride 2, last group of one)", 2056, 512, 512, 2, 4096, 16, 1024, 8192),
    ("384 local experts (stride 2, full last group)", 3072, 512, 512, 2, 4096, 16, 1024, 8192),
)


def _forward(config, workspace, schedule, inputs, task_decode, device, gate_up_context="bf16"):
    """One backend's forward through its low-level entry; ``task_decode`` forces the MXFP8 kernel's
    decode variant (None = the host's choice)."""
    x, router_weights, shared, routed = inputs
    select = mxfp8_host.select_task_decode
    if task_decode is None:
        forced = select
    else:

        def forced(num_local_experts, requested=None):
            return select(num_local_experts, task_decode)

    with mock.patch.object(mxfp8_host, "select_task_decode", forced):
        return _run(config, workspace, schedule, x, router_weights, shared, routed, None, device, gate_up_context)[:3]


def _assert_same(name, want, got, rows):
    outputs_want, combine_want, y_want = want
    outputs_got, combine_got, y_got = got
    for output, expected, actual in zip(OUTPUTS, outputs_want, outputs_got, strict=True):
        assert torch.equal(_bytes(_written(output, expected, rows)), _bytes(_written(output, actual, rows))), f"{name}: {output}"
    assert torch.equal(_bytes(combine_want), _bytes(combine_got)), f"{name}: combine_buffer"
    assert torch.equal(_bytes(y_want), _bytes(y_got)), f"{name}: output"


@pytest.mark.parametrize("gate_up_context", ("bf16", "mxfp8"))
def test_cutedsl_mxfp8_task_decode_variants_agree(context, gate_up_context):
    rank, world_size, device = context
    if world_size != 8:
        pytest.skip("the MXFP8 forward geometries here are EP8")
    if torch.cuda.get_device_capability(device) not in SUPPORTED_CAPABILITIES:
        pytest.skip("the CuTe DSL MXFP8 forward is SM100/SM103 only")
    mxfp8_host.clear_persistent_mxfp8_executor_cache()
    for name, experts, hidden, intermediate, topk, tokens, comm, minibatch, macrobatch in CASES:
        local_experts = experts // world_size
        # Every local expert is padded to a 256-row block, so the schedule must hold
        # local_experts * 256 rows plus the routed rows of every rank.
        routed_rows = tokens * topk
        capacity_factor = -(-(local_experts * 256 + routed_rows * world_size) // routed_rows) + 1
        config = f.CuteMoKConfig(
            fwd_num_comm_sms=comm, minibatch_size=minibatch, macrobatch_size=macrobatch, schedule_capacity_multiplier=capacity_factor / world_size
        )
        workspace = f.create_workspace(config, dist.group.WORLD, device=device, num_local_tokens=tokens, hidden_size=hidden, topk=topk)
        x, routes, router_weights, shared_gate, shared_up, shared_down, gate, up, down, _ = generate_inputs(
            rank, device, experts, local_experts, topk, tokens, hidden, intermediate
        )
        schedule = f.build_schedule(workspace, config, routes, num_local_experts=local_experts)
        inputs = (x, router_weights, (shared_gate, shared_up, shared_down), [mxfp8_quantize(weight, True, False)[:2] for weight in (gate, up, down)])
        rows = min(int(schedule.num_tokens[0]), config.macrobatch_size)

        auto = mxfp8_host.select_task_decode(local_experts)
        expected = TaskDecode.SMEM_BINARY if local_experts <= EXPERT_TABLE_CAP else TaskDecode.SMEM_GROUP
        assert auto is expected, (name, local_experts, auto)
        reference = _forward(config, workspace, schedule, inputs, None, device, gate_up_context)
        for variant in TaskDecode:
            if variant is TaskDecode.SMEM_BINARY and local_experts > EXPERT_TABLE_CAP:
                with pytest.raises(ValueError):
                    mxfp8_host.select_task_decode(local_experts, variant)
                continue
            result = _forward(config, workspace, schedule, inputs, variant, device, gate_up_context)
            _assert_same(f"{name}/{variant.name} vs {auto.name}", reference, result, rows)
            if rank == 0:
                print(f"{name}: {variant.name} == {auto.name} (E_local={local_experts})")
        del workspace, inputs, reference, result
        mxfp8_host.clear_persistent_mxfp8_executor_cache()
        torch.cuda.empty_cache()
