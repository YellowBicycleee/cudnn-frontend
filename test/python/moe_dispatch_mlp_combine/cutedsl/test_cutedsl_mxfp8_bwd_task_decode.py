# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The MXFP8 backward's task -> expert decode variants (TaskDecode) give the same bytes.

Run under torchrun on SM100 / SM103 (8 GPUs), like ``test_cutedsl_mxfp8_bwd.py`` whose entries it
reuses.  The MXFP8 backward compiles one kernel per decode variant (the BF16 forward's
``TaskDecode``: the per-CTA expert table searched per expert for E <= EXPERT_TABLE_CAP local
experts, per expert group above it, or the linear scan) for the routed dgrad / replay owner decode and
the routed Wgrad K ranges.  For every geometry the host's choice is checked (``_assert_backward``) and
every forced variant must reproduce it byte for byte (the eighteen raw outputs, the peer-written buffers
and the eight public gradients).  The cases reach 257 and 384 local experts (group stride 2, a short and a
full last group, the group walk of the Wgrad K range).  A variant is forced by patching the host's selector,
the only selection point of ``persistent_mxfp8_bwd``.  The 9-column-block geometry has about two row blocks
per expert, so the tile swizzle differs from row-major order and a wrong expert start row or row count
would move tiles.  Each geometry's executors -- the backward's and the forward's -- are cleared afterwards,
so the other MXFP8 tests' executor counts do not depend on test order.  Both Gate / Up context precisions
run (``gate_up_context``, as in ``test_cutedsl_mxfp8_bwd.py``).
"""

from unittest import mock

import pytest
import torch

from cudnn.moe_dispatch_mlp_combine.cutedsl import persistent_mxfp8_bwd as mxfp8_bwd_host
from cudnn.moe_dispatch_mlp_combine.cutedsl import persistent_mxfp8_fwd as mxfp8_fwd_host
from cudnn.moe_dispatch_mlp_combine.cutedsl.persistent_bf16_fwd_contract import EXPERT_TABLE_CAP, TaskDecode
from cudnn.moe_dispatch_mlp_combine.cutedsl._arch import SUPPORTED_CAPABILITIES
from .test_cutedsl_mxfp8_bwd import RAW_OUTPUTS, UPSTREAM, _assert_backward, _backward, _written
from .test_cutedsl_mxfp8_fwd import _bytes

# (name, replayed macrobatches expected, (experts, hidden, intermediate, top-k, local tokens, comm SMs,
# minibatch, macrobatch)); the local expert count is experts / world size (8).  The schedule's
# capacity is 1.5 x 8 x tokens x top-k rows per rank; every local expert is padded to 256 rows.
CASES = (
    ("2 local experts, one macrobatch", False, (16, 512, 512, 2, 512, 8, 512, 4096)),
    ("64 local experts (Qwen-sized), replay", True, (512, 1024, 512, 4, 2048, 16, 1024, 2048)),
    ("16 local experts, 9 column blocks, about two row blocks per expert, replay", True, (128, 2304, 2304, 4, 2048, 16, 1024, 4096)),
    ("256 local experts (the table's capacity, stride 1), replay", True, (2048, 512, 512, 2, 4096, 16, 1024, 8192)),
    ("257 local experts (stride 2, last group of one), replay", True, (2056, 512, 512, 2, 4096, 16, 1024, 8192)),
    ("384 local experts (stride 2, full last group), replay", True, (3072, 512, 512, 4, 4096, 16, 1024, 8192)),
)


def _backward_with_decode(rank, world_size, device, geometry, task_decode, gate_up_context="bf16"):
    """``_backward`` with the MXFP8 backward's decode forced to ``task_decode`` (None = the host's choice);
    returns (raw, gradients, reference, rows, macrobatches)."""
    select = mxfp8_bwd_host.select_task_decode
    if task_decode is None:
        forced = select
    else:

        def forced(num_local_experts, requested=None):
            return select(num_local_experts, task_decode)

    with mock.patch.object(mxfp8_bwd_host, "select_task_decode", forced):
        return _backward(rank, world_size, device, geometry, None, gate_up_context)


def _assert_same_dsl(name, want, got, rows):
    raw_want, gradients_want = want
    raw_got, gradients_got = got
    for output, expected, actual in zip(RAW_OUTPUTS, raw_want[0], raw_got[0], strict=True):
        assert torch.equal(_bytes(_written(output, expected, rows)), _bytes(_written(output, actual, rows))), f"{name}: {output}"
    for expected, actual in zip(raw_want[1], raw_got[1], strict=True):
        assert torch.equal(_bytes(expected), _bytes(actual)), f"{name}: peer buffer"
    for index, (expected, actual) in enumerate(zip(gradients_want, gradients_got, strict=True)):
        assert torch.equal(_bytes(expected), _bytes(actual)), f"{name}: gradient {index}"


@pytest.mark.parametrize("gate_up_context", ("bf16", "mxfp8"))
def test_cutedsl_mxfp8_backward_task_decode_variants_agree(context, gate_up_context):
    rank, world_size, device = context
    if world_size != 8:
        pytest.skip("the MXFP8 backward geometries here are EP8")
    if torch.cuda.get_device_capability(device) not in SUPPORTED_CAPABILITIES:
        pytest.skip("the CuTe DSL MXFP8 backward is SM100/SM103 only")
    mxfp8_bwd_host.clear_persistent_mxfp8_bwd_executor_cache()
    mxfp8_fwd_host.clear_persistent_mxfp8_executor_cache()
    for name, replay, geometry in CASES:
        local_experts = geometry[0] // world_size
        fits_table = local_experts <= EXPERT_TABLE_CAP
        auto = mxfp8_bwd_host.select_task_decode(local_experts)
        expected = TaskDecode.SMEM_BINARY if fits_table else TaskDecode.SMEM_GROUP
        assert auto is expected, (name, local_experts, auto)
        raw, gradients, reference, rows, macrobatches = _backward_with_decode(rank, world_size, device, geometry, None, gate_up_context)
        assert (macrobatches > 1) == replay, (name, macrobatches)
        # The backward test's check; upstream's bar where UPSTREAM measured the geometry (only the first case).
        _assert_backward(raw, gradients, reference, rows, geometry, None, gate_up_context, upstream=(geometry, None) in UPSTREAM)
        for variant in TaskDecode:
            if variant is auto:
                continue
            if variant is TaskDecode.SMEM_BINARY and not fits_table:
                with pytest.raises(ValueError):
                    mxfp8_bwd_host.select_task_decode(local_experts, variant)
                continue
            result = _backward_with_decode(rank, world_size, device, geometry, variant, gate_up_context)
            _assert_same_dsl(f"{name}/{variant.name} vs {auto.name}", (raw, gradients), result[:2], rows)
            if rank == 0:
                print(f"{name}: {variant.name} == {auto.name} (E_local={local_experts}, macrobatches={macrobatches})")
        mxfp8_bwd_host.clear_persistent_mxfp8_bwd_executor_cache()
        mxfp8_fwd_host.clear_persistent_mxfp8_executor_cache()
        torch.cuda.empty_cache()
