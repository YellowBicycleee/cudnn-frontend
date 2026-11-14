# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The communication tile shape of the BF16 megakernels moves bytes only: 32 rows x 512 columns and
16 rows x 1024 columns give the same bits.

The shape (contract ``CommTileShape``, one of ``COMM_TILE_SHAPES``) is the compile-time parameter of the
communication CTAs of the forward and the backward megakernel, selected by ``CuteMoKConfig.comm_tile_shape``.
Here one forward + backward runs per shape, the shape set through that config field, over the same inputs: the
output and every gradient must be bit-identical, each shape must have compiled its own forward and backward
executor, and the 16 x 1024 results are also checked against the PyTorch reference.  The two runs share one
workspace and the caching allocator, so a row one shape fails to move could pass on the bytes the other left
behind: before each run the peer-written workspace buffers and every fresh forward output and backward scratch /
gradient buffer are filled with a finite value that differs between the shapes.  H = 4096 (whole column blocks
in both shapes) and H = 1152 (a 128-column last block in both), one macrobatch and several (X replay, pushes mixed
with pulls), and the forward with the combine in the routed Down epilogue (the stream's pulls only).

Run under torchrun on eight GPUs.
"""

import contextlib
import dataclasses
from unittest import mock

import pytest
import torch
import torch.distributed as dist

from cudnn.moe_dispatch_mlp_combine import functional
from cudnn.moe_dispatch_mlp_combine.cutedsl import persistent_bf16_bwd, persistent_bf16_fwd
from cudnn.moe_dispatch_mlp_combine.cutedsl._arch import TARGET_ARCH, TARGET_CAPABILITY
from cudnn.moe_dispatch_mlp_combine.cutedsl.persistent_bf16_fwd_contract import COMM_TILE_SHAPES
from .utils import BF16_TOLERANCE, check_correctness, generate_inputs, run_reference_bf16

TOKENS, LOCAL_EXPERTS, TOPK, INTERMEDIATE = 512, 2, 2, 512
NAMES = ("output", "d_x", "d_router_weights", "d_w_routed_gate", "d_w_routed_up", "d_w_routed_down", "d_w_shared_gate", "d_w_shared_up", "d_w_shared_down")


def _shape_keys(cache, shape):
    """The executor keys of ``cache`` compiled for the communication tile shape ``shape``."""
    return {key for key in cache if ("comm_tile_shape", (shape.rows, shape.columns)) in key}


@contextlib.contextmanager
def _filled_allocations(value):
    """Inside the block every fresh forward output and every fresh backward raw13 tensor (scratch rings and weight
    gradients) starts as ``value`` instead of whatever the caching allocator returns."""
    allocate_state, allocate_raw13 = persistent_bf16_fwd._allocate_state, persistent_bf16_bwd._allocate_raw13

    def filled_state(layout, device):
        outputs, counters = allocate_state(layout, device)
        for output in outputs:
            output.fill_(value)
        return outputs, counters

    def filled_raw13(*args, **kwargs):
        raw13 = allocate_raw13(*args, **kwargs)
        for tensor in raw13:
            tensor.fill_(value)
        return raw13

    with mock.patch.object(persistent_bf16_fwd, "_allocate_state", filled_state), mock.patch.object(persistent_bf16_bwd, "_allocate_raw13", filled_raw13):
        yield


def _run(shape, fill, config, workspace, schedule, x, probs, weights, dy):
    """One forward + backward with ``config.comm_tile_shape`` set to ``shape``; fresh copies of the output
    and the gradients.  ``fill`` goes into the buffers the peers push into (ahead of this rank's entry barriers
    on its stream; a peer pushes only after them) and into the call's fresh allocations."""
    workspace.combine_buffer.fill_(fill)
    workspace.d_x_routed_buffer.fill_(fill)
    config = dataclasses.replace(config, comm_tile_shape=(shape.rows, shape.columns))
    with _filled_allocations(fill):
        y, ctx = functional.forward(config, workspace, schedule, x, probs, *weights)
        grads = functional.backward(config, workspace, schedule, ctx, dy, x, probs, *weights)
        torch.cuda.synchronize(x.device)  # a device fault is reported against this shape
    return tuple(None if tensor is None else tensor.clone() for tensor in (y, *grads))  # None: no such weight


@pytest.mark.parametrize("hidden", [4096, 1152])
@pytest.mark.parametrize(
    "macrobatch_size,epilogue_combine",
    [
        (4096, False),  # one macrobatch: all pulls, then all pushes
        (512, False),  # several: pushes and pulls alternate, the backward replays X
        (512, True),  # the forward's stream moves its pulls only
    ],
)
def test_comm_tile_shapes_give_identical_bits(context, hidden, macrobatch_size, epilogue_combine):
    rank, world, device = context
    if torch.cuda.get_device_capability(device) != TARGET_CAPABILITY:
        pytest.skip(f"compiled for {TARGET_ARCH}")
    inputs = generate_inputs(rank, device, LOCAL_EXPERTS * world, LOCAL_EXPERTS, TOPK, TOKENS, hidden, INTERMEDIATE)
    x, routes, probs, *rest = inputs
    shared, routed, dy = rest[:3], rest[3:6], rest[6]
    config = functional.CuteMoKConfig(
        fwd_num_comm_sms=2,
        bwd_num_comm_sms=2,
        minibatch_size=256,
        macrobatch_size=macrobatch_size,
        schedule_capacity_multiplier=1.5,
        epilogue_combine=epilogue_combine,
    )
    workspace = functional.create_workspace(config, dist.group.WORLD, device=device, num_local_tokens=TOKENS, hidden_size=hidden, topk=TOPK)
    schedule = functional.build_schedule(workspace, config, routes, num_local_experts=LOCAL_EXPERTS)
    weights = (*shared, *routed)
    results = {shape: _run(shape, fill, config, workspace, schedule, x, probs, weights, dy) for shape, fill in zip(COMM_TILE_SHAPES, (0.5, -1.5))}
    for shape in COMM_TILE_SHAPES:  # each shape ran its own megakernels
        assert _shape_keys(persistent_bf16_fwd._EXECUTOR_CACHE, shape), f"no forward executor for {shape}"
        assert _shape_keys(persistent_bf16_bwd._EXECUTOR_CACHE, shape), f"no backward executor for {shape}"
    default, wide = (results[shape] for shape in COMM_TILE_SHAPES)
    for name, expected, actual in zip(NAMES, default, wide):
        assert (expected is None) == (actual is None), name
        if expected is None:
            continue
        assert expected.dtype == actual.dtype and expected.shape == actual.shape, name
        different = int((expected.contiguous().view(-1).view(torch.uint8) != actual.contiguous().view(-1).view(torch.uint8)).sum())
        assert different == 0, f"{name}: {different} bytes differ between the tile shapes {COMM_TILE_SHAPES}"
    golden = run_reference_bf16(x, routes, probs, *shared, *routed, dy)
    for name, reference, result in zip(NAMES, golden, wide):
        if result is None:
            continue
        check_correctness(name, reference, result, BF16_TOLERANCE, print_stats=rank == 0)
    dist.barrier()
