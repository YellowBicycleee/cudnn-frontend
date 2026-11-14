# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Routed-only layers use no shared storage; the zero-shared oracle is equivalent.

The C++ port's ``tests/test_routed_only.py``, on the CuTe DSL kernels: all three shared weights
are None (``HAS_SHARED=false``), so the forward runs no shared task and allocates no shared
buffer, its context holds no shared activation, the epilogues sum the routed terms alone, and the
backward returns None for the shared gradients.  The reference is the full BF16 MoE with the
shared weights zeroed, which is the same function.  Under fused accumulation two microbatches add
into the routed main_grads (the shared slots are None), over several macrobatches.

Run under torchrun on eight GPUs.
"""

import pytest
import torch
import torch.distributed as dist

from cudnn.moe_dispatch_mlp_combine import functional
from cudnn.moe_dispatch_mlp_combine.cutedsl._arch import TARGET_ARCH, TARGET_CAPABILITY
from cudnn.moe_dispatch_mlp_combine.ops import mxfp8_quantize
from .utils import BF16_TOLERANCE, MXFP8_TOLERANCE, check_correctness, generate_inputs, run_reference_bf16


@pytest.mark.parametrize(
    "precision,grad_dtype",
    [
        ("bf16", None),
        ("mxfp8", None),
        ("bf16", torch.float32),
        ("mxfp8", torch.float32),
        ("bf16", torch.bfloat16),
        ("mxfp8", torch.bfloat16),
    ],
)
def test_routed_only_reference_and_accumulation(context, precision, grad_dtype):
    rank, world, device = context
    if torch.cuda.get_device_capability(device) != TARGET_CAPABILITY:
        pytest.skip(f"compiled for {TARGET_ARCH}")
    inputs = generate_inputs(rank, device, 2 * world, 2, 2, 512, 512, 256)
    x, routes, probs, *rest = inputs
    shared, weights, dy = rest[:3], rest[3:6], rest[6]
    for weight in shared:
        weight.zero_()  # the reference's shared expert contributes nothing
    if precision == "bf16" and grad_dtype is None:
        routes[:, 0], routes[:, 1] = 0, 1  # Other EP ranks own only empty experts.
    routes[0].fill_(-1)
    probs[0].zero_()
    if precision == "mxfp8":
        gate, up, down = [tuple(mxfp8_quantize(w, True, True)) for w in weights]
        fwd_weights = (gate[:2], up[:2], down[:2])
        bwd_weights = (gate, up, down[2:])
    else:
        fwd_weights = bwd_weights = weights
    # Accumulating: a macrobatch below the routed rows, so the backward replays and every routed
    # main_grad takes several Wgrad contributions per call.
    config = functional.CuteMoKConfig(
        fwd_num_comm_sms=2, bwd_num_comm_sms=2, minibatch_size=256, macrobatch_size=512 if grad_dtype else 4096, schedule_capacity_multiplier=1.5
    )
    workspace = functional.create_workspace(config, dist.group.WORLD, device=device, num_local_tokens=512, hidden_size=512, topk=2)
    schedule = functional.build_schedule(workspace, config, routes, num_local_experts=2)
    if grad_dtype:
        assert schedule.num_tokens.item() > config.macrobatch_size
    buffers = [torch.zeros_like(w, dtype=grad_dtype) for w in weights] if grad_dtype else None
    main_grads = (None, buffers[0], None, buffers[1], None, buffers[2]) if buffers else None
    expected = [torch.zeros_like(w, dtype=grad_dtype) for w in weights] if grad_dtype else None
    tolerance = MXFP8_TOLERANCE if precision == "mxfp8" else BF16_TOLERANCE
    for step in range(2 if grad_dtype else 1):
        xx, ddy = (x * -0.5, dy * 2) if step == 1 else (x, dy)
        y, ctx = functional.forward(config, workspace, schedule, xx, probs, None, None, None, *fwd_weights)
        assert ctx.gate_shared is ctx.up_shared is ctx.hidden_shared is None
        assert ctx.shared_output is ctx.shared_output_gate is None
        grads = functional.backward(
            config,
            workspace,
            schedule,
            ctx,
            ddy,
            xx,
            probs,
            None,
            None,
            None,
            *bwd_weights,
            main_grads=main_grads,
        )
        assert len(grads) == 9 and all(g is None for g in grads[5:])
        golden = list(run_reference_bf16(xx, routes, probs, *shared, *weights, ddy)[:6])
        if grad_dtype:
            for actual, buffer, total, contribution in zip(grads[2:5], buffers, expected, golden[3:]):
                assert actual is buffer
                total.add_(contribution)
            golden[3:] = expected
        for name, actual, reference in zip(("Y", "dX", "dP", "dWa", "dWu", "dWd"), (y, *grads[:5]), golden):
            check_correctness(name, reference, actual, tolerance, print_stats=rank == 0)
        # Token 0 routes nowhere: with no shared expert its output and input gradient are zero.
        torch.testing.assert_close(y[0], torch.zeros_like(y[0]), rtol=0, atol=0)
        torch.testing.assert_close(grads[0][0], torch.zeros_like(grads[0][0]), rtol=0, atol=0)


def test_routed_only_presence_is_all_or_nothing(context):
    """The C++ port's pairing rules: the three shared weights are given together or not at all,
    the shared output gate needs a shared expert, and the backward's shared weights match the
    forward context's."""
    rank, world, device = context
    if torch.cuda.get_device_capability(device) != TARGET_CAPABILITY:
        pytest.skip(f"compiled for {TARGET_ARCH}")
    x, routes, probs, *rest = generate_inputs(rank, device, 2 * world, 2, 2, 512, 512, 256)
    shared, weights, dy = rest[:3], rest[3:6], rest[6]
    config = functional.CuteMoKConfig(fwd_num_comm_sms=2, bwd_num_comm_sms=2, minibatch_size=256, macrobatch_size=4096, schedule_capacity_multiplier=1.5)
    workspace = functional.create_workspace(config, dist.group.WORLD, device=device, num_local_tokens=512, hidden_size=512, topk=2)
    schedule = functional.build_schedule(workspace, config, routes, num_local_experts=2)
    with pytest.raises(ValueError, match="all present or all absent"):
        functional.forward(config, workspace, schedule, x, probs, shared[0], None, None, *weights)
    gate_weight = torch.zeros(1, 512, dtype=torch.bfloat16, device=device)
    with pytest.raises(ValueError, match="shared output gate requires shared"):
        functional.forward(config, workspace, schedule, x, probs, None, None, None, *weights, shared_output_gate_weight=gate_weight)
    _, ctx = functional.forward(config, workspace, schedule, x, probs, None, None, None, *weights)
    with pytest.raises(ValueError, match="presence must match the forward context"):
        functional.backward(config, workspace, schedule, ctx, dy, x, probs, *shared, *weights)
    _, ctx = functional.forward(config, workspace, schedule, x, probs, *shared, *weights)
    with pytest.raises(ValueError, match="presence must match the forward context"):
        functional.backward(config, workspace, schedule, ctx, dy, x, probs, None, None, None, *weights)
    dist.barrier()
