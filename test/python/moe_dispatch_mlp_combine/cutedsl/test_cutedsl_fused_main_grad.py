# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The weight gradients written straight into MCore's main_grad buffers.

MCore trains with ``gradient_accumulation_fusion``: a weight's gradient belongs in the distributed
optimizer's ``main_grad``, which already holds earlier microbatches' contributions.  The CUDA port
writes there from the epilogue (``ACCUMULATE_WGRAD``), keeping the Tensor Core accumulator in FP32
all the way to memory unless the buffer is itself bfloat16 (``BF16_MAIN_GRAD``).  This is that
path in the CuTe DSL backward.

What the tests pin down:

* **The FP32 destination is more precise, not merely different.**  The unfused path rounds each
  gradient to bfloat16 before the host adds it; the fused one never does.  On a single-macrobatch
  geometry there is exactly one contribution per tile, so the only difference left is that final
  rounding -- which makes ``fused.to(bfloat16)`` equal to the unfused gradient *bit for bit*.  An
  ordinary tolerance check would pass just as happily on a kernel that had quietly lost a segment.
* **Empty experts contribute nothing.**  The unfused path zero-fills an expert with no rows; the
  fused path must leave main_grad alone, because writing zeros would erase the earlier
  microbatches.  Starting from a buffer that is *not* zero is what tells the two apart.
* **Replay is bit-identical**, which MCore's determinism test requires of the whole layer.

Run under torchrun on SM100 / SM103, e.g.
``torchrun --standalone --nproc-per-node=4 -m pytest -s moe_dispatch_mlp_combine/cutedsl/test_cutedsl_fused_main_grad.py``.
"""

import pytest  # noqa: F401  (run under torchrun + pytest, see module docstring)
import torch
import torch.distributed as dist

from cudnn.moe_dispatch_mlp_combine import functional
from cudnn.moe_dispatch_mlp_combine.functional import get_workspace
from cudnn.moe_dispatch_mlp_combine.cutedsl._arch import SUPPORTED_CAPABILITIES

from .utils import ungated, capacity_multiplier, generate_inputs

SUPPORTED_WORLD_SIZES = (1, 4, 8, 16, 32, 64)

#: Small, and one macrobatch: with a single contribution per dW tile the fused and unfused sums
#: differ only by the unfused path's bfloat16 rounding, which is what makes the comparison exact.
LOCAL_EXPERTS, HIDDEN, INTERMEDIATE, TOPK, LOCAL_TOKENS = 4, 256, 256, 4, 512
MACROBATCH = 131072

#: The six main_grad buffers in MOK's ABI order.
NAMES = functional.MAIN_GRAD_NAMES


def _setup(context, empty_first_local_expert: bool = False):
    rank, world_size, device = context
    if world_size not in SUPPORTED_WORLD_SIZES:
        pytest.skip(f"a workspace supports EP sizes {SUPPORTED_WORLD_SIZES}")
    if torch.cuda.get_device_capability(device) not in SUPPORTED_CAPABILITIES:
        pytest.skip("CuTe DSL backends are SM100/SM103 only")

    num_experts = LOCAL_EXPERTS * world_size
    config = functional.CuteMoKConfig(
        macrobatch_size=MACROBATCH,
        schedule_capacity_multiplier=capacity_multiplier(world_size, LOCAL_TOKENS, TOPK, LOCAL_EXPERTS),
    )
    workspace = get_workspace(config, dist.group.WORLD, device=device, num_local_tokens=LOCAL_TOKENS, hidden_size=HIDDEN, topk=TOPK)
    inputs = generate_inputs(rank, device, num_experts, LOCAL_EXPERTS, TOPK, LOCAL_TOKENS, HIDDEN, INTERMEDIATE)
    if empty_first_local_expert:
        # Route away from every rank's local expert 0, so that one has no rows anywhere.  The
        # routes are global expert ids, and local expert e of rank r is r * LOCAL_EXPERTS + e, so
        # moving the multiples of LOCAL_EXPERTS one along empties local expert 0 on every rank at
        # once -- which keeps the assertion below uniform across ranks.
        routes = inputs[1]
        routes = torch.where(routes % LOCAL_EXPERTS == 0, routes + 1, routes)
        inputs = (inputs[0], routes, *inputs[2:])
    schedule = functional.build_schedule(workspace, config, inputs[1], num_local_experts=LOCAL_EXPERTS)
    return config, workspace, schedule, inputs


def _weights(inputs):
    shared_gate, shared_up, shared_down = inputs[3:6]
    return (shared_gate, shared_up, shared_down, *inputs[6:9])


def _run(config, workspace, schedule, inputs, main_grads=None):
    """One forward and backward; returns the six weight gradients the backward reports."""
    x, _, router_weights = inputs[:3]
    d_output = inputs[9]
    weights = _weights(inputs)
    _, forward_context = functional.forward(config, workspace, schedule, x, router_weights, *weights, None)
    gradients = ungated(functional.backward(config, workspace, schedule, forward_context, d_output, x, router_weights, *weights, None, main_grads=main_grads))
    # (d_x, d_router_weights, d_w_routed_gate, d_w_routed_up, d_w_routed_down,
    #  d_w_shared_gate, d_w_shared_up, d_w_shared_down) -> MAIN_GRAD_NAMES order.
    _, _, routed_gate, routed_up, routed_down, shared_gate, shared_up, shared_down = gradients
    return (shared_gate, routed_gate, shared_up, routed_up, shared_down, routed_down)


def _main_grad_buffers(inputs, dtype, fill=0.0):
    """Six buffers shaped like the gradients, in MAIN_GRAD_NAMES order."""
    shared_gate, shared_up, shared_down, gate, up, down = _weights(inputs)
    shapes = (shared_gate.shape, gate.shape, shared_up.shape, up.shape, shared_down.shape, down.shape)
    return tuple(torch.full(shape, fill, dtype=dtype, device=shared_gate.device) for shape in shapes)


def _assert_equal(expected, actual, what: str) -> None:
    for name, left, right in zip(NAMES, expected, actual, strict=True):
        if torch.equal(left, right):
            continue
        differing = (left != right).sum().item()
        raise AssertionError(
            f"{what}: {name} differs in {differing} of {left.numel()} elements " f"(max |delta| {(left.float() - right.float()).abs().max().item():g})"
        )


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_fused_main_grad_matches_the_unfused_sum(context, dtype):
    """Into a zeroed buffer, the fused epilogue must reproduce the unfused gradient exactly.

    FP32: the fused path keeps the accumulator's precision, so the comparison is against its
    bfloat16 rounding -- the one step the unfused path takes before the host adds.
    BF16: both round the same way, so the buffers themselves match.
    """
    config, workspace, schedule, inputs = _setup(context)
    unfused = _run(config, workspace, schedule, inputs)
    buffers = _main_grad_buffers(inputs, dtype)
    fused = _run(config, workspace, schedule, inputs, main_grads=buffers)
    for name, buffer, result in zip(NAMES, buffers, fused, strict=True):
        assert result is buffer, f"{name}: the backward must report the caller's buffer"
    _assert_equal(unfused, tuple(buffer.to(torch.bfloat16) for buffer in buffers), f"fused {dtype} main_grad")


def test_fused_main_grad_accumulates_rather_than_overwrites(context):
    """Two identical microbatches into one buffer give exactly twice one.

    Both checks here are exact on purpose.  Doubling is: adding a float to itself is exact in
    binary floating point, so an epilogue that overwrote would give one times the gradient, not
    two.  And a position no tile wrote must still hold its starting value -- which is what says
    an expert with no rows is left alone rather than zero-filled, since zeroing it would erase the
    earlier microbatches that main_grad already holds.
    """
    config, workspace, schedule, inputs = _setup(context)
    once = _main_grad_buffers(inputs, torch.float32)
    _run(config, workspace, schedule, inputs, main_grads=once)
    after_one = tuple(buffer.clone() for buffer in once)
    _run(config, workspace, schedule, inputs, main_grads=once)
    _assert_equal(tuple(buffer * 2.0 for buffer in after_one), once, "two microbatches")


def test_fused_main_grad_leaves_an_empty_expert_alone(context):
    """An expert with no rows must not have its main_grad touched.

    The unfused path writes that expert's gradient as zeros -- it owns the buffer, so zeroing is
    right.  Accumulating, the buffer is MCore's and already holds earlier microbatches, so zeroing
    would erase them; the CUDA port skips ``zero_empty_routed_wgrads`` for exactly this reason.
    The check is exact because an untouched buffer is untouched.
    """
    config, workspace, schedule, inputs = _setup(context, empty_first_local_expert=True)
    assert int(schedule.tokens_per_expert[0]) == 0, "local expert 0 was meant to have no rows"
    seeded = _main_grad_buffers(inputs, torch.float32, fill=1.5)
    _run(config, workspace, schedule, inputs, main_grads=seeded)
    for name, buffer in zip(NAMES, seeded, strict=True):
        if not name.startswith("routed_"):
            continue
        empty = buffer[0]
        assert torch.equal(empty, torch.full_like(empty, 1.5)), (
            f"{name}: the empty expert's gradient was written, which would erase the earlier " "microbatches main_grad already holds"
        )


def test_fused_main_grad_replays(context):
    """The same inputs into the same starting buffer give the same bytes."""
    config, workspace, schedule, inputs = _setup(context)
    first = _main_grad_buffers(inputs, torch.float32)
    second = _main_grad_buffers(inputs, torch.float32)
    _run(config, workspace, schedule, inputs, main_grads=first)
    _run(config, workspace, schedule, inputs, main_grads=second)
    _assert_equal(first, second, "a replayed fused backward")
