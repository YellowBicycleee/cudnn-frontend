# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Weight gradients accumulated into the caller's main_grad buffers (MCore's gradient_accumulation_fusion).

Run under torchrun on SM100 / SM103.
"""

import os

import pytest
import torch
import torch.distributed as dist

from cudnn.moe_ep.cutemok import functional
from cudnn.moe_ep.cutemok.cutedsl._arch import SUPPORTED_CAPABILITIES, TARGET_ARCH, TARGET_CAPABILITY
from cudnn.moe_ep.cutemok.functional import get_workspace
from cudnn.moe_ep.cutemok.ops import mxfp8_quantize

from .utils import (
    BF16_TOLERANCE,
    MXFP8_TOLERANCE,
    capacity_multiplier,
    check_correctness,
    generate_inputs,
    run_reference_bf16,
    ungated,
)

# --------------------------------------------------------------------------------------------------------------------
# Fused main_grad accumulation (BF16)
#
# The weight gradients written straight into MCore's main_grad buffers.
#
# MCore trains with ``gradient_accumulation_fusion``: a weight's gradient belongs in the distributed
# optimizer's ``main_grad``, which already holds earlier microbatches' contributions.  The CUDA port
# writes there from the epilogue (``ACCUMULATE_WGRAD``), keeping the Tensor Core accumulator in FP32
# all the way to memory unless the buffer is itself bfloat16 (``BF16_MAIN_GRAD``).  This is that
# path in the CuTe DSL backward.
#
# What the tests pin down:
#
# * **The FP32 destination is more precise, not merely different.**  The unfused path rounds each
#   gradient to bfloat16 before the host adds it; the fused one never does.  On a single-macrobatch
#   geometry there is exactly one contribution per tile, so the only difference left is that final
#   rounding -- which makes ``fused.to(bfloat16)`` equal to the unfused gradient *bit for bit*.  An
#   ordinary tolerance check would pass just as happily on a kernel that had quietly lost a segment.
# * **Empty experts contribute nothing.**  The unfused path zero-fills an expert with no rows; the
#   fused path must leave main_grad alone, because writing zeros would erase the earlier
#   microbatches.  Starting from a buffer that is *not* zero is what tells the two apart.
# * **Replay is bit-identical**, which MCore's determinism test requires of the whole layer.
#
# Run under torchrun on SM100 / SM103, e.g.
# ``torchrun --standalone --nproc-per-node=4 -m pytest -s moe_ep/cutemok/test_main_grad.py``.


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


# --------------------------------------------------------------------------------------------------------------------
# Accumulation against the reference (BF16 and MXFP8)
#
# Fused main_grad accumulation against the BF16 reference (the C++ port's
# tests/test_bf16_wgrad_accum.py, d59188a, unchanged but for the architecture skip).
#
# Both precisions; separate Gate/Up weights, one combined FC1 parameter (MCore's single grouped
# weight, so the Up half sits I rows into both the weight and its main_grad), and the latter with
# TransformerEngine's native columnwise MXFP8 payload; FP32 and BF16 main_grads; with and without an
# expert that gets no rows; several microbatches accumulating into the same buffers over several
# macrobatches.


@pytest.mark.parametrize("leave_one_expert_empty", [False, True])
@pytest.mark.parametrize("main_grad_dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("precision", ["bf16", "mxfp8"])
@pytest.mark.parametrize(
    "routed_weight_storage",
    ["separate_legacy", "single_grouped_legacy", "single_grouped_native"],
)
def test_wgrad_accumulates_directly_into_main_grad(
    context: tuple[int, int, torch.device],
    leave_one_expert_empty: bool,
    main_grad_dtype: torch.dtype,
    precision: str,
    routed_weight_storage: str,
) -> None:
    rank, world_size, device = context
    if torch.cuda.get_device_capability(device) != TARGET_CAPABILITY:
        pytest.skip(f"compiled for {TARGET_ARCH}")
    single_grouped = routed_weight_storage != "separate_legacy"
    native_columnwise = routed_weight_storage == "single_grouped_native"
    # Exercise both the gate/up offset within FC1 and the expert stride between
    # adjacent slices in one single-grouped parameter.
    num_local_experts = 2
    num_experts = world_size * num_local_experts
    hidden_dim = 256
    intermediate_dim = 256
    topk = 1
    num_local_tokens = 512
    config = functional.CuteMoKConfig(
        fwd_num_comm_sms=2,
        bwd_num_comm_sms=2,
        minibatch_size=256,
        macrobatch_size=int(os.getenv("MOK_TEST_MACROBATCH_SIZE", "256")),
        schedule_capacity_multiplier=1.5,
    )
    workspace = functional.get_workspace(
        config,
        torch.distributed.group.WORLD,
        device=device,
        num_local_tokens=num_local_tokens,
        hidden_size=hidden_dim,
        topk=topk,
    )
    inputs = generate_inputs(
        rank,
        device,
        num_experts,
        num_local_experts,
        topk,
        num_local_tokens,
        hidden_dim,
        intermediate_dim,
    )
    (
        x,
        topk_experts,
        router_weights,
        w_shared_gate,
        w_shared_up,
        w_shared_down,
        w_routed_gate,
        w_routed_up,
        w_routed_down,
        d_output,
    ) = inputs
    if precision == "mxfp8":
        if single_grouped:
            routed_fc1 = torch.cat((w_routed_gate, w_routed_up), dim=1)
            routed_fc1_quantized = mxfp8_quantize(routed_fc1, True, True)
            routed_down_quantized = mxfp8_quantize(w_routed_down, True, True)
            forward_routed_gate = routed_fc1_quantized[:2]
            forward_routed_up = forward_routed_gate
            forward_routed_down = routed_down_quantized[:2]
            if native_columnwise:
                native_fc1_columnwise = routed_fc1_quantized[2].transpose(-2, -1).contiguous()
                backward_routed_gate = (
                    routed_fc1_quantized[0],
                    routed_fc1_quantized[1],
                    native_fc1_columnwise,
                    routed_fc1_quantized[3],
                    True,
                )
            else:
                backward_routed_gate = routed_fc1_quantized
            backward_routed_up = backward_routed_gate
            if native_columnwise:
                backward_routed_down = (
                    routed_down_quantized[2].transpose(-2, -1).contiguous(),
                    routed_down_quantized[3],
                    True,
                )
            else:
                backward_routed_down = routed_down_quantized[2:]
        else:
            routed_gate_quantized = mxfp8_quantize(w_routed_gate, True, True)
            routed_up_quantized = mxfp8_quantize(w_routed_up, True, True)
            routed_down_quantized = mxfp8_quantize(w_routed_down, True, True)
            forward_routed_gate = routed_gate_quantized[:2]
            forward_routed_up = routed_up_quantized[:2]
            forward_routed_down = routed_down_quantized[:2]
            backward_routed_gate = routed_gate_quantized
            backward_routed_up = routed_up_quantized
            backward_routed_down = routed_down_quantized[2:]
        tolerance = MXFP8_TOLERANCE
    else:
        if single_grouped:
            routed_fc1 = torch.cat((w_routed_gate, w_routed_up), dim=1)
            forward_routed_gate = routed_fc1
            forward_routed_up = routed_fc1
        else:
            forward_routed_gate = w_routed_gate
            forward_routed_up = w_routed_up
        forward_routed_down = w_routed_down
        backward_routed_gate = forward_routed_gate
        backward_routed_up = forward_routed_up
        backward_routed_down = w_routed_down
        tolerance = BF16_TOLERANCE
    if leave_one_expert_empty:
        topk_experts[:, 0] = (torch.arange(num_local_tokens, device=device) + rank) % (num_experts - 1)
    schedule = functional.build_schedule(
        workspace,
        config,
        topk_experts,
        num_local_experts=num_local_experts,
    )

    if single_grouped:
        routed_fc1_main_grad = torch.full_like(routed_fc1, 0.25, dtype=main_grad_dtype)
        main_grads = (
            torch.full_like(w_shared_gate, 0.25, dtype=main_grad_dtype),
            routed_fc1_main_grad,
            torch.full_like(w_shared_up, 0.25, dtype=main_grad_dtype),
            routed_fc1_main_grad,
            torch.full_like(w_shared_down, 0.25, dtype=main_grad_dtype),
            torch.full_like(w_routed_down, 0.25, dtype=main_grad_dtype),
        )
    else:
        main_grads = tuple(
            torch.full_like(weight, 0.25, dtype=main_grad_dtype)
            for weight in (
                w_shared_gate,
                w_routed_gate,
                w_shared_up,
                w_routed_up,
                w_shared_down,
                w_routed_down,
            )
        )
    # MCore calls the fused backward once per microbatch and expects every call
    # to add into the same main_grad buffers. Exercise that lifecycle,
    # rather than validating only one add into a pre-filled tensor.
    # Production recipes may accumulate hundreds of microbatches. Keep regular
    # CI fast, while allowing focused runs to stress the real accumulation depth.
    num_accumulations = int(os.getenv("MOK_TEST_NUM_ACCUMULATIONS", "3"))
    if num_accumulations <= 0:
        raise ValueError("MOK_TEST_NUM_ACCUMULATIONS must be positive")

    for _ in range(num_accumulations):
        output, forward_context = functional.forward(
            config,
            workspace,
            schedule,
            x,
            router_weights,
            w_shared_gate,
            w_shared_up,
            w_shared_down,
            forward_routed_gate,
            forward_routed_up,
            forward_routed_down,
        )
        d_x, d_router_weights, *weight_gradients = functional.backward(
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
            backward_routed_gate,
            backward_routed_up,
            backward_routed_down,
            main_grads=main_grads,
        )
        assert len(weight_gradients) == 7 and weight_gradients[-1] is None

    reference_all = run_reference_bf16(*inputs)
    check_correctness("output", reference_all[0], output, tolerance, rank == 0)
    reference = reference_all[1:]
    check_correctness("d_x", reference[0], d_x, tolerance, rank == 0)
    check_correctness("d_router_weights", reference[1], d_router_weights, tolerance, rank == 0)
    if single_grouped:
        reference_wgrads = (
            reference[5],
            torch.cat((reference[2], reference[3]), dim=1),
            reference[6],
            reference[7],
            reference[4],
        )
        actual_main_grads = (
            main_grads[0],
            main_grads[1],
            main_grads[2],
            main_grads[4],
            main_grads[5],
        )
        weight_names = (
            "shared_gate",
            "routed_fc1",
            "shared_up",
            "shared_down",
            "routed_down",
        )
    else:
        reference_wgrads = (
            reference[5],
            reference[2],
            reference[6],
            reference[3],
            reference[7],
            reference[4],
        )
        actual_main_grads = main_grads
        weight_names = (
            "shared_gate",
            "routed_gate",
            "shared_up",
            "routed_up",
            "shared_down",
            "routed_down",
        )
    for name, expected, actual in zip(
        weight_names,
        reference_wgrads,
        actual_main_grads,
        strict=True,
    ):
        expected_accum = torch.full_like(expected, 0.25, dtype=main_grad_dtype)
        for _ in range(num_accumulations):
            expected_accum.add_(expected.to(main_grad_dtype))
        check_correctness(
            f"main_grad/{name}",
            expected_accum,
            actual,
            (
                tolerance[0] * num_accumulations * (2 if num_accumulations > 3 else 1),
                tolerance[1] * (2 if num_accumulations > 3 else 1),
            ),
            rank == 0,
        )


# --------------------------------------------------------------------------------------------------------------------
# Packed FC1 main_grad buffers and the optimizer norm


def test_canonical_fc12_wgrad_accum(context: tuple[int, int, torch.device]) -> None:
    """Accumulate into packed shared/routed FC1 buffers through the six-weight API, then run optimizer norm."""
    from transformer_engine.pytorch.optimizers import multi_tensor_applier, multi_tensor_l2norm

    rank, world_size, device = context
    num_local_experts = int(os.getenv("MOK_TEST_NUM_LOCAL_EXPERTS", "2"))
    hidden_dim = int(os.getenv("MOK_TEST_HIDDEN_DIM", "256"))
    intermediate_dim = int(os.getenv("MOK_TEST_INTERMEDIATE_DIM", "256"))
    topk = int(os.getenv("MOK_TEST_TOPK", "1"))
    num_local_tokens = int(os.getenv("MOK_TEST_NUM_LOCAL_TOKENS", "512"))
    config = functional.CuteMoKConfig(
        fwd_num_comm_sms=int(os.getenv("MOK_TEST_FWD_NUM_COMM_SMS", "2")),
        bwd_num_comm_sms=int(os.getenv("MOK_TEST_BWD_NUM_COMM_SMS", "2")),
        minibatch_size=int(os.getenv("MOK_TEST_MINIBATCH_SIZE", "256")),
        macrobatch_size=int(os.getenv("MOK_TEST_MACROBATCH_SIZE", "256")),
        schedule_capacity_multiplier=1.5,
    )
    workspace = functional.get_workspace(
        config,
        torch.distributed.group.WORLD,
        device=device,
        num_local_tokens=num_local_tokens,
        hidden_size=hidden_dim,
        topk=topk,
    )
    inputs = generate_inputs(
        rank,
        device,
        world_size * num_local_experts,
        num_local_experts,
        topk,
        num_local_tokens,
        hidden_dim,
        intermediate_dim,
    )
    x, top_experts, router_weights, shared_gate, shared_up, shared_fc2, routed_gate, routed_up, routed_fc2, grad_output = inputs
    shared_fc1 = torch.cat((shared_gate, shared_up), dim=0)
    routed_fc1 = torch.cat((routed_gate, routed_up), dim=1)
    shared_gate_view = shared_fc1[:intermediate_dim]
    shared_up_view = shared_fc1[intermediate_dim:]
    assert shared_gate_view.is_contiguous() and shared_up_view.is_contiguous()
    schedule = functional.build_schedule(workspace, config, top_experts, num_local_experts=num_local_experts)
    main_grads = tuple(torch.full_like(weight, 0.25, dtype=torch.bfloat16) for weight in (shared_fc1, routed_fc1, shared_fc2, routed_fc2))
    # Shared weights use contiguous gate/up views. Routed gate/up share the whole FC1 tensor and use kernel offsets.
    weights = (shared_gate_view, shared_up_view, shared_fc2, routed_fc1, routed_fc1, routed_fc2)
    shared_gate_grad, shared_up_grad = main_grads[0].chunk(2, dim=0)
    abi_main_grads = (shared_gate_grad, main_grads[1], shared_up_grad, main_grads[1], main_grads[2], main_grads[3])
    assert all(gradient.is_contiguous() for gradient in abi_main_grads)
    output, forward_context = functional.forward(config, workspace, schedule, x, router_weights, *weights)
    backward_outputs = functional.backward(
        config,
        workspace,
        schedule,
        forward_context,
        grad_output,
        x,
        router_weights,
        *weights,
        main_grads=abi_main_grads,
    )

    assert len(backward_outputs) == 9 and backward_outputs[-1] is None
    d_x, d_router_weights = backward_outputs[:2]
    for actual, expected_alias in zip(
        backward_outputs[2:8],
        (abi_main_grads[1], abi_main_grads[3], abi_main_grads[5], abi_main_grads[0], abi_main_grads[2], abi_main_grads[4]),
        strict=True,
    ):
        assert actual is expected_alias

    # Match the optimizer operation that exposed the end-to-end regression; each packed allocation appears once.
    noop = torch.zeros(1, dtype=torch.int32, device=device)
    multi_tensor_applier(multi_tensor_l2norm, noop, [list(main_grads)], False)
    torch.cuda.synchronize()

    reference_all = run_reference_bf16(*inputs)
    reference = reference_all[1:]
    for name, expected, actual in zip(
        ("output", "d_x", "d_router_weights"),
        reference_all[:3],
        (output, d_x, d_router_weights),
        strict=True,
    ):
        check_correctness(name, expected, actual, BF16_TOLERANCE, rank == 0)
    expected_grads = (
        torch.cat((reference[5], reference[6]), dim=0),
        torch.cat((reference[2], reference[3]), dim=1),
        reference[7],
        reference[4],
    )
    for name, expected, actual in zip(
        ("shared_fc1", "routed_fc1", "shared_fc2", "routed_fc2"),
        expected_grads,
        main_grads,
        strict=True,
    ):
        check_correctness(
            f"main_grad/{name}",
            expected.to(torch.bfloat16).add(0.25),
            actual,
            BF16_TOLERANCE,
            rank == 0,
        )
