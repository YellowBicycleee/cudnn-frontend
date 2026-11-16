# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The shared expert: routed-only layers, the shared output gate, SiTU-GLU, the gate-weight checks and
``zero_unrouted_router_grads``.

Run under torchrun on SM100 / SM103.
"""

import pytest
import torch
import torch.distributed as dist

from cudnn.moe_ep.cutemok import functional, ops
from cudnn.moe_ep.cutemok.cutedsl._arch import TARGET_ARCH, TARGET_CAPABILITY
from cudnn.moe_ep.cutemok.ops import mxfp8_quantize

from .utils import (
    BF16_TOLERANCE,
    MXFP8_TOLERANCE,
    check_correctness,
    generate_inputs,
    run_reference_bf16,
    run_shared_output_gate_reference,
    run_swiglu_reference,
)

# --------------------------------------------------------------------------------------------------------------------
# Routed-only layers
#
# Routed-only layers use no shared storage; the zero-shared oracle is equivalent.
#
# The C++ port's ``tests/test_routed_only.py``, on the CuTe DSL kernels: all three shared weights
# are None (``HAS_SHARED=false``), so the forward runs no shared task and allocates no shared
# buffer, its context holds no shared activation, the epilogues sum the routed terms alone, and the
# backward returns None for the shared gradients.  The reference is the full BF16 MoE with the
# shared weights zeroed, which is the same function.  Under fused accumulation two microbatches add
# into the routed main_grads (the shared slots are None), over several macrobatches.
#
# Run under torchrun on eight GPUs.


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


# --------------------------------------------------------------------------------------------------------------------
# Shared output gate and SiTU-GLU
#
# Shared output-gate and SiTU-GLU correctness through the torchrun/pytest CUDA fixture.
#
# The C++ port's tests/test_shared_output_gate_reference.py (d59188a / ee7bdab), on the CuTe DSL
# kernels.  As there, the ungated SiTU case discards the forward's context and backpropagates through
# ``functional.recompute_forward_context``.


GATED_RESULT_NAMES = (
    "output",
    "d_x",
    "d_router_weights",
    "d_w_routed_gate",
    "d_w_routed_up",
    "d_w_routed_down",
    "d_w_shared_gate",
    "d_w_shared_up",
    "d_w_shared_down",
    "d_w_shared_output_gate",
)


def _dense_inputs(context):
    rank, world_size, device = context
    if torch.cuda.get_device_capability(device) != TARGET_CAPABILITY:
        pytest.skip(f"compiled for {TARGET_ARCH}")
    inputs = generate_inputs(rank, device, 2 * world_size, 2, 2, 512, 256, 256)
    generator = torch.Generator(device=device).manual_seed(919 + rank)
    weight = torch.randn(1, 256, generator=generator, device=device, dtype=torch.bfloat16) / 16
    return inputs, weight


def _routed_weight_args(weights, precision):
    if precision == "bf16":
        return weights, weights
    gate, up, down = [tuple(ops.mxfp8_quantize(weight, True, True)) for weight in weights]
    return (gate[:2], up[:2], down[:2]), (gate, up, down[2:])


def _bf16_product_rounding_witness(device, rows, hidden):
    # Both products are near +/-1. BF16 rounds the positive product down,
    # yielding exact cancellation. FP32 retains its 1/16384 residual.
    x = torch.zeros(rows, hidden, device=device, dtype=torch.bfloat16)
    x[:, :2] = 1
    shared = torch.zeros_like(x)
    shared[:, 0] = 1 + 1 / 128
    shared[:, 1] = 1
    dy = torch.zeros_like(x)
    dy[:, 0] = 1 + 1 / 128
    dy[:, 1] = -(1 + 1 / 64)
    weight = torch.zeros(1, hidden, device=device, dtype=torch.bfloat16)
    weight[0, 0], weight[0, 1] = 1, -1  # Z=0 and G=0.5, but Wg is nonzero.
    return x, shared, weight, dy


@pytest.mark.parametrize("main_grad_dtype", [torch.float32, torch.bfloat16])
def test_gate_wgrad_fp32_contribution_and_accumulation(context, main_grad_dtype):
    rows, hidden = 32, 64
    x = torch.ones(rows, hidden, device=context[2], dtype=torch.bfloat16)
    x[-1].fill_(1 / 64)
    main_grad = torch.full((1, hidden), -rows, device=context[2], dtype=main_grad_dtype)
    for scale in (1, 1 / rows):
        dz = torch.full((rows, 1), scale, device=context[2], dtype=torch.bfloat16)
        # The exact dot is representable in FP32, but not BF16.
        contribution = torch.full_like(main_grad, (rows - 1 + 1 / 64) * scale, dtype=torch.float32)
        assert not torch.equal(contribution, contribution.bfloat16().float())
        fresh = functional._shared_output_gate_wgrad(dz, x)
        assert fresh.dtype == torch.float32
        torch.testing.assert_close(fresh, contribution, rtol=0, atol=0)
        expected = (main_grad.float() + contribution).to(main_grad_dtype)
        if main_grad_dtype == torch.bfloat16:
            pre_rounded = (main_grad.float() + contribution.bfloat16().float()).bfloat16()
            assert not torch.equal(expected, pre_rounded)
        returned = functional._shared_output_gate_wgrad(dz, x, main_grad)
        assert returned is main_grad
        torch.testing.assert_close(returned, expected, rtol=0, atol=0)
    main_grad.zero_()
    returned = functional._shared_output_gate_wgrad(dz, x, main_grad)
    assert returned is main_grad
    torch.testing.assert_close(returned, contribution.to(main_grad_dtype), rtol=0, atol=0)


def _mok_schedule(context, inputs, macrobatch_size=4096):
    _, _, device = context
    x, experts, probs, *_ = inputs
    config = functional.CuteMoKConfig(
        fwd_num_comm_sms=2,
        bwd_num_comm_sms=2,
        minibatch_size=256,
        macrobatch_size=macrobatch_size,
        schedule_capacity_multiplier=1.5,
    )
    workspace = functional.get_workspace(
        config,
        dist.group.WORLD,
        device=device,
        num_local_tokens=x.shape[0],
        hidden_size=x.shape[1],
        topk=probs.shape[1],
    )
    schedule = functional.build_schedule(workspace, config, experts, num_local_experts=2)
    assert (schedule.num_tokens.item() > macrobatch_size) == (macrobatch_size == 512)
    return config, workspace, schedule


def test_mok_gate_backward_uses_bf16_product_rounding(context):
    """Exact cancellation distinguishes BF16-product from FP32-product dG."""
    inputs, _ = _dense_inputs(context)
    x, experts, probs, *rest = inputs
    weights = rest[:-1]
    x, shared, weight, dy = _bf16_product_rounding_witness(context[2], x.shape[0], x.shape[1])
    for matrix in weights:
        matrix.zero_()
    # Only one shared intermediate channel is active. BF16(silu(16))=16,
    # so the down GEMM produces our exact S; routed experts remain zero.
    weights[0][0, 0] = 16
    weights[1][0, 0] = 1
    weights[2][0, 0] = (1 + 1 / 128) / 16
    weights[2][1, 0] = 1 / 16
    inputs = (x, experts, probs, *weights, dy)
    config, workspace, schedule = _mok_schedule(context, inputs)
    output, saved = functional.forward(
        config,
        workspace,
        schedule,
        x,
        probs,
        *weights,
        shared_output_gate_weight=weight,
    )
    torch.testing.assert_close(saved.shared_output, shared, rtol=0, atol=0)
    torch.testing.assert_close(output, shared * 0.5, rtol=0, atol=0)
    _, _, _, _, _, golden_dw = run_shared_output_gate_reference(x, shared, weight, dy)
    gradients = functional.backward(
        config,
        workspace,
        schedule,
        saved,
        dy,
        x,
        probs,
        *weights,
        shared_output_gate_weight=weight,
    )
    assert len(gradients) == 9 and gradients[-1].dtype == torch.float32
    torch.testing.assert_close(golden_dw, torch.zeros_like(golden_dw), rtol=0, atol=0)
    torch.testing.assert_close(gradients[-1].double(), golden_dw, rtol=0, atol=0)

    # FP32-product dG would yield dZ=1/65536 per token instead of zero.


@pytest.mark.parametrize("precision", ["bf16", "mxfp8"])
@pytest.mark.parametrize("main_grad_dtype", [None, torch.float32, torch.bfloat16])
def test_mok_gated_matches_reference(context, precision, main_grad_dtype):
    inputs, weight = _dense_inputs(context)
    weights = inputs[3:9]
    forward_weights, backward_weights = _routed_weight_args(weights[3:], precision)
    accumulate = main_grad_dtype is not None
    state = _mok_schedule(context, inputs, 512 if accumulate else 4096)
    weight.requires_grad_(True)
    inputs[0].requires_grad_(True)
    second = (inputs[0].detach().mul(-0.5).requires_grad_(), *inputs[1:-1], inputs[-1] * 2)
    main_grads = gate_main_grad = None
    if accumulate:
        main_grads = tuple(torch.zeros_like(weights[i], dtype=main_grad_dtype) for i in (0, 3, 1, 4, 2, 5))
        gate_main_grad = torch.full_like(weight, 0.25, dtype=main_grad_dtype)
        # Returned MLP gradients are routed-first, unlike the main_grads ABI.
        returned_buffers = tuple(main_grads[i] for i in (1, 3, 5, 0, 2, 4)) + (gate_main_grad,)
        expected_buffers = [buffer.clone() for buffer in returned_buffers]

    for window in range(2 if accumulate else 1):
        if window:
            for buffer in (*returned_buffers, *expected_buffers):
                buffer.zero_()
        pending = []
        for batch in (inputs, second):
            x, _, probs = batch[:3]
            output, saved = functional.forward(
                *state,
                x,
                probs,
                *weights[:3],
                *forward_weights,
                shared_output_gate_weight=weight,
            )
            assert saved.shared_output.shape == x.shape
            assert saved.shared_output.dtype == saved.shared_output_gate.dtype == torch.bfloat16
            assert saved.shared_output_gate.shape == (x.shape[0], 1)
            assert not saved.shared_output_gate.requires_grad and saved.shared_output_gate.grad_fn is None
            gate = torch.sigmoid(torch.nn.functional.linear(x, weight))
            torch.testing.assert_close(saved.shared_output_gate, gate, rtol=0, atol=0)
            shared = run_swiglu_reference(x.detach() @ weights[0].T, x.detach() @ weights[1].T) @ weights[2].T
            check_correctness("shared_output", shared, saved.shared_output, BF16_TOLERANCE)
            snapshots = (saved.shared_output.clone(), saved.shared_output_gate.clone())
            pending.append((batch, output, saved, snapshots))

        # Both contexts must survive another forward on the same workspace.
        for batch, output, saved, snapshots in reversed(pending):
            torch.testing.assert_close(saved.shared_output, snapshots[0], rtol=0, atol=0)
            torch.testing.assert_close(saved.shared_output_gate, snapshots[1], rtol=0, atol=0)
            gradients = functional.backward(
                *state,
                saved,
                batch[-1],
                batch[0],
                batch[2],
                *weights[:3],
                *backward_weights,
                main_grads=main_grads,
                shared_output_gate_weight=weight,
                shared_output_gate_main_grad=gate_main_grad,
            )
            reference = list(run_reference_bf16(*batch, shared_output_gate_weight=weight))
            assert len(gradients) == 9 and gradients[-1].dtype == (main_grad_dtype or torch.float32)
            if accumulate:
                for actual, buffer, expected, contribution in zip(
                    gradients[2:],
                    returned_buffers,
                    expected_buffers,
                    reference[3:],
                    strict=True,
                ):
                    assert actual is buffer and actual.dtype == main_grad_dtype
                    expected.add_(contribution)
                reference[3:] = expected_buffers
            for index, (name, actual, golden) in enumerate(zip(GATED_RESULT_NAMES, (output, *gradients), reference, strict=True)):
                tolerance = MXFP8_TOLERANCE if precision == "mxfp8" and index < 6 else BF16_TOLERANCE
                check_correctness(name, golden, actual, tolerance, print_stats=context[0] == 0)


@pytest.mark.parametrize("precision,gated", [("bf16", False), ("mxfp8", True)])
def test_situ_fused_shared(context, precision, gated):
    inputs, weight = _dense_inputs(context)
    x, _, probs = inputs[:3]
    weights = inputs[3:9]
    fwd_weights, bwd_weights = _routed_weight_args(weights[3:], precision)
    state = _mok_schedule(context, inputs, macrobatch_size=512)  # Force routed replay.
    gate_kw = dict(shared_output_gate_weight=weight) if gated else {}
    beta = (2.0, 8.0)  # Non-default values detect a dropped activation parameter.

    def activation(a, u):
        a, u = a.float(), u.float()
        return (beta[0] * torch.tanh(a / beta[0]) * torch.sigmoid(a) * beta[1] * torch.tanh(u / beta[1])).bfloat16()

    golden = run_reference_bf16(
        *inputs,
        **gate_kw,
        routed_activation=activation,
        shared_activation=activation,
    )
    y, ctx = functional.forward(*state, x, probs, *weights[:3], *fwd_weights, **gate_kw, situ_beta=beta)
    assert ctx.situ_beta == beta  # the backward reads the betas off the context
    if not gated:
        ctx = functional.recompute_forward_context(
            *state,
            x,
            *weights[:2],
            *fwd_weights[:2],
            situ_beta=beta,
        )
        assert ctx.situ_beta == beta and ctx.x_routed is None
    grads = functional.backward(*state, ctx, inputs[-1], x, probs, *weights[:3], *bwd_weights, **gate_kw)
    assert len(grads) == 9 and (gated or grads[-1] is None)
    for index, (name, actual, target) in enumerate(zip(GATED_RESULT_NAMES, (y, *grads), golden)):
        tolerance = MXFP8_TOLERANCE if precision == "mxfp8" and index < 6 else BF16_TOLERANCE
        check_correctness(name, target, actual, tolerance, print_stats=context[0] == 0)


# --------------------------------------------------------------------------------------------------------------------
# Gate-weight checks and zero_unrouted_router_grads
#
# Gate-weight checks and the zero_unrouted_router_grads option of mcore-integration #7.
#
# The C++ port's tests/test_eager_dispatch.py tests of these two behaviours, with its problem
# builder, verbatim; the rest of that file is about its custom-op / direct-launch dispatch, which the
# CuTe DSL build (direct launches only) does not have.


def _problem(
    context,
    precision,
    *,
    gated=True,
    accumulate=True,
    single_grouped=True,
    num_local_tokens=512,
    hidden_dim=256,
):
    """One rank's MoE problem with unrouted slots; single-grouped is MCore's argument layout."""
    rank, world_size, device = context
    num_local_experts = 2
    intermediate_dim = 256
    topk = 2
    config = functional.CuteMoKConfig(
        fwd_num_comm_sms=2,
        bwd_num_comm_sms=2,
        minibatch_size=256,
        macrobatch_size=1024,
        schedule_capacity_multiplier=1.5,
    )
    workspace = functional.get_workspace(
        config,
        dist.group.WORLD,
        device=device,
        num_local_tokens=num_local_tokens,
        hidden_size=hidden_dim,
        topk=topk,
    )
    (
        x,
        top_experts,
        router_weights,
        w_shared_gate,
        w_shared_up,
        w_shared_down,
        w_routed_gate,
        w_routed_up,
        w_routed_down,
        d_output,
    ) = generate_inputs(
        rank,
        device,
        world_size * num_local_experts,
        num_local_experts,
        topk,
        num_local_tokens,
        hidden_dim,
        intermediate_dim,
    )
    top_experts = top_experts.to(torch.int32)
    top_experts[::7, 1] = -1
    top_experts[::11] = -1
    router_weights = router_weights.masked_fill(top_experts < 0, 0.0)
    if single_grouped:
        routed_fc1 = torch.cat((w_routed_gate, w_routed_up), dim=1)
        if precision == "mxfp8":
            fc1 = ops.mxfp8_quantize(routed_fc1, True, True)
            down = ops.mxfp8_quantize(w_routed_down, True, True)
            forward_routed = (fc1[:2], fc1[:2], down[:2])
            # TE-native columnwise payloads, as MCore single-grouped MXFP8 weights provide them.
            fc1 = (fc1[0], fc1[1], fc1[2].transpose(-2, -1).contiguous(), fc1[3], True)
            backward_routed = (fc1, fc1, (down[2].transpose(-2, -1).contiguous(), down[3], True))
        else:
            forward_routed = backward_routed = (routed_fc1, routed_fc1, w_routed_down)
        routed_main_grad_shapes = (routed_fc1, routed_fc1, w_routed_down)
    else:
        assert precision == "bf16"
        forward_routed = backward_routed = (w_routed_gate, w_routed_up, w_routed_down)
        routed_main_grad_shapes = forward_routed
    main_grads = gate_main_grad = None
    if accumulate:
        routed_gate_main_grad = torch.zeros_like(routed_main_grad_shapes[0], dtype=torch.float32)
        routed_up_main_grad = routed_gate_main_grad if single_grouped else torch.zeros_like(routed_main_grad_shapes[1], dtype=torch.float32)
        main_grads = (
            torch.zeros_like(w_shared_gate, dtype=torch.float32),
            routed_gate_main_grad,
            torch.zeros_like(w_shared_up, dtype=torch.float32),
            routed_up_main_grad,
            torch.zeros_like(w_shared_down, dtype=torch.float32),
            torch.zeros_like(w_routed_down, dtype=torch.float32),
        )
    gate_kwargs = {}
    if gated:
        generator = torch.Generator(device=device).manual_seed(77 + rank)
        gate_weight = torch.randn(1, hidden_dim, generator=generator, device=device, dtype=torch.bfloat16) / 16
        gate_kwargs = {"shared_output_gate_weight": gate_weight}
        if accumulate:
            gate_main_grad = torch.zeros_like(gate_weight, dtype=torch.float32)
    return dict(
        config=config,
        workspace=workspace,
        num_local_experts=num_local_experts,
        x=x,
        top_experts=top_experts,
        router_weights=router_weights,
        d_output=d_output,
        shared=(w_shared_gate, w_shared_up, w_shared_down),
        forward_routed=forward_routed,
        backward_routed=backward_routed,
        main_grads=main_grads,
        gate_main_grad=gate_main_grad,
        gate_kwargs=gate_kwargs,
    )


def _schedule(p, top_experts=None):
    return functional.build_schedule(
        p["workspace"],
        p["config"],
        p["top_experts"] if top_experts is None else top_experts,
        num_local_experts=p["num_local_experts"],
    )


def _forward(p, schedule, x=None, router_weights=None, **gate_kwargs):
    return functional.forward(
        p["config"],
        p["workspace"],
        schedule,
        p["x"] if x is None else x,
        p["router_weights"] if router_weights is None else router_weights,
        *p["shared"],
        *p["forward_routed"],
        **(p["gate_kwargs"] | gate_kwargs),
    )


def _backward(p, schedule, forward_context, d_output=None, x=None, router_weights=None, **kwargs):
    options = dict(p["gate_kwargs"], main_grads=p["main_grads"])
    if p["gate_main_grad"] is not None:
        options["shared_output_gate_main_grad"] = p["gate_main_grad"]
    return functional.backward(
        p["config"],
        p["workspace"],
        schedule,
        forward_context,
        p["d_output"] if d_output is None else d_output,
        p["x"] if x is None else x,
        p["router_weights"] if router_weights is None else router_weights,
        *p["shared"],
        *p["backward_routed"],
        **(options | kwargs),
    )


def _step(p, **backward_kwargs):
    schedule = _schedule(p)
    output, forward_context = _forward(p, schedule)
    return output, _backward(p, schedule, forward_context, **backward_kwargs)


@pytest.mark.parametrize("bad_weight", ["fp32", "short", "column", "strided"])
def test_gate_weight_checked_by_forward_and_backward(context, bad_weight):
    """forward and backward both reject a gate weight that is not contiguous BF16 [1, H], before launching anything.

    backward used to hand any weight to the fused gate kernels, which read it as BF16 [1, H] without checks.
    """
    p = _problem(context, "mxfp8", gated=True)
    weight = p["gate_kwargs"]["shared_output_gate_weight"]
    hidden = weight.shape[1]
    bad = {
        "fp32": lambda: weight.float(),
        "short": lambda: torch.cat((weight, weight), dim=1)[:, : hidden // 2],  # contiguous [1, H/2]
        "column": lambda: weight.reshape(hidden, 1),
        "strided": lambda: torch.stack((weight, weight), dim=-1).reshape(1, 2 * hidden)[:, ::2],
    }[bad_weight]()
    schedule = _schedule(p)
    with pytest.raises(ValueError, match="shared_output_gate_weight must be"):
        _forward(p, schedule, shared_output_gate_weight=bad)
    _, forward_context = _forward(p, schedule)
    with pytest.raises(ValueError, match="shared_output_gate_weight must be"):
        _backward(p, schedule, forward_context, shared_output_gate_weight=bad)
    torch.cuda.synchronize()
    # Nothing was launched: every main grad is still zero, and the same context still runs its backward.
    assert not any(main_grad.any() for main_grad in (*p["main_grads"], p["gate_main_grad"]))
    _backward(p, schedule, forward_context)
    torch.cuda.synchronize()
    assert p["main_grads"][0].any() and p["gate_main_grad"].any()


def test_zero_unrouted_router_grads_opt_out(context):
    problem = _problem(context, "mxfp8", gated=True)
    unrouted = problem["top_experts"] < 0
    assert unrouted.any() and (~unrouted).any()
    output, (d_x, d_router_weights, *_) = _step(problem)
    output_opt_out, (d_x_opt_out, d_router_weights_opt_out, *_) = _step(problem, zero_unrouted_router_grads=False)
    torch.cuda.synchronize()

    assert torch.equal(output, output_opt_out) and torch.equal(d_x, d_x_opt_out)
    assert not d_router_weights[unrouted].any()
    assert torch.equal(d_router_weights[~unrouted], d_router_weights_opt_out[~unrouted])
    # Unrouted slots are unspecified but never uninitialized memory.
    assert torch.isfinite(d_router_weights_opt_out).all()
