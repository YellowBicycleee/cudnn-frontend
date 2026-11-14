# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Gate-weight checks and the zero_unrouted_router_grads option of mcore-integration #7.

The C++ port's tests/test_eager_dispatch.py tests of these two behaviours, with its problem
builder, verbatim; the rest of that file is about its custom-op / direct-launch dispatch, which the
CuTe DSL build (direct launches only) does not have.
"""

import pytest
import torch
import torch.distributed as dist
import torch.utils._pytree as pytree  # noqa: F401  (kept with the builder)

from cudnn.moe_dispatch_mlp_combine import functional, ops

from .utils import generate_inputs


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
