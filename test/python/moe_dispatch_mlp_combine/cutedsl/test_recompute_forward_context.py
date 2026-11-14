# Copyright 2026 Cursor Research
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Derived from Mixture-of-Kittens (https://github.com/cursor/mixture-of-kittens, commit 22fc95ae);
# ported to CuTe DSL and modified by NVIDIA.

"""Backward without the forward's routed context: the kernel replays the first macrobatch too.

Run under torchrun on SM100 / SM103, e.g.
``torchrun --nnodes=2 --nproc-per-node=4 -m pytest -s moe_dispatch_mlp_combine/cutedsl/test_recompute_forward_context.py``.

Four backwards of the same layer and inputs, each after its own forward (a context with routed
activations serves one backward: the kernel overwrites its Hidden):

  saved      ``backward`` with the context ``forward`` returned
  dropped    ``backward`` with ``forward_context.without_routed()``: the shared activations are the forward's,
             so only the routed replay differs from ``saved``
  recomputed ``backward`` with ``recompute_forward_context(...)``: the shared activations come from the forward
             megakernel's shared FC1 tasks, launched alone
  none       ``backward`` with ``forward_context=None``: nothing of the forward is passed

``recompute_forward_context`` must return the forward context's shared Gate / Up / Hidden byte for byte (BF16
unclamped and clamped, MXFP8 with both Gate / Up representations, SiTU-GLU).  ``dropped``, ``recomputed`` and
``none`` must therefore equal ``saved`` bit for bit (the replay recomputes the forward's routed activations
exactly, in BF16 and in both MXFP8 Gate / Up representations), and ``saved`` is compared with the PyTorch
reference (BF16, unclamped).  Geometries: one macrobatch (the default kernel has no replay there), several
macrobatches, and one minibatch per macrobatch.

The shared FC1 launch takes its sizes at run time: layers of different token counts, hidden and intermediate
sizes run one executor per clamp mode, in the forward host of the layer's precision
(``test_one_executor_serves_every_size``).
"""

import pytest
import torch
import torch.distributed as dist

from cudnn.moe_dispatch_mlp_combine import functional
from cudnn.moe_dispatch_mlp_combine.cutedsl._arch import SUPPORTED_CAPABILITIES
from cudnn.moe_dispatch_mlp_combine.functional import get_workspace
from cudnn.moe_dispatch_mlp_combine.ops import mxfp8_quantize

from .utils import (
    BACKWARD_RESULT_NAMES,
    BF16_TOLERANCE,
    capacity_multiplier,
    check_correctness,
    generate_inputs,
    run_reference_bf16,
    ungated,
)

NUM_LOCAL_EXPERTS, HIDDEN, INTERMEDIATE, TOPK, NUM_LOCAL_TOKENS, MINIBATCH = 8, 1024, 512, 4, 2048, 2048
# (num_local_tokens, hidden, intermediate): the file's layer, and a smaller one with a 128-wide hidden and
# intermediate tail.
SIZES = ((NUM_LOCAL_TOKENS, HIDDEN, INTERMEDIATE), (1024, 640, 640))
SHARED_ACTIVATIONS = ("gate_shared", "up_shared", "hidden_shared")
# (name, macrobatch size).  A rank's balanced share is 8192 routed rows.
GEOMETRIES = (
    ("one macrobatch", 16384),
    ("several macrobatches", 4096),
    ("one minibatch per macrobatch", 2048),
)


def _layer(context, macrobatch_size, sizes=SIZES[0], **config_fields):
    rank, world_size, device = context
    num_local_tokens, hidden, intermediate = sizes
    if torch.cuda.get_device_capability(device) not in SUPPORTED_CAPABILITIES:
        pytest.skip("CuTe DSL backends are SM100/SM103 only")
    config = functional.CuteMoKConfig(
        minibatch_size=MINIBATCH,
        macrobatch_size=macrobatch_size,
        fwd_num_comm_sms=30,
        bwd_num_comm_sms=14,
        schedule_capacity_multiplier=capacity_multiplier(world_size, num_local_tokens, TOPK, NUM_LOCAL_EXPERTS),
        **config_fields,
    )
    workspace = get_workspace(config, dist.group.WORLD, device=device, num_local_tokens=num_local_tokens, hidden_size=hidden, topk=TOPK)
    inputs = generate_inputs(rank, device, NUM_LOCAL_EXPERTS * world_size, NUM_LOCAL_EXPERTS, TOPK, num_local_tokens, hidden, intermediate)
    schedule = functional.build_schedule(workspace, config, inputs[1], num_local_experts=NUM_LOCAL_EXPERTS)
    return config, workspace, schedule, inputs


def _forward_weights(inputs, precision):
    """The routed weights as ``forward`` takes them: BF16 tensors, or MXFP8 (e4m3, scale) pairs."""
    if precision == "bf16":
        return tuple(inputs[6:9])
    return tuple(mxfp8_quantize(weight, True, False)[:2] for weight in inputs[6:9])


def _assert_shared_bytes(label, forward_context, recomputed):
    """The recomputed context's shared activations are the forward's byte for byte, in tensors of their own."""
    for name in SHARED_ACTIVATIONS:
        want, got = getattr(forward_context, name), getattr(recomputed, name)
        assert got is not want and got.dtype == want.dtype and got.shape == want.shape, (label, name)
        differing = int((want.view(torch.int16) != got.view(torch.int16)).sum())
        assert differing == 0, f"{label}: {differing} elements of the recomputed {name} differ from the forward's"


def _forward_then_recompute(label, config, workspace, schedule, inputs, forward_weights, swiglu_limit=None, situ_beta=None):
    """A forward, then ``recompute_forward_context`` of the same arguments: the shared activations must agree."""
    x, _, router_weights, w_shared_gate, w_shared_up, w_shared_down = inputs[:6]
    _, forward_context = functional.forward(
        config, workspace, schedule, x, router_weights, w_shared_gate, w_shared_up, w_shared_down, *forward_weights, swiglu_limit, situ_beta=situ_beta
    )
    recomputed = functional.recompute_forward_context(
        config, workspace, schedule, x, w_shared_gate, w_shared_up, *forward_weights[:2], swiglu_limit, situ_beta=situ_beta
    )
    _assert_shared_bytes(label, forward_context, recomputed)
    return recomputed


def _four_backwards(label, context, config, workspace, schedule, inputs, forward_weights, backward_weights, swiglu_limit):
    """The gradients of ``saved``, ``dropped``, ``recomputed`` and ``none`` (module docstring), cloned.  The
    recomputed context is compared with the forward's before the backward takes it."""
    _, _, device = context
    x, _, router_weights, w_shared_gate, w_shared_up, w_shared_down, _, _, _, d_output = inputs
    shared = (w_shared_gate, w_shared_up, w_shared_down)

    def backward(make_context):
        _, forward_context = functional.forward(config, workspace, schedule, x, router_weights, *shared, *forward_weights, swiglu_limit)
        gradients = ungated(
            functional.backward(
                config, workspace, schedule, make_context(forward_context), d_output, x, router_weights, *shared, *backward_weights, swiglu_limit
            )
        )
        torch.cuda.synchronize(device)  # a device fault is reported against this call
        return [gradient.clone() for gradient in gradients]

    def recompute(forward_context):
        recomputed = functional.recompute_forward_context(config, workspace, schedule, x, w_shared_gate, w_shared_up, *forward_weights[:2], swiglu_limit)
        _assert_shared_bytes(label, forward_context, recomputed)
        return recomputed

    saved = backward(lambda forward_context: forward_context)
    dropped = backward(lambda forward_context: forward_context.without_routed())
    recomputed = backward(recompute)
    none = backward(lambda forward_context: None)
    return saved, dropped, recomputed, none


def _compare(label, rank, saved, dropped, recomputed, none):
    """``dropped``, ``recomputed`` and ``none`` equal ``saved`` bit for bit."""
    for variant, gradients in (("dropped", dropped), ("recomputed", recomputed), ("none", none)):
        differing = [
            (name, int((want != got).sum())) for name, want, got in zip(BACKWARD_RESULT_NAMES, saved, gradients, strict=True) if not torch.equal(want, got)
        ]
        if rank == 0:
            print(f"{label} {variant} vs saved, bit for bit: " + (", ".join(f"{name}: {count} differ" for name, count in differing) or "same"))
        assert not differing, (label, variant, differing)


@pytest.mark.parametrize("swiglu_limit", (None, 4.0), ids=("unclamped", "clamped"))
def test_bf16(context: tuple[int, int, torch.device], swiglu_limit) -> None:
    rank = context[0]
    for name, macrobatch_size in GEOMETRIES:
        config, workspace, schedule, inputs = _layer(context, macrobatch_size)
        routed = _forward_weights(inputs, "bf16")
        saved, dropped, recomputed, none = _four_backwards(f"BF16 {name}", context, config, workspace, schedule, inputs, routed, routed, swiglu_limit)
        if swiglu_limit is None:
            for grad_name, want, got in zip(BACKWARD_RESULT_NAMES, run_reference_bf16(*inputs)[1:], saved, strict=True):
                check_correctness(f"BF16 {name} saved {grad_name} vs reference", want, got, BF16_TOLERANCE)
        _compare(f"BF16 {name}", rank, saved, dropped, recomputed, none)


@pytest.mark.parametrize("gate_up_context", ("mxfp8", "bf16"))
def test_mxfp8(context: tuple[int, int, torch.device], gate_up_context) -> None:
    rank = context[0]
    for name, macrobatch_size in GEOMETRIES:
        config, workspace, schedule, inputs = _layer(context, macrobatch_size, gate_up_context=gate_up_context)
        gate, up, down = (mxfp8_quantize(weight, True, True) for weight in inputs[6:9])
        forward_weights = (gate[:2], up[:2], down[:2])
        backward_weights = (gate, up, down[2:])
        label = f"MXFP8 ({gate_up_context} Gate / Up context) {name}"
        saved, dropped, recomputed, none = _four_backwards(label, context, config, workspace, schedule, inputs, forward_weights, backward_weights, None)
        _compare(label, rank, saved, dropped, recomputed, none)


@pytest.mark.parametrize("precision", ("bf16", "mxfp8"))
def test_situ(context: tuple[int, int, torch.device], precision) -> None:
    """SiTU-GLU: the betas reach the shared FC1 launch, and the recomputed context carries them."""
    config, workspace, schedule, inputs = _layer(context, GEOMETRIES[0][1])
    beta = (2.0, 8.0)  # not the defaults: a dropped beta changes Hidden
    recomputed = _forward_then_recompute(f"{precision} SiTU-GLU", config, workspace, schedule, inputs, _forward_weights(inputs, precision), situ_beta=beta)
    assert recomputed.situ_beta == beta


def _shared_fc1_executors(host):
    """The keys of the shared-FC1-only executors a forward host compiled."""
    return {key for key in host._EXECUTOR_CACHE if ("shared_fc1_only", True) in key}


@pytest.mark.parametrize("precision", ("bf16", "mxfp8"))
def test_one_executor_serves_every_size(context: tuple[int, int, torch.device], precision) -> None:
    """Token count, hidden and intermediate size are runtime values of the shared FC1 launch: layers of different
    sizes run one executor per clamp mode, compiled by the forward host of the layer's precision alone."""
    from cudnn.moe_dispatch_mlp_combine.cutedsl import persistent_bf16_fwd, persistent_mxfp8_fwd

    host, other = (persistent_bf16_fwd, persistent_mxfp8_fwd) if precision == "bf16" else (persistent_mxfp8_fwd, persistent_bf16_fwd)
    clamp_axis, situ_axis = ("swiglu", "situ") if precision == "bf16" else ("is_clamped", "is_situ")
    others = _shared_fc1_executors(other)
    for swiglu_limit in (None, 4.0):
        for sizes in SIZES:
            config, workspace, schedule, inputs = _layer(context, GEOMETRIES[1][1], sizes)
            _forward_then_recompute(
                f"{precision} {sizes} limit {swiglu_limit}", config, workspace, schedule, inputs, _forward_weights(inputs, precision), swiglu_limit
            )
    executors = _shared_fc1_executors(host)
    for is_clamped in (False, True):
        # SiTU-GLU is another compile-time axis (test_situ compiles its own).
        assert sum((clamp_axis, is_clamped) in key and (situ_axis, False) in key for key in executors) == 1, executors
    assert _shared_fc1_executors(other) == others, "the other precision's host compiled a shared FC1 executor"


@pytest.mark.parametrize("precision", ("bf16", "mxfp8"))
def test_recompute_backward_takes_its_own_comm_sms(context: tuple[int, int, torch.device], precision) -> None:
    """``bwd_recompute_num_comm_sms`` is the communication SM count of the backward that replays the first
    macrobatch too (``dropped``, ``recomputed``, ``none``); the backward on a saved context keeps
    ``bwd_num_comm_sms``.  The count is a runtime launch value: the gradients are the same bit for bit."""
    from unittest import mock

    from cudnn.moe_dispatch_mlp_combine.cutedsl import persistent_bf16_bwd, persistent_mxfp8_bwd

    host, entry = (persistent_bf16_bwd, "backward_bf16") if precision == "bf16" else (persistent_mxfp8_bwd, "backward_mxfp8")
    config, workspace, schedule, inputs = _layer(context, GEOMETRIES[1][1], bwd_recompute_num_comm_sms=20)
    assert config.bwd_num_comm_sms == 14
    forward_weights = _forward_weights(inputs, precision)
    backward_weights = forward_weights
    if precision == "mxfp8":
        gate, up, down = (mxfp8_quantize(weight, True, True) for weight in inputs[6:9])
        backward_weights = (gate, up, down[2:])
    launched = []
    original = getattr(host, entry)

    def spy(*args, **kwargs):
        launched.append(kwargs["num_comm_sms"])
        return original(*args, **kwargs)

    with mock.patch.object(host, entry, spy):
        saved, dropped, recomputed, none = _four_backwards(
            f"{precision} comm SMs", context, config, workspace, schedule, inputs, forward_weights, backward_weights, None
        )
    assert launched == [14, 20, 20, 20], launched
    _compare(f"{precision} with its own comm SMs", context[0], saved, dropped, recomputed, none)


def test_context_contract(context: tuple[int, int, torch.device]) -> None:
    """A context holds all four routed activations or none; a context without them serves any number of backwards."""
    config, workspace, schedule, inputs = _layer(context, GEOMETRIES[0][1])
    x, _, router_weights, w_shared_gate, w_shared_up, w_shared_down, gate, up, down, d_output = inputs
    weights = (w_shared_gate, w_shared_up, w_shared_down, gate, up, down)
    _, forward_context = functional.forward(config, workspace, schedule, x, router_weights, *weights)
    dropped = forward_context.without_routed()
    assert dropped.gate_shared is forward_context.gate_shared and dropped.x_routed is None
    partial = functional.CuteMoKForwardContext(
        x_routed=forward_context.x_routed,
        gate_shared=forward_context.gate_shared,
        gate_routed=None,
        up_shared=forward_context.up_shared,
        up_routed=None,
        hidden_shared=forward_context.hidden_shared,
        hidden_routed=None,
    )
    with pytest.raises(ValueError):
        functional.backward(config, workspace, schedule, partial, d_output, x, router_weights, *weights)
    first = [g.clone() for g in ungated(functional.backward(config, workspace, schedule, dropped, d_output, x, router_weights, *weights))]
    second = ungated(functional.backward(config, workspace, schedule, dropped, d_output, x, router_weights, *weights))
    for name, want, got in zip(BACKWARD_RESULT_NAMES, first, second, strict=True):
        assert torch.equal(want, got), name
