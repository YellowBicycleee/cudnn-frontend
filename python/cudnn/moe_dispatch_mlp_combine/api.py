# Copyright 2026 Cursor Research
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Derived from Mixture-of-Kittens (https://github.com/cursor/mixture-of-kittens, commit 22fc95ae);
# ported to CuTe DSL and modified by NVIDIA.

"""The operator API: ``moe_dispatch_mlp_combine_prepare``, ``moe_dispatch_mlp_combine_forward`` and
``moe_dispatch_mlp_combine_backward`` (the names cuDNN frontend exports as ``cudnn.ops.*``).

Thin wrappers over the plan / execute entries of ``cudnn.moe_dispatch_mlp_combine.functional``: ``prepare`` is ``make_plan``,
``forward`` is ``schedule_execute`` followed by ``forward_execute``, ``backward`` is ``backward_execute``.  The
wrappers add one thing, the decline of an input layout or dtype the execute steps do not serve, raised as
``NotImplementedError`` naming the tensor before anything is launched.  They compile nothing and allocate no
device memory: the plan compiled the kernels and owns the counters and scratch; the caller passes the outputs.
"""

import torch
import torch.distributed as dist

from .._tupledict import TupleDict
from .functional import (
    CuteMoKConfig,
    CuteMoKForwardContext,
    CuteMoKPlan,
    CuteMoKSchedule,
    SplitRoutedWeight,
    backward_execute,
    forward_execute,
    make_plan,
    schedule_execute,
)

#: The keys of the backward's result, in order.  The ninth is the shared output gate weight's gradient, which this
#: API does not compute (None).
_BACKWARD_RESULT_NAMES = (
    "d_x",
    "d_router_weights",
    "d_w_routed_gate",
    "d_w_routed_up",
    "d_w_routed_down",
    "d_w_shared_gate",
    "d_w_shared_up",
    "d_w_shared_down",
    "d_w_output_gate",
)

__all__ = [
    "MoeDispatchMlpCombineConfig",
    "MoeDispatchMlpCombinePlan",
    "moe_dispatch_mlp_combine_backward",
    "moe_dispatch_mlp_combine_forward",
    "moe_dispatch_mlp_combine_prepare",
]

#: The runtime configuration of every call (communication SMs, macrobatch / minibatch sizes, schedule capacity).
MoeDispatchMlpCombineConfig = CuteMoKConfig
#: What ``moe_dispatch_mlp_combine_prepare`` returns; ``*_shapes`` give the shapes of the caller's buffers.
MoeDispatchMlpCombinePlan = CuteMoKPlan


def _decline_layout(name: str, tensor, dtype: torch.dtype) -> None:
    """``NotImplementedError`` for a tensor the execute steps do not read natively: they take contiguous
    ``[num_local_tokens, ...]`` tensors of one dtype and never convert or copy them into another layout."""
    if tensor is None:
        return
    if not isinstance(tensor, torch.Tensor):
        raise NotImplementedError(f"{name} must be a torch.Tensor; got {type(tensor).__name__}")
    if tensor.dtype != dtype or tensor.ndim != 2 or not tensor.is_contiguous():
        raise NotImplementedError(
            f"{name}: only a contiguous 2-D {dtype} tensor is supported; got {tensor.dtype} of shape "
            f"{tuple(tensor.shape)} and strides {tuple(tensor.stride())}"
        )


def _decline_routed_weights(plan, weights) -> None:
    """The routed weights' form must be the plan's: BF16 tensors (or BF16 ``SplitRoutedWeight``) for a BF16 plan,
    MXFP8 tuples (or ``SplitRoutedWeight`` with scales) for an MXFP8 plan."""
    if not isinstance(plan, CuteMoKPlan):
        raise TypeError("plan must come from moe_dispatch_mlp_combine_prepare")
    for name, weight in zip(("routed_gate_weights", "routed_up_weights", "routed_down_weights"), weights):
        if isinstance(weight, tuple) or (isinstance(weight, SplitRoutedWeight) and weight.scale is not None):
            form = "mxfp8"
        elif isinstance(weight, SplitRoutedWeight) or (isinstance(weight, torch.Tensor) and weight.dtype == torch.bfloat16):
            form = "bf16"
        else:
            raise NotImplementedError(
                f"{name}: a BF16 tensor, an MXFP8 (data, scale, ...) tuple or a SplitRoutedWeight is supported; "
                f"got {getattr(weight, 'dtype', type(weight).__name__)}"
            )
        if form != plan.routed_precision:
            raise NotImplementedError(
                f"{name}: {form} routed weights on a plan prepared for {plan.routed_precision} " "(moe_dispatch_mlp_combine_prepare's routed_precision)"
            )


def moe_dispatch_mlp_combine_prepare(
    config: CuteMoKConfig,
    group: dist.ProcessGroup,
    *,
    device: torch.device,
    max_tokens_per_rank: int,
    hidden_size: int,
    intermediate_size: int,
    num_local_experts: int,
    topk: int,
    routed_hidden_size: int | None = None,
    swiglu_clamped: bool = False,
    situ: bool = False,
    has_shared: bool = True,
    split_routed_weights: bool = False,
    shared_output_gate: bool = False,
    backward: bool = True,
    main_grad_dtype: torch.dtype | None = None,
    routed_precision: str = "bf16",
    native_columnwise: bool = False,
) -> CuteMoKPlan:
    """Prepare one layer configuration: compile every kernel the forward and backward launch and allocate the
    symmetric buffers, counters and scratch they use.  Collective over ``group``; outside CUDA graph capture.

    The arguments are ``cudnn.moe_dispatch_mlp_combine.functional.make_plan``'s, which this calls.  The token count, macrobatch /
    minibatch sizes and communication SMs are not part of the plan: they are runtime values of every call (token
    counts up to ``max_tokens_per_rank``; macrobatches up to ``config.macrobatch_size``).

    Returns:
        MoeDispatchMlpCombinePlan.  ``plan.schedule_shapes(tokens)``, ``plan.forward_shapes(config, tokens)`` and
        ``plan.backward_shapes(tokens)`` give the shapes of the buffers the forward and backward write.
    """
    return make_plan(
        config,
        group,
        device=device,
        max_tokens_per_rank=max_tokens_per_rank,
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        num_local_experts=num_local_experts,
        topk=topk,
        routed_hidden_size=routed_hidden_size,
        swiglu_clamped=swiglu_clamped,
        situ=situ,
        has_shared=has_shared,
        split_routed_weights=split_routed_weights,
        shared_output_gate=shared_output_gate,
        backward=backward,
        main_grad_dtype=main_grad_dtype,
        routed_precision=routed_precision,
        native_columnwise=native_columnwise,
    )


def moe_dispatch_mlp_combine_forward(
    x: torch.Tensor,
    top_experts: torch.Tensor,
    router_weights: torch.Tensor,
    shared_gate_weights: torch.Tensor | None,
    shared_up_weights: torch.Tensor | None,
    shared_down_weights: torch.Tensor | None,
    routed_gate_weights: torch.Tensor | tuple | SplitRoutedWeight,
    routed_up_weights: torch.Tensor | tuple | SplitRoutedWeight,
    routed_down_weights: torch.Tensor | tuple | SplitRoutedWeight,
    *,
    plan: CuteMoKPlan,
    config: CuteMoKConfig,
    out: torch.Tensor,
    context_out: tuple,
    schedule_out: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    swiglu_limit: float | None = None,
    routed_input: torch.Tensor | None = None,
    shared_output_gate: torch.Tensor | None = None,
    situ_beta: tuple[float, float] | None = None,
) -> TupleDict:
    """Dispatch the tokens to their top-k experts, run the expert MLPs and combine the results.  Collective over
    the plan's group; every rank passes the same token count.

    Launches ``schedule_execute``'s kernels (route all-gather, barrier, three scheduler kernels) and then
    ``forward_execute``'s (staging copy, barrier, counter reset, forward megakernel, barrier, epilogue).

    Inputs:
        x:                 bfloat16 [T, hidden_size], T a multiple of 256 in [512, max_tokens_per_rank]
        top_experts:       int32 [T, topk], global expert ids; -1 marks an unused route slot
        router_weights:    float32 [T, topk]
        shared_*_weights:  bfloat16 [I, H], [I, H], [H, I], or three None for a layer without a shared expert
        routed_*_weights:  bfloat16 [E_local, I, H], [E_local, I, H], [E_local, H, I]; MXFP8 (data, scale)
                           tuples; or SplitRoutedWeight.  One object as both gate and up: a combined FC1.
        plan:              from moe_dispatch_mlp_combine_prepare
        config:            MoeDispatchMlpCombineConfig of this call
        out:               bfloat16 [T, routed_hidden_size], the output (``plan.forward_shapes(config, T)[1]``)
        context_out:       the saved activations' buffers (``plan.forward_shapes(config, T)[0]``)
        schedule_out:      three int32 buffers (``plan.schedule_shapes(T)``)
        swiglu_limit, routed_input, shared_output_gate (G = sigmoid(x @ w^T), bfloat16 [T, 1]), situ_beta:
                           the options of ``cudnn.moe_dispatch_mlp_combine.functional.forward_execute``

    Returns:
        TupleDict(output, schedule, forward_context): ``output`` is ``out`` (with differing hiddens the pair
        (y_shared, y_routed)); ``schedule`` and ``forward_context`` live in ``schedule_out`` and ``context_out``
        and are passed to the backward.  Keep those buffers untouched until then.
    """
    for name, tensor, dtype in (
        ("x", x, torch.bfloat16),
        ("top_experts", top_experts, torch.int32),
        ("router_weights", router_weights, torch.float32),
        ("routed_input", routed_input, torch.bfloat16),
    ):
        _decline_layout(name, tensor, dtype)
    _decline_routed_weights(plan, (routed_gate_weights, routed_up_weights, routed_down_weights))
    schedule = schedule_execute(plan, config, top_experts, out=schedule_out)
    output, forward_context = forward_execute(
        plan,
        config,
        schedule,
        x,
        router_weights,
        shared_gate_weights,
        shared_up_weights,
        shared_down_weights,
        routed_gate_weights,
        routed_up_weights,
        routed_down_weights,
        swiglu_limit,
        routed_input,
        context_out=context_out,
        out=out,
        shared_output_gate=shared_output_gate,
        situ_beta=situ_beta,
    )
    return TupleDict(output=output, schedule=schedule, forward_context=forward_context)


def moe_dispatch_mlp_combine_backward(
    grad_output: torch.Tensor,
    x: torch.Tensor,
    router_weights: torch.Tensor,
    shared_gate_weights: torch.Tensor | None,
    shared_up_weights: torch.Tensor | None,
    shared_down_weights: torch.Tensor | None,
    routed_gate_weights: torch.Tensor | tuple | SplitRoutedWeight,
    routed_up_weights: torch.Tensor | tuple | SplitRoutedWeight,
    routed_down_weights: torch.Tensor | tuple | SplitRoutedWeight,
    schedule: CuteMoKSchedule,
    forward_context: CuteMoKForwardContext,
    *,
    plan: CuteMoKPlan,
    config: CuteMoKConfig,
    out: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
    d_router_weights_out: torch.Tensor,
    d_weights_out: tuple | None = None,
    main_grads: tuple | None = None,
    main_grad_storage_tables: tuple | None = None,
    swiglu_limit: float | None = None,
    routed_grad_output: torch.Tensor | None = None,
    routed_input: torch.Tensor | None = None,
    zero_unrouted_router_grads: bool = True,
) -> TupleDict:
    """The gradients of ``moe_dispatch_mlp_combine_forward`` with respect to x, the router weights and the six
    weights.  Collective over the plan's group.  Launches ``backward_execute``'s kernels (staging copy, barrier,
    counter reset, metadata builder or tensormap prologue, backward megakernel, barrier, epilogue, router-weight
    gradient copy and, with ``zero_unrouted_router_grads``, the masking of unrouted slots).

    Inputs: the forward's, with the weights' backward forms (MXFP8: (e4m3, scale, e4m3_t, scale_t) for gate / up,
    (e4m3_t, scale_t) for down), and:
        grad_output:          bfloat16 [T, hidden_size]
        schedule, forward_context: the forward's results
        out:                  bfloat16 [T, hidden_size] for d_x (``plan.backward_shapes(T)[0]``)
        d_router_weights_out: float32 [T, topk]
        d_weights_out:        the six weight gradients' buffers (``plan.backward_shapes(T)[2]``), or None with
                              ``main_grads``, the buffers the gradients are accumulated into
        main_grad_storage_tables, swiglu_limit, routed_grad_output, routed_input, zero_unrouted_router_grads:
                              the options of ``cudnn.moe_dispatch_mlp_combine.functional.backward_execute``

    Returns:
        TupleDict of ``backward_execute``'s nine entries over the caller's buffers, in its order: d_x,
        d_router_weights, d_w_routed_gate, d_w_routed_up, d_w_routed_down, d_w_shared_gate,
        d_w_shared_up, d_w_shared_down and d_w_output_gate (None: not computed here).
    """
    for name, tensor, dtype in (
        ("grad_output", grad_output, torch.bfloat16),
        ("x", x, torch.bfloat16),
        ("router_weights", router_weights, torch.float32),
        ("routed_grad_output", routed_grad_output, torch.bfloat16),
        ("routed_input", routed_input, torch.bfloat16),
    ):
        _decline_layout(name, tensor, dtype)
    _decline_routed_weights(plan, (routed_gate_weights, routed_up_weights, routed_down_weights))
    gradients = backward_execute(
        plan,
        config,
        schedule,
        forward_context,
        grad_output,
        x,
        router_weights,
        shared_gate_weights,
        shared_up_weights,
        shared_down_weights,
        routed_gate_weights,
        routed_up_weights,
        routed_down_weights,
        swiglu_limit,
        main_grads,
        main_grad_storage_tables,
        routed_grad_output,
        routed_input,
        out=out,
        d_router_weights_out=d_router_weights_out,
        d_weights_out=d_weights_out,
        zero_unrouted_router_grads=zero_unrouted_router_grads,
    )
    return TupleDict(zip(_BACKWARD_RESULT_NAMES, gradients, strict=True))
