# Copyright 2026 Cursor Research
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# generate_inputs is derived from the test utilities of Mixture-of-Kittens
# (https://github.com/cursor/mixture-of-kittens, commit 22fc95ae); modified by NVIDIA.

"""Helpers of the tests that call the layer the way MCore does: inputs, call arguments, precision."""

from __future__ import annotations

import torch
import torch.distributed as dist

GRADIENT_NAMES = ("d_x", "d_router_weights", "d_w_routed_gate", "d_w_routed_up", "d_w_routed_down", "d_w_shared_gate", "d_w_shared_up", "d_w_shared_down")
# Same as utils.py BF16_TOLERANCE / MXFP8_TOLERANCE: (max abs error, relative L1 error).
TOLERANCE = (0.5, 0.01)
BF16_TOLERANCE = (0.5, 0.01)
MXFP8_TOLERANCE = (1.0, 0.1)
TOLERANCES = {"bf16": BF16_TOLERANCE, "mxfp8": MXFP8_TOLERANCE}


def generate_inputs(
    rank: int, device: torch.device, seed: int, *, num_experts: int, local_experts: int, topk: int, tokens: int, hidden: int, intermediate: int
):
    """Random inputs of one rank, built as utils.generate_inputs builds them.

    Returns:
        (x, topk_experts, router_weights, w_shared_gate, w_shared_up, w_shared_down, w_routed_gate,
        w_routed_up, w_routed_down, d_output): top-k of Gaussian router logits (close to balanced
        routing), softmax router weights, BF16 activations and weights scaled by 1/sqrt(fan-in).
    """
    g = torch.Generator(device=device).manual_seed(seed + rank)
    bf16 = dict(dtype=torch.bfloat16, device=device)
    router_logits = torch.randn(tokens, num_experts, generator=g, device=device)
    topk_values, topk_experts = torch.topk(router_logits, topk, dim=1)
    router_weights = torch.softmax(topk_values.float(), dim=-1)
    x = torch.randn(tokens, hidden, generator=g, **bf16)
    w_shared_gate = torch.randn(intermediate, hidden, generator=g, **bf16) * hidden**-0.5
    w_shared_up = torch.randn(intermediate, hidden, generator=g, **bf16) * hidden**-0.5
    w_shared_down = torch.randn(hidden, intermediate, generator=g, **bf16) * intermediate**-0.5
    w_routed_gate = torch.randn(local_experts, intermediate, hidden, generator=g, **bf16) * hidden**-0.5
    w_routed_up = torch.randn(local_experts, intermediate, hidden, generator=g, **bf16) * hidden**-0.5
    w_routed_down = torch.randn(local_experts, hidden, intermediate, generator=g, **bf16) * intermediate**-0.5
    d_output = torch.randn(tokens, hidden, generator=g, **bf16) * hidden**-0.5
    return (x, topk_experts, router_weights, w_shared_gate, w_shared_up, w_shared_down, w_routed_gate, w_routed_up, w_routed_down, d_output)


def error_metric(name: str, reference: torch.Tensor, actual: torch.Tensor, tolerance=None) -> dict:
    """Max abs and relative L1 error of ``actual`` against ``reference``.

    Args:
        tolerance: (max abs error, relative L1 error) pair; ``TOLERANCE`` (BF16) when omitted.
    """
    tol = TOLERANCE if tolerance is None else tolerance
    diff = (reference.float() - actual.float()).abs()
    abs_max = diff.max().item()
    rel_l1 = (diff.sum() / reference.float().abs().sum().clamp_min(1e-30)).item()
    finite = bool(torch.isfinite(actual.float()).all().item())
    return dict(name=name, abs_error_max=abs_max, relative_l1_error=rel_l1, passed=finite and abs_max <= tol[0] and rel_l1 <= tol[1])


def clone_grads(grads):
    """Clones a backward's gradients, keeping the ``None`` entries some builds return."""
    return tuple(None if g is None else g.clone() for g in grads)


def grad_metrics(reference, actual, tolerance=None):
    """Error metrics over the gradients ``GRADIENT_NAMES`` covers.

    MOK builds differ in how many gradients ``backward`` returns and leave entries ``None`` for an
    absent weight (a routed-only layer has no shared-expert gradients), so pairs where either side is
    ``None`` are skipped rather than compared.
    """
    return [error_metric(name, a, b, tolerance) for name, a, b in zip(GRADIENT_NAMES, reference, actual) if a is not None and b is not None]


def quantize_routed(w_routed_gate, w_routed_up, w_routed_down):
    """MXFP8 routed-weight arguments of the public API, from BF16 routed weights.

    ``cudnn.moe_ep.cutemok.ops.mxfp8_quantize(w, True, True)`` returns (fp8, scale, transposed fp8, transposed scale).
    The forward takes rowwise (fp8, scale) pairs; the backward takes the four-tuple for gate/up
    (rowwise for the recompute, columnwise for the dgrad) and the columnwise pair for down.

    Returns:
        (forward_weights, backward_weights): two 3-tuples in (gate, up, down) order, each ready to
        splat into ``functional.forward`` / ``functional.backward``.
    """
    from cudnn.moe_ep.cutemok.ops import mxfp8_quantize

    gate, up, down = (tuple(mxfp8_quantize(w, True, True)) for w in (w_routed_gate, w_routed_up, w_routed_down))
    forward = (gate[:2], up[:2], down[:2])
    backward = (gate, up, down[2:])
    return forward, backward


#: How the tools hand the weights to MOK.  ``plain``: dense ``[E, I, H]`` gate / up / down tensors
#: (MXFP8 as ``quantize_routed`` tuples), weight gradients returned.  ``mcore``: the arguments Megatron's
#: MOK adapter (megatron/core/transformer/moe/megakernel/mok/runtime.py) passes when the routed experts
#: are separate parameters, which is the default (no ``--moe-single-grouped-weight``): every expert's
#: FC1 ``[2I, H]`` and FC2 ``[H, I]`` its own allocation behind a ``SplitRoutedWeight`` (MXFP8 with
#: TransformerEngine's native columnwise payload), FC1 given as both the gate and the up argument, the
#: shared gate / up row views of one shared FC1, and the weight gradients accumulated into FP32
#: main_grads (gradient_accumulation_fusion) through per-expert storage tables.
LAYOUTS = ("plain", "mcore")


def shared_output_gate_weight(rank: int, device: torch.device, seed: int, hidden: int) -> torch.Tensor:
    """A ``[1, hidden]`` BF16 shared output gate weight (Qwen3.5's ``--moe-shared-expert-gate``)."""
    g = torch.Generator(device=device).manual_seed(seed + 7919 * (rank + 1))
    return torch.randn(1, hidden, generator=g, dtype=torch.bfloat16, device=device) * hidden**-0.5


def _native_split_mxfp8(experts):
    """One MCore split MXFP8 view over per-expert BF16 matrices (each quantized on its own).

    MCore takes TransformerEngine's rowwise payload ``[M, K]`` and its columnwise payload in the
    parameter's own ``[M, K]`` order (``native_columnwise``), each scale swizzled to the tcgen05 layout
    (the columnwise one as the ``[K, M]`` matrix MOK's dgrad reads).  ``mxfp8_quantize`` quantizes the
    same 32-element blocks; its transposed payload transposed back is the native columnwise payload, and
    its two scales are already in the swizzled layouts.
    """
    from cudnn.moe_ep.cutemok import functional, ops

    rows, row_scales, columns, column_scales = [], [], [], []
    for weight in experts:
        data, scale, transposed, transposed_scale = ops.mxfp8_quantize(weight.unsqueeze(0), True, True)
        rows.append(data[0])
        row_scales.append(scale)
        columns.append(transposed[0].transpose(0, 1).contiguous())
        column_scales.append(transposed_scale)
    return functional.SplitRoutedWeight(
        data=rows[0],
        storage_table=ops.make_routed_weight_storage_table_mxfp8(rows),
        scale=row_scales[0],
        scale_storage_table=ops.make_routed_scale_storage_table(row_scales),
        scale_tensors=tuple(row_scales),
        transposed_data=columns[0],
        transposed_scale=column_scales[0],
        transposed_storage_table=ops.make_routed_weight_storage_table_mxfp8(columns),
        transposed_scale_storage_table=ops.make_routed_scale_storage_table(column_scales),
        transposed_scale_tensors=tuple(column_scales),
        native_columnwise=True,
    ), (rows, columns)


def call_arguments(layout: str, precision: str, shared_weights, routed_weights, *, output_gate: torch.Tensor | None = None):
    """Weight arguments of ``functional.forward`` / ``functional.backward`` in one ``LAYOUTS`` layout.

    Args:
        shared_weights: BF16 (gate ``[I, H]``, up ``[I, H]``, down ``[H, I]``).
        routed_weights: BF16 (gate ``[E, I, H]``, up ``[E, I, H]``, down ``[E, H, I]``); the ``mcore``
            layout copies them into per-expert allocations, so the caller may drop them afterwards.
        output_gate: the shared output gate weight, or None for none.

    Returns:
        (forward_weights, backward_weights, forward_kwargs, backward_kwargs, keepalive): the six weight
        arguments of each call (shared gate, up, down, routed gate, up, down), their keyword arguments,
        and the per-expert allocations the storage tables name, which must outlive every call.
    """
    from cudnn.moe_ep.cutemok import functional, ops

    if layout not in LAYOUTS:
        raise ValueError(f"layout must be one of {LAYOUTS}, got {layout!r}")
    forward_kwargs, backward_kwargs, keepalive = {}, {}, []
    if output_gate is not None:
        forward_kwargs["shared_output_gate_weight"] = output_gate
        backward_kwargs["shared_output_gate_weight"] = output_gate
    if layout == "plain":
        if precision == "mxfp8":
            forward_routed, backward_routed = quantize_routed(*routed_weights)
        else:
            forward_routed = backward_routed = tuple(routed_weights)
        return ((*shared_weights, *forward_routed), (*shared_weights, *backward_routed), forward_kwargs, backward_kwargs, keepalive)

    gate, up, down = routed_weights
    experts, intermediate, hidden = gate.shape
    shared_fc1 = torch.cat(shared_weights[:2])  # runtime.py narrows one [2I, H] parameter
    shared = (shared_fc1[:intermediate], shared_fc1[intermediate:], shared_weights[2])
    fc1 = [torch.cat((gate[e], up[e])) for e in range(experts)]  # each expert its own allocation
    fc2 = [down[e].clone() for e in range(experts)]
    if precision == "mxfp8":
        (fc1_view, fc1_payloads), (fc2_view, fc2_payloads) = _native_split_mxfp8(fc1), _native_split_mxfp8(fc2)
        keepalive += [fc1_payloads, fc2_payloads]
        del fc1, fc2
    else:
        fc1_view = functional.SplitRoutedWeight(data=fc1[0], storage_table=ops.make_routed_weight_storage_table_bf16(fc1))
        fc2_view = functional.SplitRoutedWeight(data=fc2[0], storage_table=ops.make_routed_weight_storage_table_bf16(fc2))
        keepalive += [fc1, fc2]
    weights = (*shared, fc1_view, fc1_view, fc2_view)

    # FP32 main_grads, as MCore's DDP buffer holds them: the routed experts' buffers are consecutive
    # views of one flat allocation, the shared FC1's gate / up are row views of one buffer.
    f32 = dict(dtype=torch.float32, device=gate.device)
    shared_fc1_grad = torch.zeros(2 * intermediate, hidden, **f32)
    shared_fc2_grad = torch.zeros(hidden, intermediate, **f32)
    flat = torch.zeros(experts * 3 * intermediate * hidden, **f32)
    size = 2 * intermediate * hidden
    fc1_grads = [flat[e * size : (e + 1) * size].view(2 * intermediate, hidden) for e in range(experts)]
    base = experts * size
    fc2_grads = [flat[base + e * hidden * intermediate : base + (e + 1) * hidden * intermediate].view(hidden, intermediate) for e in range(experts)]
    fc1_table = ops.make_routed_d_weight_storage_table(fc1_grads)
    fc2_table = ops.make_routed_d_weight_storage_table(fc2_grads)
    keepalive.append(flat)
    backward_kwargs["main_grads"] = (shared_fc1_grad[:intermediate], fc1_grads[0], shared_fc1_grad[intermediate:], fc1_grads[0], shared_fc2_grad, fc2_grads[0])
    backward_kwargs["main_grad_storage_tables"] = (fc1_table, fc1_table, fc2_table)
    if output_gate is not None:
        backward_kwargs["shared_output_gate_main_grad"] = torch.zeros(output_gate.shape, **f32)
    return weights, weights, forward_kwargs, backward_kwargs, keepalive


def compared_gradients(grads, layout: str):
    """The gradients a precision check compares: all of them in the ``plain`` layout; in ``mcore`` only
    d_x and d_router_weights, since the weight gradients accumulate into main_grads call after call."""
    return tuple(grads) if layout == "plain" else tuple(grads[:2])


def all_ranks(flag: bool, device: torch.device) -> bool:
    """True when ``flag`` is true on every rank."""
    t = torch.tensor([int(flag)], device=device)
    dist.all_reduce(t, op=dist.ReduceOp.MIN)
    return bool(t.item())
