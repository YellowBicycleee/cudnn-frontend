# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Geometry variants of the megakernels: a 128-wide hidden or intermediate tail (H or I % 256 == 128) and the
communication tile shape of the BF16 megakernels.

Run under torchrun on eight GPUs (SM100 / SM103).
"""

import contextlib
import dataclasses
from unittest import mock

import pytest
import torch
import torch.distributed as dist

from cudnn.moe_ep.cutemok import functional, ops
from cudnn.moe_ep.cutemok.cutedsl import persistent_bf16_bwd, persistent_bf16_fwd
from cudnn.moe_ep.cutemok.cutedsl._arch import TARGET_ARCH, TARGET_CAPABILITY
from cudnn.moe_ep.cutemok.cutedsl.persistent_bf16_fwd_contract import COMM_TILE_SHAPES

from .utils import BF16_TOLERANCE, MXFP8_TOLERANCE, check_correctness, generate_inputs, run_reference_bf16

# --------------------------------------------------------------------------------------------------------------------
# Hidden tail
#
# A 128-aligned hidden size (H % 256 != 0), e.g. H = 2688 = 21 x 128.
#
# The CUDA kernels admit H >= 256 with H % 128 == 0 since mcore-integration #5, as for the intermediate
# size: the last 256-wide H tile is a tail whose out-of-range half the TMA loads zero-fill and the TMA
# stores clip, and every H tile count and readiness target counts it (ceil).  The CuTe DSL kernels follow.
# Checked against the independent PyTorch reference (output and every gradient), BF16 and MXFP8, for:
#
# * one macrobatch and several (replayed Gate/Up, owner buckets);
# * MCore's arguments (split per-expert FC1 / FC2, FC1 as Gate and Up, FP32 main_grads in per-expert
#   buffers with NaN gaps: nothing may be written past a tail) and the shared output gate;
# * experts without rows, fresh weight gradients (the BF16 backward's plain zero stores of their dW);
# * an H tail together with an I tail.
#
# H and I are runtime sizes; only whether one of them has a tail selects code (``has_tail``): every 256-aligned
# layer runs one set of executors and every layer with a tail another
# (``test_aligned_sizes_and_tail_sizes_each_share_executors``).
#
# Run under torchrun on eight GPUs.


TOKENS, LOCAL_EXPERTS, TOPK = 512, 2, 2


NAMES = (
    "output",
    "d_x",
    "d_router_weights",
    "d_w_routed_gate",
    "d_w_routed_up",
    "d_w_routed_down",
    "d_w_shared_gate",
    "d_w_shared_up",
    "d_w_shared_down",
    "d_shared_output_gate",
)


SHAPES = [(384, 512), (640, 512), (640, 640)]  # (hidden, intermediate): H tails, and H and I tails together


ALIGNED = [(512, 512), (768, 512)]  # both multiples of 256


def _skip_unless_target(device):
    if torch.cuda.get_device_capability(device) != TARGET_CAPABILITY:
        pytest.skip(f"compiled for {TARGET_ARCH}")


def _config(macrobatch_size):
    return functional.CuteMoKConfig(
        fwd_num_comm_sms=2, bwd_num_comm_sms=2, minibatch_size=256, macrobatch_size=macrobatch_size, schedule_capacity_multiplier=1.5
    )


def _gate_weight(rank, device, hidden):
    g = torch.Generator(device=device).manual_seed(77 + rank)
    return torch.randn(1, hidden, generator=g, device=device, dtype=torch.bfloat16) * hidden**-0.5


def _split_mcore_hidden_tail_arguments(routed, shared, hidden, intermediate):
    """MCore's split arguments with FP32 main_grads as per-expert views of one flat buffer, scrambled and
    with NaN gaps between them (a write past a tail lands in a gap)."""
    gate, up, down = routed
    fc1 = [torch.cat((gate[e], up[e])) for e in range(LOCAL_EXPERTS)]
    fc2 = [down[e].clone() for e in range(LOCAL_EXPERTS)]
    fc1_view = functional.SplitRoutedWeight(data=fc1[0], storage_table=ops.make_routed_weight_storage_table_bf16(fc1))
    fc2_view = functional.SplitRoutedWeight(data=fc2[0], storage_table=ops.make_routed_weight_storage_table_bf16(fc2))
    device = gate.device
    order = [(3 * e + 1) % LOCAL_EXPERTS for e in range(LOCAL_EXPERTS)]
    views = {}
    for name, shape in (("fc1", (2 * intermediate, hidden)), ("fc2", (hidden, intermediate))):
        numel, gap = shape[0] * shape[1], 4096
        flat = torch.full((LOCAL_EXPERTS * (numel + gap),), float("nan"), dtype=torch.float32, device=device)
        experts = [None] * LOCAL_EXPERTS
        for slot, expert in enumerate(order):
            view = flat[slot * (numel + gap) : slot * (numel + gap) + numel].view(shape)
            view.zero_()
            experts[expert] = view
        views[name] = (flat, experts)
    fc1_grads, fc2_grads = views["fc1"][1], views["fc2"][1]
    tables = (ops.make_routed_d_weight_storage_table(fc1_grads),) * 2 + (ops.make_routed_d_weight_storage_table(fc2_grads),)
    shared_fc1 = torch.cat(shared[:2])
    shared_args = (shared_fc1[:intermediate], shared_fc1[intermediate:], shared[2])
    shared_fc1_grad = torch.zeros(2 * intermediate, hidden, dtype=torch.float32, device=device)
    shared_fc2_grad = torch.zeros(hidden, intermediate, dtype=torch.float32, device=device)
    shared_grads = (shared_fc1_grad[:intermediate], shared_fc1_grad[intermediate:], shared_fc2_grad)
    main_grads = (shared_grads[0], fc1_grads[0], shared_grads[1], fc1_grads[0], shared_grads[2], fc2_grads[0])
    return ((*shared_args, fc1_view, fc1_view, fc2_view), main_grads, tables, fc1_grads, fc2_grads, shared_grads, (fc1, fc2, views))


def _check_gaps(keepalive, fc1_grads, fc2_grads):
    for name, (flat, _) in keepalive[2].items():
        written = torch.zeros_like(flat, dtype=torch.bool)
        for view in (fc1_grads if name == "fc1" else fc2_grads):
            offset = view.data_ptr() - flat.data_ptr()
            written[offset // 4 : offset // 4 + view.numel()] = True
        assert torch.isnan(flat[~written]).all(), "a main_grad write landed outside its expert's buffer"


@pytest.mark.parametrize("hidden,intermediate", SHAPES)
@pytest.mark.parametrize(
    "variant,macrobatch_size",
    [
        ("grouped", 4096),  # one macrobatch: dense task decode
        ("grouped", 512),  # several: replay + owner buckets
        ("mcore", 512),  # split + combined FC1 + FP32 main_grads
        ("mcore_gated", 512),  # + shared output gate (Qwen)
    ],
)
def test_hidden_tail_matches_reference(context, hidden, intermediate, variant, macrobatch_size):
    rank, world, device = context
    _skip_unless_target(device)
    inputs = generate_inputs(rank, device, LOCAL_EXPERTS * world, LOCAL_EXPERTS, TOPK, TOKENS, hidden, intermediate)
    x, routes, probs, *rest = inputs
    shared, routed, dy = list(rest[:3]), rest[3:6], rest[6]
    gated = variant == "mcore_gated"
    gate_weight = _gate_weight(rank, device, hidden) if gated else None
    config = _config(macrobatch_size)
    workspace = functional.create_workspace(config, dist.group.WORLD, device=device, num_local_tokens=TOKENS, hidden_size=hidden, topk=TOPK)
    schedule = functional.build_schedule(workspace, config, routes, num_local_experts=LOCAL_EXPERTS)
    fwd_kwargs, bwd_kwargs = {}, {}
    if gated:
        fwd_kwargs["shared_output_gate_weight"] = bwd_kwargs["shared_output_gate_weight"] = gate_weight
    if variant == "grouped":
        weights = (*shared, *routed)
    else:
        weights, main_grads, tables, fc1_grads, fc2_grads, shared_grads, keepalive = _split_mcore_hidden_tail_arguments(routed, shared, hidden, intermediate)
        bwd_kwargs.update(main_grads=main_grads, main_grad_storage_tables=tables)
        if gated:
            bwd_kwargs["shared_output_gate_main_grad"] = torch.zeros(1, hidden, dtype=torch.float32, device=device)
    y, ctx = functional.forward(config, workspace, schedule, x, probs, *weights, **fwd_kwargs)
    grads = functional.backward(config, workspace, schedule, ctx, dy, x, probs, *weights, **bwd_kwargs)
    torch.cuda.synchronize(device)
    golden = run_reference_bf16(x, routes, probs, *shared, *routed, dy, **({"shared_output_gate_weight": gate_weight} if gated else {}))
    if variant == "grouped":
        actual = (y, *grads)
    else:
        i = intermediate
        actual = [
            y,
            grads[0],
            grads[1],
            torch.stack([g[:i] for g in fc1_grads]),
            torch.stack([g[i:] for g in fc1_grads]),
            torch.stack(fc2_grads),
            *shared_grads,
        ]
        if gated:
            actual.append(bwd_kwargs["shared_output_gate_main_grad"])
        _check_gaps(keepalive, fc1_grads, fc2_grads)
    for name, reference, result in zip(NAMES, golden, actual):
        check_correctness(name, reference, result, BF16_TOLERANCE, print_stats=rank == 0)
    dist.barrier()


@pytest.mark.parametrize("hidden,intermediate", SHAPES)
@pytest.mark.parametrize("empty_parity", [0, 1])
def test_hidden_tail_empty_experts(context, hidden, intermediate, empty_parity):
    """Experts without rows, fresh weight gradients: their dW is written with plain zero stores, which
    an H (and I) tail tile must keep inside the matrix (see test_intermediate_tail_empty_experts)."""
    rank, world, device = context
    _skip_unless_target(device)
    # Half of the experts get no route, so TOPK distinct routes need 2 * TOPK experts: more local ones on one GPU.
    local = max(LOCAL_EXPERTS, -(-2 * TOPK // world))
    inputs = generate_inputs(rank, device, local * world, local, TOPK, TOKENS, hidden, intermediate)
    x, routes, probs, *rest = inputs
    shared, routed, dy = list(rest[:3]), rest[3:6], rest[6]
    g = torch.Generator(device=device).manual_seed(91 + rank)
    pool = torch.arange(1 - empty_parity, local * world, 2, device=device)
    routes = pool[torch.rand(TOKENS, pool.numel(), generator=g, device=device).argsort(dim=1)[:, :TOPK]].to(routes.dtype)
    config = _config(512)
    workspace = functional.create_workspace(config, dist.group.WORLD, device=device, num_local_tokens=TOKENS, hidden_size=hidden, topk=TOPK)
    schedule = functional.build_schedule(workspace, config, routes, num_local_experts=local)
    weights = (*shared, *routed)
    y, ctx = functional.forward(config, workspace, schedule, x, probs, *weights)
    grads = functional.backward(config, workspace, schedule, ctx, dy, x, probs, *weights)
    torch.cuda.synchronize(device)
    golden = run_reference_bf16(x, routes, probs, *shared, *routed, dy)
    empty = [e for e in range(local) if (rank * local + e) % 2 == empty_parity]
    for name, matrix in zip(("d_w_routed_gate", "d_w_routed_up", "d_w_routed_down"), grads[2:5]):
        for e in empty:
            assert not matrix[e].any(), f"{name}: expert {e} has no rows, its gradient must be zero"
    for name, reference, result in zip(NAMES, golden, (y, *grads)):
        check_correctness(name, reference, result, BF16_TOLERANCE, print_stats=rank == 0)
    dist.barrier()


@pytest.mark.parametrize("hidden,intermediate", SHAPES)
@pytest.mark.parametrize("layout", ["plain", "mcore"])
def test_mxfp8_hidden_tail_matches_reference(context, hidden, intermediate, layout):
    from .call_helpers import call_arguments

    rank, world, device = context
    _skip_unless_target(device)
    inputs = generate_inputs(rank, device, LOCAL_EXPERTS * world, LOCAL_EXPERTS, TOPK, TOKENS, hidden, intermediate)
    x, routes, probs, *rest = inputs
    shared, routed, dy = list(rest[:3]), rest[3:6], rest[6]
    gate_weight = _gate_weight(rank, device, hidden) if layout == "mcore" else None
    config = _config(512)
    workspace = functional.create_workspace(config, dist.group.WORLD, device=device, num_local_tokens=TOKENS, hidden_size=hidden, topk=TOPK)
    schedule = functional.build_schedule(workspace, config, routes, num_local_experts=LOCAL_EXPERTS)
    fwd_weights, bwd_weights, fwd_kwargs, bwd_kwargs, keepalive = call_arguments(layout, "mxfp8", shared, routed, output_gate=gate_weight)
    y, ctx = functional.forward(config, workspace, schedule, x, probs, *fwd_weights, **fwd_kwargs)
    grads = functional.backward(config, workspace, schedule, ctx, dy, x, probs, *bwd_weights, **bwd_kwargs)
    torch.cuda.synchronize(device)
    golden = run_reference_bf16(x, routes, probs, *shared, *routed, dy, **({"shared_output_gate_weight": gate_weight} if gate_weight is not None else {}))
    if layout == "plain":
        actual = (y, *grads)
    else:
        i, e = intermediate, LOCAL_EXPERTS
        flat = keepalive[-1]
        fc1 = flat[: e * 2 * i * hidden].view(e, 2 * i, hidden)
        fc2 = flat[e * 2 * i * hidden :].view(e, hidden, i)
        main = bwd_kwargs["main_grads"]
        actual = (y, grads[0], grads[1], fc1[:, :i], fc1[:, i:], fc2, main[0], main[2], main[4], bwd_kwargs["shared_output_gate_main_grad"])
    for name, reference, result in zip(NAMES, golden, actual):
        check_correctness(name, reference, result, MXFP8_TOLERANCE, print_stats=rank == 0)
    dist.barrier()


def _executor_keys():
    """The keys of every megakernel executor this process compiled (BF16 and MXFP8 forward and backward, and the
    BF16 backward's metadata builder)."""
    from cudnn.moe_ep.cutemok.cutedsl import _persistent_bf16_bwd_metadata, persistent_bf16_bwd, persistent_bf16_fwd, persistent_mxfp8_bwd, persistent_mxfp8_fwd

    return tuple(
        set(cache)
        for cache in (
            persistent_bf16_fwd._EXECUTOR_CACHE,
            persistent_bf16_bwd._EXECUTOR_CACHE,
            _persistent_bf16_bwd_metadata._CACHE,
            persistent_mxfp8_fwd._EXECUTOR_CACHE,
            persistent_mxfp8_bwd._EXECUTOR_CACHE,
        )
    )


@pytest.mark.parametrize("precision", ["bf16", "mxfp8"])
@pytest.mark.parametrize("macrobatch_size", [4096, 512])
def test_aligned_sizes_and_tail_sizes_each_share_executors(context, precision, macrobatch_size):
    """The hidden and the intermediate size are runtime arguments of the kernels; ``has_tail`` (one of them is not a
    multiple of 256) is the only thing about them that selects code.  Two 256-aligned layers run the same executors;
    layers with a 128-wide H tail, an I tail and both run the same executors, at most one more per kernel and keyed
    ``("has_tail", True)``; an aligned layer after them compiles none; and every layer matches the reference (full
    tiles and the tail tile of one grid in the same launch)."""
    from .call_helpers import call_arguments

    rank, world, device = context
    _skip_unless_target(device)
    config = _config(macrobatch_size)
    tolerance = MXFP8_TOLERANCE if precision == "mxfp8" else BF16_TOLERANCE
    aligned_keys = tail_keys = None
    for hidden, intermediate in (*ALIGNED, *SHAPES, (512, 640), ALIGNED[0]):
        inputs = generate_inputs(rank, device, LOCAL_EXPERTS * world, LOCAL_EXPERTS, TOPK, TOKENS, hidden, intermediate)
        x, routes, probs, *rest = inputs
        shared, routed, dy = list(rest[:3]), rest[3:6], rest[6]
        workspace = functional.create_workspace(config, dist.group.WORLD, device=device, num_local_tokens=TOKENS, hidden_size=hidden, topk=TOPK)
        schedule = functional.build_schedule(workspace, config, routes, num_local_experts=LOCAL_EXPERTS)
        fwd_weights, bwd_weights, _, _, _ = call_arguments("plain", precision, shared, routed)
        y, ctx = functional.forward(config, workspace, schedule, x, probs, *fwd_weights)
        grads = functional.backward(config, workspace, schedule, ctx, dy, x, probs, *bwd_weights)
        torch.cuda.synchronize(device)
        golden = run_reference_bf16(x, routes, probs, *shared, *routed, dy)
        for name, reference, result in zip(NAMES, golden, (y, *grads)):
            check_correctness(f"H{hidden}/I{intermediate}/{name}", reference, result, tolerance, print_stats=rank == 0)
        dist.barrier()
        keys = _executor_keys()
        shape = f"hidden {hidden} / intermediate {intermediate}"
        if aligned_keys is None:
            aligned_keys = keys
        elif tail_keys is None and hidden % 256 == 0 and intermediate % 256 == 0:
            assert keys == aligned_keys, f"{shape} compiled executors of its own"
        elif tail_keys is None:
            # The first layer with a tail: at most one executor more per kernel (none where an earlier test
            # compiled it), and what selects it is has_tail.  The metadata builder's key carries the bare flag.
            tail_keys = keys
            compiled = [after - before for before, after in zip(aligned_keys, keys)]
            assert all(len(new) <= 1 for new in compiled), f"{shape} compiled more than one executor per kernel"
            megakernels = compiled[:2] + compiled[3:]
            assert all(("has_tail", True) in key for new in megakernels for key in new), f"{shape} compiled an executor that is not the has_tail one"
        else:
            assert keys == tail_keys, f"{shape} compiled executors of its own"


# --------------------------------------------------------------------------------------------------------------------
# Intermediate tail
#
# A 128-aligned intermediate size (I % 256 != 0), e.g. I = 640.
#
# The CUDA kernels admit I >= 256 with I % 128 == 0 since mcore-integration #5; the BF16 CuTe DSL
# kernels follow (the forward tiles I at 128 already, the backward counts a predicated 256-wide tail
# tile).  Checked against the independent PyTorch reference (output and every gradient) for the
# paths the MCore integration takes on top of the plain one:
#
# * one macrobatch (the backward's dense task decode) and several (replayed Gate/Up and the owner
#   buckets the metadata kernel builds -- its per-minibatch task count must count the tail tile);
# * MCore's arguments: per-expert FC1 / FC2 allocations behind SplitRoutedWeight, FC1 passed as both
#   Gate and Up (Up starts I rows into it, i.e. mid-way through a 256-row tile), and FP32 main_grads
#   in per-expert buffers -- scrambled, with gaps -- so the Gate tail tile reduce-adds into the
#   buffer's Up rows (zeros) and nothing may overwrite them;
# * the shared output gate (Qwen) and routed-only with SiTU-GLU.
#
# Grouped weights accumulating into grouped FP32 main_grads are checked too, the weight and the main_grad each
# one combined FC1 buffer or separate Gate and Up ones (a Gate tail tile's rows past I are Up's first rows of a
# combined buffer).
#
# The MXFP8 path handles the tail too (as the C++ kernels do), checked at the MXFP8 tolerance.
#
# Run under torchrun on eight GPUs.


TOKENS, HIDDEN, LOCAL_EXPERTS, TOPK = 512, 512, 2, 2


def _split_mcore_arguments(routed, intermediate, has_shared, shared):
    """MCore's split arguments: every expert's FC1 [2I, H] and FC2 [H, I] its own allocation, FC1
    as both Gate and Up, and FP32 main_grads as per-expert views of one flat buffer in a scrambled
    order with gaps (a kernel assuming a uniform stride, or writing past a tail, shows up)."""
    gate, up, down = routed
    fc1 = [torch.cat((gate[e], up[e])) for e in range(LOCAL_EXPERTS)]
    fc2 = [down[e].clone() for e in range(LOCAL_EXPERTS)]
    fc1_view = functional.SplitRoutedWeight(data=fc1[0], storage_table=ops.make_routed_weight_storage_table_bf16(fc1))
    fc2_view = functional.SplitRoutedWeight(data=fc2[0], storage_table=ops.make_routed_weight_storage_table_bf16(fc2))
    device = gate.device
    order = [(3 * e + 1) % LOCAL_EXPERTS for e in range(LOCAL_EXPERTS)]
    views = {}
    for name, shape in (("fc1", (2 * intermediate, HIDDEN)), ("fc2", (HIDDEN, intermediate))):
        numel, gap = shape[0] * shape[1], 4096
        flat = torch.full((LOCAL_EXPERTS * (numel + gap),), float("nan"), dtype=torch.float32, device=device)
        experts = [None] * LOCAL_EXPERTS
        for slot, expert in enumerate(order):
            view = flat[slot * (numel + gap) : slot * (numel + gap) + numel].view(shape)
            view.zero_()
            experts[expert] = view
        views[name] = (flat, experts)
    fc1_grads, fc2_grads = views["fc1"][1], views["fc2"][1]
    fc1_table = ops.make_routed_d_weight_storage_table(fc1_grads)
    fc2_table = ops.make_routed_d_weight_storage_table(fc2_grads)
    if has_shared:
        shared_fc1 = torch.cat(shared[:2])
        shared_args = (shared_fc1[:intermediate], shared_fc1[intermediate:], shared[2])
        shared_fc1_grad = torch.zeros(2 * intermediate, HIDDEN, dtype=torch.float32, device=device)
        shared_fc2_grad = torch.zeros(HIDDEN, intermediate, dtype=torch.float32, device=device)
        shared_grads = (shared_fc1_grad[:intermediate], shared_fc1_grad[intermediate:], shared_fc2_grad)
    else:
        shared_args = shared_grads = (None, None, None)
    main_grads = (shared_grads[0], fc1_grads[0], shared_grads[1], fc1_grads[0], shared_grads[2], fc2_grads[0])
    keepalive = (fc1, fc2, views)
    return ((*shared_args, fc1_view, fc1_view, fc2_view), main_grads, (fc1_table, fc1_table, fc2_table), fc1_grads, fc2_grads, shared_grads, keepalive)


@pytest.mark.parametrize("intermediate", [640, 384])
@pytest.mark.parametrize("empty_parity", [0, 1])
@pytest.mark.parametrize("combined", [False, True])
def test_intermediate_tail_empty_experts(context, intermediate, empty_parity, combined):
    """Experts without rows, fresh weight gradients (no main_grads): the backward writes their dW as
    zeros with plain stores, which a tail tile (I % 256 != 0) must keep inside the matrix -- past I
    lie the next expert's rows (Gate / Up), the next row (Down), or the end of the buffer for the last
    expert.  Every local expert of one parity gets no route; Gate and Up separate or one combined FC1."""
    rank, world, device = context
    _skip_unless_target(device)
    # Half of the experts get no route, so TOPK distinct routes need 2 * TOPK experts: more local ones on one GPU.
    local = max(LOCAL_EXPERTS, -(-2 * TOPK // world))
    inputs = generate_inputs(rank, device, local * world, local, TOPK, TOKENS, HIDDEN, intermediate)
    x, routes, probs, *rest = inputs
    shared, routed, dy = list(rest[:3]), rest[3:6], rest[6]
    # Routes to experts of the other parity only, TOPK distinct ones per token.
    g = torch.Generator(device=device).manual_seed(91 + rank)
    pool = torch.arange(1 - empty_parity, local * world, 2, device=device)
    choice = torch.rand(TOKENS, pool.numel(), generator=g, device=device).argsort(dim=1)[:, :TOPK]
    routes = pool[choice].to(routes.dtype)
    config = _config(512)
    workspace = functional.create_workspace(config, dist.group.WORLD, device=device, num_local_tokens=TOKENS, hidden_size=HIDDEN, topk=TOPK)
    schedule = functional.build_schedule(workspace, config, routes, num_local_experts=local)
    gate, up, down = routed
    if combined:
        fc1 = torch.cat((gate, up), dim=1)  # [E, 2I, H] passed as both Gate and Up
        weights = (*shared, fc1, fc1, down)
    else:
        weights = (*shared, gate, up, down)
    y, ctx = functional.forward(config, workspace, schedule, x, probs, *weights)
    grads = functional.backward(config, workspace, schedule, ctx, dy, x, probs, *weights)
    torch.cuda.synchronize(device)
    golden = run_reference_bf16(x, routes, probs, *shared, gate, up, down, dy)
    d_gate, d_up = grads[2], grads[3]
    if d_gate.shape[1] == 2 * intermediate:  # one combined FC1 gradient under both names
        d_gate, d_up = d_gate[:, :intermediate], d_up[:, intermediate:]
    actual = (y, grads[0], grads[1], d_gate, d_up, grads[4], *grads[5:8])
    empty = [e for e in range(local) if (rank * local + e) % 2 == empty_parity]
    for name, matrix in (("d_w_routed_gate", d_gate), ("d_w_routed_up", d_up), ("d_w_routed_down", grads[4])):
        for e in empty:
            assert not matrix[e].any(), f"{name}: expert {e} has no rows, its gradient must be zero"
    for name, reference, result in zip(NAMES, golden, actual):
        check_correctness(name, reference, result, BF16_TOLERANCE, print_stats=rank == 0)
    dist.barrier()


@pytest.mark.parametrize("intermediate", [640, 384])
@pytest.mark.parametrize(
    "variant,macrobatch_size",
    [
        ("grouped", 4096),  # one macrobatch: dense task decode
        ("grouped", 512),  # several: replay + owner buckets
        ("mcore", 512),  # split + combined FC1 + FP32 main_grads
        ("mcore_gated", 512),  # + shared output gate (Qwen)
        ("mcore_routed_only_situ", 512),  # routed-only + SiTU-GLU
    ],
)
def test_intermediate_tail_matches_reference(context, intermediate, variant, macrobatch_size):
    rank, world, device = context
    _skip_unless_target(device)
    inputs = generate_inputs(rank, device, LOCAL_EXPERTS * world, LOCAL_EXPERTS, TOPK, TOKENS, HIDDEN, intermediate)
    x, routes, probs, *rest = inputs
    shared, routed, dy = list(rest[:3]), rest[3:6], rest[6]
    gated = variant == "mcore_gated"
    has_shared = variant != "mcore_routed_only_situ"
    situ = None if has_shared else (2.0, 8.0)
    gate_weight = None
    if gated:
        g = torch.Generator(device=device).manual_seed(77 + rank)
        gate_weight = torch.randn(1, HIDDEN, generator=g, device=device, dtype=torch.bfloat16) * HIDDEN**-0.5
    if not has_shared:
        for weight in shared:
            weight.zero_()  # the reference's shared expert contributes nothing

    config = _config(macrobatch_size)
    workspace = functional.create_workspace(config, dist.group.WORLD, device=device, num_local_tokens=TOKENS, hidden_size=HIDDEN, topk=TOPK)
    schedule = functional.build_schedule(workspace, config, routes, num_local_experts=LOCAL_EXPERTS)
    rows = torch.tensor(int(schedule.num_tokens[0]), device=device)
    dist.all_reduce(rows, op=dist.ReduceOp.MAX)
    if macrobatch_size < 4096:
        assert int(rows) > macrobatch_size  # the multi-macrobatch decode really runs somewhere

    fwd_kwargs, bwd_kwargs = {}, {}
    if gated:
        fwd_kwargs["shared_output_gate_weight"] = bwd_kwargs["shared_output_gate_weight"] = gate_weight
    if situ is not None:
        fwd_kwargs["situ_beta"] = situ
    if variant == "grouped":
        weights = (*shared, *routed)
    else:
        weights, main_grads, tables, fc1_grads, fc2_grads, shared_grads, keepalive = _split_mcore_arguments(routed, intermediate, has_shared, shared)
        bwd_kwargs.update(main_grads=main_grads, main_grad_storage_tables=tables)
        if gated:
            bwd_kwargs["shared_output_gate_main_grad"] = torch.zeros(1, HIDDEN, dtype=torch.float32, device=device)

    y, ctx = functional.forward(config, workspace, schedule, x, probs, *weights, **fwd_kwargs)
    grads = functional.backward(config, workspace, schedule, ctx, dy, x, probs, *weights, **bwd_kwargs)
    torch.cuda.synchronize(device)

    reference_kwargs = {}
    if gated:
        reference_kwargs["shared_output_gate_weight"] = gate_weight
    if situ is not None:

        def activation(a, u):
            a, u = a.float(), u.float()
            return (situ[0] * torch.tanh(a / situ[0]) * torch.sigmoid(a) * situ[1] * torch.tanh(u / situ[1])).bfloat16()

        reference_kwargs["routed_activation"] = activation
    golden = run_reference_bf16(x, routes, probs, *shared, *routed, dy, **reference_kwargs)

    if variant == "grouped":
        actual = (y, *grads)
    else:
        i = intermediate
        actual = [
            y,
            grads[0],
            grads[1],
            torch.stack([g[:i] for g in fc1_grads]),
            torch.stack([g[i:] for g in fc1_grads]),
            torch.stack(fc2_grads),
            *shared_grads,
        ]
        if gated:
            actual.append(bwd_kwargs["shared_output_gate_main_grad"])
        # The scrambled flat buffers' gaps were NaN-filled: nothing may have been written there.
        for flat, _ in keepalive[2].values():
            written = torch.zeros_like(flat, dtype=torch.bool)
            for view in (fc1_grads if flat is keepalive[2]["fc1"][0] else fc2_grads):
                offset = view.data_ptr() - flat.data_ptr()
                written[offset // 4 : offset // 4 + view.numel()] = True
            assert torch.isnan(flat[~written]).all(), "a main_grad write landed outside its expert's buffer"
    for name, reference, result in zip(NAMES, golden, actual):
        if result is None:
            assert not has_shared and name.startswith("d_w_shared")
            continue
        check_correctness(name, reference, result, BF16_TOLERANCE, print_stats=rank == 0)
    dist.barrier()


@pytest.mark.parametrize("intermediate", [640, 384])
@pytest.mark.parametrize(
    "variant,macrobatch_size",
    [
        ("grouped", 4096),  # one macrobatch
        ("grouped", 512),  # several: replayed Gate / Up / SwiGLU, owner buckets
        ("mcore", 512),  # split, TE-native columnwise, combined FC1, FP32 main_grads
        ("mcore_gated", 512),  # + shared output gate (Qwen)
        ("mcore_routed_only_situ", 512),  # routed-only + SiTU-GLU
    ],
)
def test_mxfp8_intermediate_tail_matches_reference(context, intermediate, variant, macrobatch_size):
    """The MXFP8 path at I % 256 != 0 (C++ mcore-integration #5 runs MXFP8 tails too): the
    shared expert's 256-wide tail tiles, the routed tiles' tail half (its B / A scale half-block
    out of range, its quantized replay output's absent scale tile) -- against the BF16 reference
    at the MXFP8 tolerance."""
    from .call_helpers import call_arguments
    from .utils import MXFP8_TOLERANCE

    rank, world, device = context
    _skip_unless_target(device)
    inputs = generate_inputs(rank, device, LOCAL_EXPERTS * world, LOCAL_EXPERTS, TOPK, TOKENS, HIDDEN, intermediate)
    x, routes, probs, *rest = inputs
    shared, routed, dy = list(rest[:3]), rest[3:6], rest[6]
    gated = variant == "mcore_gated"
    has_shared = variant != "mcore_routed_only_situ"
    situ = None if has_shared else (2.0, 8.0)
    gate_weight = None
    if gated:
        g = torch.Generator(device=device).manual_seed(77 + rank)
        gate_weight = torch.randn(1, HIDDEN, generator=g, device=device, dtype=torch.bfloat16) * HIDDEN**-0.5
    if not has_shared:
        for weight in shared:
            weight.zero_()
    config = _config(macrobatch_size)
    workspace = functional.create_workspace(config, dist.group.WORLD, device=device, num_local_tokens=TOKENS, hidden_size=HIDDEN, topk=TOPK)
    schedule = functional.build_schedule(workspace, config, routes, num_local_experts=LOCAL_EXPERTS)
    rows = torch.tensor(int(schedule.num_tokens[0]), device=device)
    dist.all_reduce(rows, op=dist.ReduceOp.MAX)
    if macrobatch_size < 4096:
        assert int(rows) > macrobatch_size

    layout = "plain" if variant == "grouped" else "mcore"
    fwd_weights, bwd_weights, fwd_kwargs, bwd_kwargs, keepalive = call_arguments(layout, "mxfp8", shared, routed, output_gate=gate_weight)
    if not has_shared:
        fwd_weights = bwd_weights = (None, None, None, *fwd_weights[3:])
        main = list(bwd_kwargs["main_grads"])
        main[0] = main[2] = main[4] = None
        bwd_kwargs["main_grads"] = tuple(main)
    if situ is not None:
        fwd_kwargs["situ_beta"] = situ
    y, ctx = functional.forward(config, workspace, schedule, x, probs, *fwd_weights, **fwd_kwargs)
    grads = functional.backward(config, workspace, schedule, ctx, dy, x, probs, *bwd_weights, **bwd_kwargs)
    torch.cuda.synchronize(device)

    reference_kwargs = {}
    if gated:
        reference_kwargs["shared_output_gate_weight"] = gate_weight
    if situ is not None:

        def activation(a, u):
            a, u = a.float(), u.float()
            return (situ[0] * torch.tanh(a / situ[0]) * torch.sigmoid(a) * situ[1] * torch.tanh(u / situ[1])).bfloat16()

        reference_kwargs["routed_activation"] = activation
    golden = run_reference_bf16(x, routes, probs, *shared, *routed, dy, **reference_kwargs)
    if layout == "plain":
        actual = (y, *grads)
    else:
        i = intermediate
        flat = keepalive[-1]
        fc1_size = 2 * i * HIDDEN
        fc1 = flat[: LOCAL_EXPERTS * fc1_size].view(LOCAL_EXPERTS, 2 * i, HIDDEN)
        fc2 = flat[LOCAL_EXPERTS * fc1_size :].view(LOCAL_EXPERTS, HIDDEN, i)
        main = bwd_kwargs["main_grads"]
        actual = [y, grads[0], grads[1], fc1[:, :i], fc1[:, i:], fc2, main[0], main[2], main[4]]
        if gated:
            actual.append(bwd_kwargs["shared_output_gate_main_grad"])
    for name, reference, result in zip(NAMES, golden, actual):
        if result is None:
            assert not has_shared and name.startswith("d_w_shared")
            continue
        check_correctness(name, reference, result, MXFP8_TOLERANCE, print_stats=rank == 0)
    dist.barrier()


@pytest.mark.parametrize("intermediate", [640, 384])
@pytest.mark.parametrize("precision", ["bf16", "mxfp8"])
@pytest.mark.parametrize("macrobatch_size", [4096, 512])
def test_combined_fc1_fresh_gradients(context, intermediate, precision, macrobatch_size):
    """One combined FC1 [E, 2I, H] passed as both Gate and Up, fresh weight gradients (no main_grads): the
    backward writes Gate and Up into one [E, 2I, H] buffer, and a Gate tail tile (I % 256 != 0) must not
    store over Up's first rows (accumulating into main_grad, as MCore does, adds zeros there instead)."""
    from .utils import MXFP8_TOLERANCE

    rank, world, device = context
    _skip_unless_target(device)
    inputs = generate_inputs(rank, device, LOCAL_EXPERTS * world, LOCAL_EXPERTS, TOPK, TOKENS, HIDDEN, intermediate)
    x, routes, probs, *rest = inputs
    shared, (gate, up, down), dy = list(rest[:3]), rest[3:6], rest[6]
    config = _config(macrobatch_size)
    workspace = functional.create_workspace(config, dist.group.WORLD, device=device, num_local_tokens=TOKENS, hidden_size=HIDDEN, topk=TOPK)
    schedule = functional.build_schedule(workspace, config, routes, num_local_experts=LOCAL_EXPERTS)
    fc1 = torch.cat((gate, up), dim=1)
    if precision == "bf16":
        fwd_routed = bwd_routed = (fc1, fc1, down)
        tolerance = BF16_TOLERANCE
    else:
        fc1_q = tuple(ops.mxfp8_quantize(fc1, True, True))
        down_q = tuple(ops.mxfp8_quantize(down, True, True))
        fc1_fwd = fc1_q[:2]
        fwd_routed = (fc1_fwd, fc1_fwd, down_q[:2])
        bwd_routed = (fc1_q, fc1_q, down_q[2:])
        tolerance = MXFP8_TOLERANCE
    y, ctx = functional.forward(config, workspace, schedule, x, probs, *shared, *fwd_routed)
    grads = functional.backward(config, workspace, schedule, ctx, dy, x, probs, *shared, *bwd_routed)
    torch.cuda.synchronize(device)
    golden = run_reference_bf16(x, routes, probs, *shared, gate, up, down, dy)
    d_gate, d_up = grads[2], grads[3]
    if d_gate.shape[1] == 2 * intermediate:  # one combined FC1 gradient under both names
        d_gate, d_up = d_gate[:, :intermediate], d_up[:, intermediate:]
    actual = (y, grads[0], grads[1], d_gate, d_up, *grads[4:8])
    for name, reference, result in zip(NAMES, golden, actual):
        check_correctness(f"{precision}/{name}", reference, result, tolerance, print_stats=rank == 0)
    dist.barrier()


def _guarded(generator, *shape):
    """A float32 buffer of ``shape`` seeded with values in [-2, 2), the first elements of a flat allocation whose
    remainder is NaN: a write past the buffer's end lands there.  Returns the buffer and the remainder."""
    numel = 1
    for extent in shape:
        numel *= extent
    flat = torch.full((numel + 4096,), float("nan"), dtype=torch.float32, device=generator.device)
    flat[:numel] = (torch.rand(numel, generator=generator, device=generator.device) - 0.5) * 4
    return flat[:numel].view(shape), flat[numel:]


@pytest.mark.parametrize("intermediate", [640, 384])
@pytest.mark.parametrize("precision", ["bf16", "mxfp8"])
@pytest.mark.parametrize(
    "weights,main_grad",
    [
        ("combined", "combined"),  # one [E, 2I, H] weight and one [E, 2I, H] main_grad, each under both names
        ("separate", "combined"),  # Gate and Up weights of their own, one combined main_grad
        ("combined", "separate"),  # one combined weight, Gate and Up main_grads of their own
    ],
)
def test_grouped_main_grads_with_an_intermediate_tail(context, intermediate, precision, weights, main_grad):
    """Grouped routed weights accumulating into grouped FP32 main_grads at I % 256 != 0, several macrobatches.

    The rows a Gate tail tile has past I are Up's first rows in a combined buffer, inside the buffer's bounds:
    a combined weight holds real Up data there, which the replayed Gate must not store, and a combined main_grad
    may only have zeros added there.  The main_grads are seeded and end in a NaN remainder, so a store over
    Up's rows, a lost or misplaced contribution and a write past a buffer all show against the reference."""
    from .utils import MXFP8_TOLERANCE

    rank, world, device = context
    _skip_unless_target(device)
    inputs = generate_inputs(rank, device, LOCAL_EXPERTS * world, LOCAL_EXPERTS, TOPK, TOKENS, HIDDEN, intermediate)
    x, routes, probs, *rest = inputs
    shared, (gate, up, down), dy = list(rest[:3]), rest[3:6], rest[6]
    config = _config(512)
    workspace = functional.create_workspace(config, dist.group.WORLD, device=device, num_local_tokens=TOKENS, hidden_size=HIDDEN, topk=TOPK)
    schedule = functional.build_schedule(workspace, config, routes, num_local_experts=LOCAL_EXPERTS)
    rows = torch.tensor(int(schedule.num_tokens[0]), device=device)
    dist.all_reduce(rows, op=dist.ReduceOp.MAX)
    assert int(rows) > 512  # a later macrobatch replays Gate / Up somewhere

    if weights == "combined":
        gate_weight = up_weight = torch.cat((gate, up), dim=1)  # [E, 2I, H] passed as both Gate and Up
    else:
        gate_weight, up_weight = gate, up
    if precision == "bf16":
        fwd_routed = bwd_routed = (gate_weight, up_weight, down)
        tolerance = BF16_TOLERANCE
    else:
        gate_q = tuple(ops.mxfp8_quantize(gate_weight, True, True))
        up_q = gate_q if up_weight is gate_weight else tuple(ops.mxfp8_quantize(up_weight, True, True))
        down_q = tuple(ops.mxfp8_quantize(down, True, True))
        gate_fwd = gate_q[:2]
        up_fwd = gate_fwd if up_q is gate_q else up_q[:2]
        fwd_routed = (gate_fwd, up_fwd, down_q[:2])
        bwd_routed = (gate_q, up_q, down_q[2:])
        tolerance = MXFP8_TOLERANCE

    g = torch.Generator(device=device).manual_seed(53 + rank)
    if main_grad == "combined":
        gate_grad, fc1_rest = _guarded(g, LOCAL_EXPERTS, 2 * intermediate, HIDDEN)
        up_grad, remainders = gate_grad, [fc1_rest]
    else:
        gate_grad, gate_rest = _guarded(g, LOCAL_EXPERTS, intermediate, HIDDEN)
        up_grad, up_rest = _guarded(g, LOCAL_EXPERTS, intermediate, HIDDEN)
        remainders = [gate_rest, up_rest]
    down_grad, down_rest = _guarded(g, LOCAL_EXPERTS, HIDDEN, intermediate)
    remainders.append(down_rest)
    shared_grads = []
    for shape in ((intermediate, HIDDEN), (intermediate, HIDDEN), (HIDDEN, intermediate)):
        buffer, rest_of_buffer = _guarded(g, *shape)
        shared_grads.append(buffer)
        remainders.append(rest_of_buffer)
    main_grads = (shared_grads[0], gate_grad, shared_grads[1], up_grad, shared_grads[2], down_grad)
    seeds = tuple(buffer.clone() for buffer in main_grads)

    y, ctx = functional.forward(config, workspace, schedule, x, probs, *shared, *fwd_routed)
    grads = functional.backward(config, workspace, schedule, ctx, dy, x, probs, *shared, *bwd_routed, main_grads=main_grads)
    torch.cuda.synchronize(device)
    golden = run_reference_bf16(x, routes, probs, *shared, gate, up, down, dy)

    # What the backward added to each buffer; a combined buffer holds Gate in its first I rows, Up below.
    added = [buffer - seed for buffer, seed in zip(main_grads, seeds)]
    d_gate, d_up = added[1], added[3]
    if main_grad == "combined":
        d_gate, d_up = d_gate[:, :intermediate], d_up[:, intermediate:]
    actual = (y, grads[0], grads[1], d_gate, d_up, added[5], added[0], added[2], added[4])
    for name, reference, result in zip(NAMES, golden, actual):
        check_correctness(f"{precision}/{weights} weights/{main_grad} main_grad/{name}", reference, result, tolerance, print_stats=rank == 0)
    for remainder in remainders:
        assert torch.isnan(remainder).all(), "a main_grad write landed past its buffer"
    dist.barrier()


# --------------------------------------------------------------------------------------------------------------------
# Communication tile shapes
#
# The communication tile shape of the BF16 megakernels moves bytes only: 32 rows x 512 columns and
# 16 rows x 1024 columns give the same bits.
#
# The shape (contract ``CommTileShape``, one of ``COMM_TILE_SHAPES``) is the compile-time parameter of the
# communication CTAs of the forward and the backward megakernel, selected by ``CuteMoKConfig.comm_tile_shape``.
# Here one forward + backward runs per shape, the shape set through that config field, over the same inputs: the
# output and every gradient must be bit-identical, each shape must have compiled its own forward and backward
# executor, and the 16 x 1024 results are also checked against the PyTorch reference.  The two runs share one
# workspace and the caching allocator, so a row one shape fails to move could pass on the bytes the other left
# behind: before each run the peer-written workspace buffers and every fresh forward output and backward scratch /
# gradient buffer are filled with a finite value that differs between the shapes.  H = 4096 (whole column blocks
# in both shapes) and H = 1152 (a 128-column last block in both), one macrobatch and several (X replay, pushes mixed
# with pulls), and the forward with the combine in the routed Down epilogue (the stream's pulls only).
#
# Run under torchrun on eight GPUs.


TOKENS, LOCAL_EXPERTS, TOPK, INTERMEDIATE = 512, 2, 2, 512


COMM_TILE_NAMES = (
    "output",
    "d_x",
    "d_router_weights",
    "d_w_routed_gate",
    "d_w_routed_up",
    "d_w_routed_down",
    "d_w_shared_gate",
    "d_w_shared_up",
    "d_w_shared_down",
)


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
    for name, expected, actual in zip(COMM_TILE_NAMES, default, wide):
        assert (expected is None) == (actual is None), name
        if expected is None:
            continue
        assert expected.dtype == actual.dtype and expected.shape == actual.shape, name
        different = int((expected.contiguous().view(-1).view(torch.uint8) != actual.contiguous().view(-1).view(torch.uint8)).sum())
        assert different == 0, f"{name}: {different} bytes differ between the tile shapes {COMM_TILE_SHAPES}"
    golden = run_reference_bf16(x, routes, probs, *shared, *routed, dy)
    for name, reference, result in zip(COMM_TILE_NAMES, golden, wide):
        if result is None:
            continue
        check_correctness(name, reference, result, BF16_TOLERANCE, print_stats=rank == 0)
    dist.barrier()
