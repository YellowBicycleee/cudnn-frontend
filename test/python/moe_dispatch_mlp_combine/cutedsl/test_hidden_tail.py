# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""A 128-aligned hidden size (H % 256 != 0), e.g. H = 2688 = 21 x 128.

The CUDA kernels admit H >= 256 with H % 128 == 0 since mcore-integration #5, as for the intermediate
size: the last 256-wide H tile is a tail whose out-of-range half the TMA loads zero-fill and the TMA
stores clip, and every H tile count and readiness target counts it (ceil).  The CuTe DSL kernels follow.
Checked against the independent PyTorch reference (output and every gradient), BF16 and MXFP8, for:

* one macrobatch and several (replayed Gate/Up, owner buckets);
* MCore's arguments (split per-expert FC1 / FC2, FC1 as Gate and Up, FP32 main_grads in per-expert
  buffers with NaN gaps: nothing may be written past a tail) and the shared output gate;
* experts without rows, fresh weight gradients (the BF16 backward's plain zero stores of their dW);
* an H tail together with an I tail.

H and I are runtime sizes; only whether one of them has a tail selects code (``has_tail``): every 256-aligned
layer runs one set of executors and every layer with a tail another
(``test_aligned_sizes_and_tail_sizes_each_share_executors``).

Run under torchrun on eight GPUs.
"""

import pytest
import torch
import torch.distributed as dist

from cudnn.moe_dispatch_mlp_combine import functional, ops
from cudnn.moe_dispatch_mlp_combine.cutedsl._arch import TARGET_ARCH, TARGET_CAPABILITY
from .utils import BF16_TOLERANCE, MXFP8_TOLERANCE, check_correctness, generate_inputs, run_reference_bf16

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


def _split_mcore_arguments(routed, shared, hidden, intermediate):
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
        weights, main_grads, tables, fc1_grads, fc2_grads, shared_grads, keepalive = _split_mcore_arguments(routed, shared, hidden, intermediate)
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
    from cudnn.moe_dispatch_mlp_combine.cutedsl import (
        _persistent_bf16_bwd_metadata,
        persistent_bf16_bwd,
        persistent_bf16_fwd,
        persistent_mxfp8_bwd,
        persistent_mxfp8_fwd,
    )

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
