# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""GPU-only EP8 MXFP8 backward through its low-level entry and the public one.

Run under torchrun on SM100 / SM103 (8 GPUs), for both Gate / Up context precisions
(``gate_up_context="mxfp8"``: e4m3 / scale pairs; ``"bf16"``: one BF16 tensor each).  Each geometry
runs the MXFP8 forward for its context, then ``backward_mxfp8`` on a fresh copy of that context
(the replay overwrites its routed part) and the public ``functional.backward``.  Every raw output
and the two symmetric buffers the peers write must be written and finite, the communication
counters must hold the arrivals the kernel's waits expect, and the public gradients must be as
close to the BF16 PyTorch reference as upstream Mixture-of-Kittens' (``MIN_COSINE``, ``NOT_WORSE``,
``UPSTREAM``; validate/README.md).  The latent layer is compared byte for byte
with the ordinary backward of each hidden, a repeated call with itself, and two layers launched
back to back each with its own reference.  The forward's geometries run through one compiled
executor per clamp mode, with a clamp limit the inputs reach.  ``recompute_forward_context`` (macrobatch 0 replayed
instead of read from the forward's routed context) is compared byte for byte with the backward on the
saved context.
"""

import dataclasses
from unittest import mock

import pytest
import torch
import torch.distributed as dist

from cudnn.moe_dispatch_mlp_combine import functional as f
from cudnn.moe_dispatch_mlp_combine.cutedsl import persistent_mxfp8_bwd as mxfp8_bwd_host
from cudnn.moe_dispatch_mlp_combine.cutedsl import persistent_mxfp8_fwd as mxfp8_fwd_host
from cudnn.moe_dispatch_mlp_combine.ops import (
    make_routed_d_weight_storage_table,
    make_routed_scale_storage_table,
    make_routed_weight_storage_table_mxfp8,
    mxfp8_quantize,
)
from cudnn.moe_dispatch_mlp_combine.cutedsl._arch import SUPPORTED_CAPABILITIES
from .test_cutedsl_mxfp8_fwd import CLAMP_LIMIT, GEOMETRIES, LATENT_GEOMETRIES, _bytes, _sync_ranks
from .utils import generate_inputs, run_reference_bf16, ungated

# The return order (persistent_mxfp8_bwd._OUTPUTS).
RAW_OUTPUTS = (
    "d_x_shared",
    "d_x_routed",
    "d_gate_shared",
    "d_gate_fp8_routed",
    "d_gate_sc_routed",
    "d_up_shared",
    "d_up_fp8_routed",
    "d_up_sc_routed",
    "d_hidden_shared",
    "d_hidden_routed",
    "d_y_fp8_routed",
    "d_y_sc_routed",
    "d_w_shared_gate",
    "d_w_routed_gate",
    "d_w_shared_up",
    "d_w_routed_up",
    "d_w_shared_down",
    "d_w_routed_down",
)
GRADIENTS = ("d_x", "d_router_weights", "d_w_routed_gate", "d_w_routed_up", "d_w_routed_down", "d_w_shared_gate", "d_w_shared_up", "d_w_shared_down")
# The transposed e4m3 d_gate / d_up and their scales (state buffers, not among the eighteen outputs).
TRANSPOSED = ("d_gate_fp8_t_routed", "d_gate_sc_t_routed", "d_up_fp8_t_routed", "d_up_sc_t_routed")
# The public gradients against the BF16 reference's.  Every gradient: cosine at least MIN_COSINE.  The four the
# routed Gate / Up context reaches, with either context precision: not worse than upstream Mixture-of-Kittens --
# relative L1 at most NOT_WORSE times upstream's (+ 1e-9), cosine at least the lower of MIN_COSINE and upstream's
# (less COSINE_ROUNDING, for the float32 dot products the cosine is computed with).
MIN_COSINE = 0.996
NOT_WORSE = 1.01
COSINE_ROUNDING = 1e-6
GATE_UP_GRADIENTS = ("d_x", "d_router_weights", "d_w_routed_gate", "d_w_routed_up")
# A geometry whose routed tokens fit one macrobatch: the rank's capacity (8 ranks x 512 tokens x topk 2
# x 1.5 / 8 ranks) is below the macrobatch of 4096 rows, so no replay.
SINGLE_MACROBATCH_GEOMETRY = (16, 512, 512, 2, 512, 8, 512, 4096)
# Upstream Mixture-of-Kittens (cursor/mixture-of-kittens 22fc95a, CUDA) on these tests' inputs (GB300, 8 ranks):
# (relative L1, cosine) to the BF16 reference of GATE_UP_GRADIENTS, in that order, per (geometry, clamp limit).
# With the e4m3 Gate / Up context this kernel returns upstream's bytes, so it meets these values with no margin
# to spare; the cosines are rounded down to six decimals.  At a clamp limit the inputs reach (CLAMP_LIMIT)
# upstream's cosines of d_x, d_w_routed_gate and d_w_routed_up are below MIN_COSINE (0.968 for d_w_routed_gate),
# and nowhere else: those entries are the only ones that lower a bar.  validate/README.md has the measurement,
# what is known of the cause, and the scripts that repeat it.
UPSTREAM = {
    (GEOMETRIES[0], None): ((0.044117, 0.999020), (0.079384, 0.996857), (0.074759, 0.997199), (0.073451, 0.997295)),
    (GEOMETRIES[0], CLAMP_LIMIT): ((0.140403, 0.989846), (0.076119, 0.997098), (0.248848, 0.967853), (0.216087, 0.975870)),
    (GEOMETRIES[1], None): ((0.034994, 0.999382), (0.078103, 0.996945), (0.074776, 0.997195), (0.073336, 0.997300)),
    (GEOMETRIES[1], CLAMP_LIMIT): ((0.112394, 0.993597), (0.075942, 0.997107), (0.247250, 0.967804), (0.214485, 0.976036)),
    (GEOMETRIES[2], None): ((0.043986, 0.999026), (0.078109, 0.996968), (0.074721, 0.997198), (0.073316, 0.997302)),
    (GEOMETRIES[2], CLAMP_LIMIT): ((0.139918, 0.989919), (0.075357, 0.997129), (0.247211, 0.968130), (0.215149, 0.976082)),
    (GEOMETRIES[3], None): ((0.043904, 0.999030), (0.078151, 0.996953), (0.074704, 0.997204), (0.073254, 0.997307)),
    (GEOMETRIES[3], CLAMP_LIMIT): ((0.140273, 0.989865), (0.075937, 0.997102), (0.248945, 0.967798), (0.215489, 0.976027)),
    (GEOMETRIES[4], None): ((0.043986, 0.999026), (0.078109, 0.996968), (0.074711, 0.997199), (0.073307, 0.997302)),
    (GEOMETRIES[4], CLAMP_LIMIT): ((0.139918, 0.989919), (0.075357, 0.997129), (0.247209, 0.968131), (0.215144, 0.976083)),
    (GEOMETRIES[5], None): ((0.034994, 0.999382), (0.078103, 0.996945), (0.074768, 0.997196), (0.073328, 0.997300)),
    (GEOMETRIES[5], CLAMP_LIMIT): ((0.112394, 0.993597), (0.075942, 0.997107), (0.247248, 0.967805), (0.214482, 0.976036)),
    (GEOMETRIES[6], None): ((0.052596, 0.998609), (0.080596, 0.996704), (0.074647, 0.997206), (0.073411, 0.997290)),
    (GEOMETRIES[6], CLAMP_LIMIT): ((0.165377, 0.985689), (0.076648, 0.997035), (0.229151, 0.968473), (0.206575, 0.975481)),
    (SINGLE_MACROBATCH_GEOMETRY, None): ((0.044206, 0.999017), (0.079777, 0.996818), (0.074677, 0.997203), (0.073435, 0.997292)),
    (SINGLE_MACROBATCH_GEOMETRY, CLAMP_LIMIT): ((0.140505, 0.989836), (0.075572, 0.997121), (0.239294, 0.968205), (0.212031, 0.975556)),
}


def _expert_slices(part, local_experts):
    """One grouped MXFP8 part per expert: the payload's L mode, or the scale's expert-major rows."""
    if part.ndim == 3:
        return [part[index] for index in range(local_experts)]
    rows = part.shape[0] // local_experts
    return [part[index * rows : (index + 1) * rows] for index in range(local_experts)]


def _reorder(part, local_experts, order):
    """A grouped part with its experts in ``order``."""
    return torch.cat(
        [_expert_slices(part, local_experts)[index].unsqueeze(0) if part.ndim == 3 else _expert_slices(part, local_experts)[index] for index in order]
    ).contiguous()


def _split_mxfp8_weight(quantized, local_experts, order, keepalive):
    """A ``SplitRoutedWeight`` over one grouped MXFP8 backward weight's experts, in ``order``.

    All four payloads, since the backward reads both: the rowwise pair for the replay and the
    columnwise pair for the dgrad, each with its own tcgen05 scale.  Every expert's parts are
    allocations of their own, made in the natural order and handed over in ``order``, so the
    tables are not monotonic in address; ``keepalive`` holds them, as MCore's parameters do.
    """
    order = tuple(order)
    parts = []
    for part in quantized:
        allocations = [expert.clone() for expert in _expert_slices(part, local_experts)]
        keepalive.append(allocations)
        parts.append([allocations[index] for index in order])
    experts, scales, experts_t, scales_t = parts
    return f.SplitRoutedWeight(
        data=experts[0],
        storage_table=make_routed_weight_storage_table_mxfp8(experts),
        scale=scales[0],
        scale_storage_table=make_routed_scale_storage_table(scales),
        scale_tensors=tuple(scales),
        transposed_data=experts_t[0],
        transposed_storage_table=make_routed_weight_storage_table_mxfp8(experts_t),
        transposed_scale=scales_t[0],
        transposed_scale_storage_table=make_routed_scale_storage_table(scales_t),
        transposed_scale_tensors=tuple(scales_t),
    )


def _split_backward_case(context, empty_first_local_expert: bool = False):
    """The geometry, inputs, forward context and grouped MXFP8 weights the split tests compare to.

    ``empty_first_local_expert`` routes away from every rank's local expert 0, so that one has no
    rows anywhere -- which is what tells a fused Wgrad that leaves main_grad alone apart from one
    that zero-fills it.
    """
    rank, world_size, device = context
    if world_size != 8:
        pytest.skip("the MXFP8 backward geometries here are EP8")
    if torch.cuda.get_device_capability(device) not in SUPPORTED_CAPABILITIES:
        pytest.skip("the CuTe DSL MXFP8 backward is SM100/SM103 only")
    experts, hidden, intermediate, topk, tokens, comm, minibatch, macrobatch = SINGLE_MACROBATCH_GEOMETRY
    local_experts = experts // world_size
    assert local_experts >= 2, "a table that reads one expert everywhere would pass vacuously"
    config = f.CuteMoKConfig(
        fwd_num_comm_sms=comm, bwd_num_comm_sms=comm, minibatch_size=minibatch, macrobatch_size=macrobatch, schedule_capacity_multiplier=1.5
    )
    workspace = f.create_workspace(config, dist.group.WORLD, device=device, num_local_tokens=tokens, hidden_size=hidden, topk=topk)
    x, routes, router_weights, shared_gate, shared_up, shared_down, gate, up, down, d_output = generate_inputs(
        rank, device, experts, local_experts, topk, tokens, hidden, intermediate
    )
    if empty_first_local_expert:
        # The routes are global expert ids and local expert e of rank r is r * local_experts + e,
        # so moving the multiples of local_experts one along empties local expert 0 on every rank
        # at once -- which keeps the assertion uniform across ranks.
        routes = torch.where(routes % local_experts == 0, routes + 1, routes)
    schedule = f.build_schedule(workspace, config, routes, num_local_experts=local_experts)
    quantized = [tuple(mxfp8_quantize(weight, True, True)) for weight in (gate, up, down)]
    shared = (shared_gate, shared_up, shared_down)
    _, ctx = f.forward(config, workspace, schedule, x, router_weights, *shared, quantized[0][:2], quantized[1][:2], quantized[2][:2], None)
    _sync_ranks(device)
    return config, workspace, schedule, ctx, d_output, x, router_weights, shared, quantized, local_experts


def _split_backward(case, routed, main_grads=None, main_grad_storage_tables=None):
    config, workspace, schedule, ctx, d_output, x, router_weights, shared = case[:8]
    gradients = ungated(
        f.backward(
            config,
            workspace,
            schedule,
            _clone_context(ctx),
            d_output,
            x,
            router_weights,
            *shared,
            *routed,
            None,
            main_grads=main_grads,
            main_grad_storage_tables=main_grad_storage_tables,
        )
    )
    _sync_ranks(device=x.device)
    return tuple(gradient.clone() for gradient in gradients)


def _mxfp8_main_grad_buffers(case, dtype, fill=0.0):
    """Six buffers shaped like the gradients, in MOK's main_grads ABI order."""
    shared_gate, shared_up, shared_down = case[7]
    gate, _, down = (weight[0] for weight in case[8])
    shapes = (shared_gate.shape, gate.shape, shared_up.shape, gate.shape, shared_down.shape, down.shape)
    return tuple(torch.full(tuple(shape), fill, dtype=dtype, device=shared_gate.device) for shape in shapes)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_cutedsl_mxfp8_fused_main_grad(context, dtype):
    """The MXFP8 backward writes its weight gradients into main_grad, as the BF16 one does.

    Into a zeroed buffer on a single-macrobatch geometry the fused result must reproduce the
    unfused gradient exactly -- for float32 after its bfloat16 rounding, which is the one step
    the unfused path takes before the host adds.
    """
    case = _split_backward_case(context)
    quantized = case[8]
    grouped = (quantized[0], quantized[1], quantized[2][2:])
    unfused = _split_backward(case, grouped)
    buffers = _mxfp8_main_grad_buffers(case, dtype)
    fused = _split_backward(case, grouped, main_grads=buffers)
    del fused
    # GRADIENTS order is (d_x, d_router_weights, routed gate/up/down, shared gate/up/down); the
    # buffers are MAIN_GRAD_NAMES order (shared/routed interleaved per matrix).
    expected = (unfused[5], unfused[2], unfused[6], unfused[3], unfused[7], unfused[4])
    for name, want, buffer in zip(f.MAIN_GRAD_NAMES, expected, buffers, strict=True):
        assert torch.equal(want, buffer.to(torch.bfloat16)), f"{name}: the fused MXFP8 weight gradient differs from the unfused one"


def _grouped_weights(case):
    """The case's MXFP8 weights as the grouped backward takes them."""
    quantized = case[8]
    return (quantized[0], quantized[1], quantized[2][2:])


def test_cutedsl_mxfp8_fused_main_grad_accumulates_rather_than_overwrites(context):
    """Two identical microbatches into one buffer give exactly twice one.

    Exact on purpose: adding a float to itself is exact in binary floating point, so an epilogue
    that overwrote would leave one times the gradient, not two.  This is the check that the
    shared expert's Wgrad accumulates too -- against a zeroed buffer a store and an add agree.
    """
    case = _split_backward_case(context)
    grouped = _grouped_weights(case)
    once = _mxfp8_main_grad_buffers(case, torch.float32)
    _split_backward(case, grouped, main_grads=once)
    after_one = tuple(buffer.clone() for buffer in once)
    _split_backward(case, grouped, main_grads=once)
    for name, first, twice in zip(f.MAIN_GRAD_NAMES, after_one, once, strict=True):
        assert torch.equal(first * 2.0, twice), (
            f"{name}: two microbatches did not give twice one, so the epilogue overwrote " "main_grad rather than accumulating into it"
        )


def test_cutedsl_mxfp8_fused_main_grad_leaves_an_empty_expert_alone(context):
    """An expert with no rows must not have its main_grad touched.

    Unfused, MOK owns the gradient buffer and zero-fills such an expert.  Accumulating, the
    buffer is MCore's and already holds earlier microbatches, so zeroing would erase them -- the
    CUDA port skips ``zero_empty_routed_wgrads`` for exactly this reason.  Starting from a buffer
    that is not zero is what tells the two apart.
    """
    case = _split_backward_case(context, empty_first_local_expert=True)
    schedule = case[2]
    assert int(schedule.tokens_per_expert[0]) == 0, "local expert 0 was meant to have no rows"
    seeded = _mxfp8_main_grad_buffers(case, torch.float32, fill=1.5)
    _split_backward(case, _grouped_weights(case), main_grads=seeded)
    for name, buffer in zip(f.MAIN_GRAD_NAMES, seeded, strict=True):
        if not name.startswith("routed_"):
            continue
        empty = buffer[0]
        assert torch.equal(empty, torch.full_like(empty, 1.5)), (
            f"{name}: the empty expert's gradient was written, which would erase the earlier " "microbatches main_grad already holds"
        )


def test_cutedsl_mxfp8_fused_main_grad_replays(context):
    """The same inputs into the same starting buffer give the same bytes."""
    case = _split_backward_case(context)
    grouped = _grouped_weights(case)
    first = _mxfp8_main_grad_buffers(case, torch.float32)
    second = _mxfp8_main_grad_buffers(case, torch.float32)
    _split_backward(case, grouped, main_grads=first)
    _split_backward(case, grouped, main_grads=second)
    for name, left, right in zip(f.MAIN_GRAD_NAMES, first, second, strict=True):
        assert torch.equal(left, right), f"a replayed fused MXFP8 backward: {name} differs"


def _assert_gradients_identical(expected, actual, what):
    for name, want, got in zip(GRADIENTS, expected, actual, strict=True):
        assert torch.equal(want, got), f"{what}: {name} differs; the two arms read the same memory, so this is an " "addressing error, not a numerical one"


def _split_main_grad_backward_both_layouts(context, order):
    """The same weights and seeded main_grads in both layouts, through the backward.

    Split routed weights come with split main_grads -- the C++ port's pairing, and MCore's -- so
    the comparison is of what lands in main_grad: grouped ``[E, 2I, H]`` / ``[E, H, I]`` buffers
    against one ``[2I, H]`` / ``[H, I]`` allocation per expert, both seeded with the same non-zero
    values.  The grouped arm's experts are put in ``order``, so expert ``e`` holds the same
    weights in both arms.  Returns, per arm, (d_x, d_router_weights, shared main_grads, routed FC1
    main_grads by expert, routed Down main_grads by expert).
    """
    case = _split_backward_case(context)
    quantized, local_experts = case[8], case[9]
    order = tuple(order)
    device = case[5].device
    generator = torch.Generator(device=device)
    generator.manual_seed(4321 + dist.get_rank())
    shared_gate, shared_up, shared_down = case[7]
    gate = quantized[0][0]
    down = quantized[2][0]
    experts, intermediate, hidden = gate.shape

    def seeded(*shape):
        return torch.rand(*shape, generator=generator, device=device) - 0.5

    fc1_seed = seeded(experts, 2 * intermediate, hidden)
    down_seed = seeded(*down.shape)
    shared_seed = (seeded(2 * intermediate, hidden), seeded(*shared_down.shape))

    def shared_buffers():
        fc1 = shared_seed[0].clone()
        return fc1, fc1[:intermediate], fc1[intermediate:], shared_seed[1].clone()

    grouped_weights = [tuple(_reorder(part, local_experts, order) for part in weight) for weight in quantized]
    grouped_fc1, grouped_down = fc1_seed[list(order)].clone(), down_seed[list(order)].clone()
    shared_fc1, s_gate, s_up, s_down = shared_buffers()
    gradients = _split_backward(
        case, (grouped_weights[0], grouped_weights[1], grouped_weights[2][2:]), main_grads=(s_gate, grouped_fc1, s_up, grouped_fc1, s_down, grouped_down)
    )
    grouped = (gradients[0], gradients[1], (shared_fc1, s_down), list(grouped_fc1), list(grouped_down))

    keepalive = []
    split_weights = tuple(_split_mxfp8_weight(weight, local_experts, order, keepalive) for weight in quantized)

    def own(tensor):
        allocations = [tensor[index].clone() for index in range(local_experts)]
        return [allocations[index] for index in order]

    fc1_grads, down_grads = own(fc1_seed), own(down_seed)
    fc1_table = make_routed_d_weight_storage_table(fc1_grads)
    shared_fc1, s_gate, s_up, s_down = shared_buffers()
    gradients = _split_backward(
        case,
        split_weights,
        main_grads=(s_gate, fc1_grads[0], s_up, fc1_grads[0], s_down, down_grads[0]),
        main_grad_storage_tables=(fc1_table, fc1_table, make_routed_d_weight_storage_table(down_grads)),
    )
    split = (gradients[0], gradients[1], (shared_fc1, s_down), fc1_grads, down_grads)
    return grouped, split


def _assert_same_backward(grouped, split, what):
    labels = ("d_x", "d_router_weights", "shared main_grads", "routed FC1 main_grad", "routed Down main_grad")
    for label, left, right in zip(labels, grouped, split, strict=True):
        lefts = left if isinstance(left, (list, tuple)) else (left,)
        rights = right if isinstance(right, (list, tuple)) else (right,)
        for index, (a, b) in enumerate(zip(lefts, rights, strict=True)):
            assert torch.equal(a, b), (
                f"{what}: {label}[{index}] differs in {(a != b).sum().item()} of {a.numel()} "
                "elements; the two arms read the same bytes, so this is an addressing error"
            )


def test_cutedsl_mxfp8_split_backward_matches_grouped(context):
    """Ten operands and three main_grads behind descriptors: the five weights the dgrad reads,
    their five scales, and the Down, Gate and Up reduce-add destinations."""
    grouped, split = _split_main_grad_backward_both_layouts(context, order=range(2))
    _assert_same_backward(grouped, split, "split MXFP8 routed weights and main_grads")


def test_cutedsl_mxfp8_split_backward_follows_the_table(context):
    """Hand the experts over in reverse; every operand and main_grad must follow its table."""
    grouped, split = _split_main_grad_backward_both_layouts(context, order=(1, 0))
    _assert_same_backward(grouped, split, "a reversed MXFP8 expert table in the backward")


def _clone_context(ctx):
    """A copy of every tensor the context holds.  Absent fields (an ungated or routed-only
    context's None) and the SiTU betas (plain floats) are kept as they are."""

    def clone(value):
        if isinstance(value, torch.Tensor):
            return value.clone()
        if isinstance(value, tuple):
            return tuple(clone(item) for item in value)
        return value

    return dataclasses.replace(ctx, **{field.name: clone(getattr(ctx, field.name)) for field in dataclasses.fields(ctx)})


def _context_dict(ctx):
    """The entry's context arguments: e4m3 / scale Gate / Up pairs, or the BF16 tensors."""
    x_t, x_sc_t = ctx.x_routed
    hidden_t, hidden_sc_t = ctx.hidden_routed
    context = dict(
        x_fp8_t_routed=x_t,
        x_sc_t_routed=x_sc_t,
        gate_shared=ctx.gate_shared,
        up_shared=ctx.up_shared,
        hidden_shared=ctx.hidden_shared,
        hidden_fp8_t_routed=hidden_t,
        hidden_sc_t_routed=hidden_sc_t,
    )
    if isinstance(ctx.gate_routed, tuple):
        (gate, gate_sc), (up, up_sc) = ctx.gate_routed, ctx.up_routed
        context.update(gate_fp8_routed=gate, gate_sc_routed=gate_sc, up_fp8_routed=up, up_sc_routed=up_sc)
    else:
        context.update(gate_routed_bf16=ctx.gate_routed, up_routed_bf16=ctx.up_routed)
    return context


def _written(name, tensor, rows):
    """The part of a raw output the last macrobatch wrote (``rows`` macrobatch-local rows): the
    routed ring buffers; the weight gradients and the shared tensors whole."""
    if name.startswith("d_w_") or name.endswith("_shared"):
        return tensor
    if name.endswith("_sc_routed"):
        return tensor[: rows // 128]
    return tensor[:rows]


def _prepare(ops, workspace, d_output, x, router_weights, device):
    workspace.d_y_buffer.copy_(d_output)
    workspace.x_buffer.copy_(x)
    workspace.router_weight_buffer.copy_(router_weights)
    workspace.d_x_routed_buffer.zero_()
    workspace.d_router_weight_buffer.zero_()
    _sync_ranks(device)  # every peer's receive buffers are zero before anyone sends
    ops.barrier_all(workspace.barrier_buffer, workspace.barrier_buffer_ptrs, workspace.barrier_buffer_multicast_ptr, workspace.barrier_target)


def _finish(ops, workspace, device):
    ops.barrier_all(workspace.barrier_buffer, workspace.barrier_buffer_ptrs, workspace.barrier_buffer_multicast_ptr, workspace.barrier_target)
    _sync_ranks(device)
    return workspace.d_x_routed_buffer.clone(), workspace.d_router_weight_buffer.clone()


def _run_raw(
    config, workspace, schedule, ctx, d_output, x, router_weights, shared, gate_q, up_q, down_q, swiglu_limit, device, recompute_forward_context=False
):
    """The backward through its low-level entry on a fresh copy of the context.  Returns the raw outputs, the
    peer-written buffers and a dict of the call's state: the transposed e4m3 d_gate / d_up with their scales
    (``TRANSPOSED``), its communication counters (``STREAM_COUNTERS``) and the address of its ``tensormaps``
    buffer (``tensormaps_ptr``).

    ``recompute_forward_context``: only the shared expert's context is passed, and the routed context the host allocates
    is filled with 0x7F bytes first (an e4m3 NaN, a BF16 3.4e38), so a read that precedes the replay's write reaches
    the outputs instead of finding whatever an earlier call left in the allocator's block."""
    shared_gate, shared_up, shared_down = shared
    ops = f._device_ops(config)
    context = _context_dict(_clone_context(ctx))
    variant = {}
    if recompute_forward_context:
        context = {name: context[name] for name in mxfp8_bwd_host.SHARED_CONTEXT_NAMES}
        variant = dict(recompute_forward_context=True, gate_up_context=config.gate_up_context)
    _prepare(ops, workspace, d_output, x, router_weights, device)
    states, scratch = [], []
    prepare = mxfp8_bwd_host.prepare_persistent_mxfp8_bwd_state
    prepare_routed_context = mxfp8_bwd_host.prepare_persistent_mxfp8_bwd_routed_context

    def capture(**kwargs):
        states.append(prepare(**kwargs))
        return states[-1]

    def poisoned(**kwargs):
        scratch.append(prepare_routed_context(**kwargs))
        for buffer in scratch[-1].values():
            buffer.view(torch.uint8).fill_(0x7F)
        return scratch[-1]

    with (
        mock.patch.object(mxfp8_bwd_host, "prepare_persistent_mxfp8_bwd_state", capture),
        mock.patch.object(mxfp8_bwd_host, "prepare_persistent_mxfp8_bwd_routed_context", poisoned),
    ):
        outputs = mxfp8_bwd_host.backward_mxfp8(
            workspace,
            schedule,
            context,
            shared_gate,
            tuple(gate_q),
            shared_up,
            tuple(up_q),
            shared_down,
            tuple(down_q[2:]),
            macrobatch_size=config.macrobatch_size,
            minibatch_size=f._bwd_minibatch_size(config),
            num_comm_sms=config.bwd_num_comm_sms,
            swiglu_limit=swiglu_limit,
            **variant,
        )
    assert len(scratch) == int(recompute_forward_context), "the routed context is the host's scratch with recompute_forward_context only"
    outputs = [t.clone() for t in outputs]
    state = {name: getattr(states[0], name).clone() for name in TRANSPOSED}
    for counter in STREAM_COUNTERS:
        state[counter] = states[0].counters[counter].clone()
    state["tensormaps_ptr"] = states[0].tensormaps.data_ptr()
    buffers = _finish(ops, workspace, device)
    assert len(outputs) == len(RAW_OUTPUTS)
    return outputs, buffers, state


STREAM_COUNTERS = ("d_y_routed_ready", "replayed_x_routed_ready", "d_x_routed_ready", "routed_buffers_done")


def _assert_stream_counters(
    counters, num_tokens, macrobatch, minibatch, hidden, intermediate, local_experts, schedule_capacity, recompute_forward_context=False
):
    """The communication counters' final values after one backward, each the arrivals the kernel's waits expect
    (a shortfall deadlocks the test, an excess would release a consumer early and only shows here):
    ``d_y_routed_ready[g]`` = the stream's dY pull blocks of minibatch g (ceil(rows / 32) x ceil(H / 256)), then
    one head word per macrobatch of the schedule capacity, macrobatch m's the (m + 1)-th from the end = its stream's
    block count ((pulls + pushes) x column groups, a replayed macrobatch pulling x and dY), the words of
    macrobatches that did not run still zero; ``replayed_x_routed_ready[g]`` = the x pull blocks of a replayed
    minibatch, 0 for macrobatch 0's; ``d_x_routed_ready[g]`` = the routed Gate / Up dgrad tasks of minibatch g
    (ceil(rows / 256) x H / 256), both CTAs of the cluster; ``routed_buffers_done[m]`` = CLUSTER_SIZE x (the
    macrobatch's routed tasks + its Wgrad tasks) + one per push block before the last macrobatch
    (``_run_mxfp8_comm_prefix``'s retire count; the task counts are ``_task_counts``'), and nothing at all with a
    single macrobatch (no task retires its buffers when no later macrobatch waits for them).
    ``recompute_forward_context``: macrobatch 0 is a replayed macrobatch (its x pull blocks, and x and dY pulls in its
    head word); the retire counts are the same."""
    counts = {name: tensor.tolist() for name, tensor in counters.items()}
    minibatches = -(-schedule_capacity // minibatch)
    macrobatches = -(-schedule_capacity // macrobatch)
    macro_count = -(-num_tokens // macrobatch)
    d_y = counts["d_y_routed_ready"]
    assert len(d_y) == minibatches + macrobatches, (len(d_y), minibatches, macrobatches)
    assert len(counts["replayed_x_routed_ready"]) == len(counts["d_x_routed_ready"]) == minibatches
    assert len(counts["routed_buffers_done"]) == macrobatches
    column_blocks = -(-hidden // 256)
    expected_d_y, expected_x, expected_d_x = [], [], []
    for g in range(minibatches):
        rows = max(0, min(minibatch, num_tokens - g * minibatch))
        blocks = -(-rows // 32) * column_blocks
        expected_d_y.append(blocks)
        expected_x.append(blocks if recompute_forward_context or g * minibatch >= macrobatch else 0)
        expected_d_x.append(-(-rows // 256) * (hidden // 256) * 2)
    assert d_y[:minibatches] == expected_d_y, (d_y, expected_d_y)
    assert counts["replayed_x_routed_ready"] == expected_x, (counts["replayed_x_routed_ready"], expected_x)
    assert counts["d_x_routed_ready"] == expected_d_x, (counts["d_x_routed_ready"], expected_d_x)
    groups = -(-column_blocks // 4)
    heads = [0] * macrobatches
    for m in range(macro_count):
        tiles = min(macrobatch, num_tokens - m * macrobatch) // 32
        heads[-1 - m] = (tiles * (2 if recompute_forward_context or m > 0 else 1) + tiles) * groups
    assert d_y[minibatches:] == heads, (d_y[minibatches:], heads, num_tokens)
    # _task_counts: the routed tasks of one minibatch and the Wgrad tasks of one macrobatch (3 x E x weight tiles).
    intermediate_blocks, hidden_blocks, swiglu_columns = intermediate // 256, hidden // 256, intermediate // 128
    routed = (minibatch // 256) * intermediate_blocks + -(-(minibatch // 128 * swiglu_columns) // 4) + (minibatch // 256) * hidden_blocks
    wgrad = 3 * local_experts * intermediate_blocks * hidden_blocks
    expected_done = [0] * macrobatches
    for m in range(macro_count if macro_count > 1 else 0):
        rows = min(macrobatch, num_tokens - m * macrobatch)
        expected_done[m] = 2 * (-(-rows // minibatch) * routed + wgrad)
        if m + 1 < macro_count:
            expected_done[m] += rows // 32 * column_blocks
    assert counts["routed_buffers_done"] == expected_done, (counts["routed_buffers_done"], expected_done)


def _backward(rank, world_size, device, geometry, swiglu_limit, gate_up_context="bf16"):
    """The raw and the public backward on one geometry, on a context from the MXFP8 forward.  Returns the raw run
    (``_run_raw``), the public gradients, the BF16 reference's gradients, the rows of the last macrobatch and the
    macrobatch count."""
    experts, hidden, intermediate, topk, tokens, comm, minibatch, macrobatch = geometry
    config = f.CuteMoKConfig(
        fwd_num_comm_sms=comm,
        bwd_num_comm_sms=comm,
        minibatch_size=minibatch,
        macrobatch_size=macrobatch,
        schedule_capacity_multiplier=1.5,
        gate_up_context=gate_up_context,
    )
    workspace = f.create_workspace(config, dist.group.WORLD, device=device, num_local_tokens=tokens, hidden_size=hidden, topk=topk)
    x, routes, router_weights, shared_gate, shared_up, shared_down, gate, up, down, d_output = generate_inputs(
        rank, device, experts, experts // world_size, topk, tokens, hidden, intermediate
    )
    schedule = f.build_schedule(workspace, config, routes, num_local_experts=experts // world_size)
    gate_q, up_q, down_q = (tuple(mxfp8_quantize(weight, True, True)) for weight in (gate, up, down))
    shared = (shared_gate, shared_up, shared_down)
    _, ctx = f.forward(config, workspace, schedule, x, router_weights, *shared, gate_q[:2], up_q[:2], down_q[:2], swiglu_limit)
    _sync_ranks(device)
    num_tokens = int(schedule.num_tokens[0])
    macrobatches = max(1, -(-num_tokens // macrobatch))
    rows = num_tokens - (macrobatches - 1) * macrobatch
    raw = _run_raw(config, workspace, schedule, ctx, d_output, x, router_weights, shared, gate_q, up_q, down_q, swiglu_limit, device)
    _assert_stream_counters(
        {name: raw[2][name] for name in STREAM_COUNTERS},
        num_tokens,
        macrobatch,
        minibatch,
        hidden,
        intermediate,
        experts // world_size,
        workspace.schedule_capacity,
    )
    reference = run_reference_bf16(x, routes, router_weights, *shared, gate, up, down, d_output, swiglu_limit)[1:]
    gradients = ungated(
        f.backward(config, workspace, schedule, _clone_context(ctx), d_output, x, router_weights, *shared, gate_q, up_q, down_q[2:], swiglu_limit)
    )
    _sync_ranks(device)
    return raw, tuple(g.clone() for g in gradients), reference, rows, macrobatches


def _dequantize(fp8, scales):
    """MoK's MXFP8 tile layout to FP32: ``fp8`` (rows, cols) e4m3 in 32-column blocks, ``scales`` (rows / 128,
    cols / 128, 32, 16) e8m0, row ``r``'s block ``b`` of a 128 x 128 tile at byte (r % 32) * 16 + (r / 32) * 4 + b.
    Returns the values and each element's block scale."""
    rows, cols = fp8.shape
    exponents = scales.view(rows // 128, cols // 128, 32, 4, 4).permute(0, 3, 2, 1, 4).reshape(rows, cols // 32)
    scale = torch.exp2(exponents.float() - 127.0).repeat_interleave(32, dim=1)
    return fp8.float() * scale, scale


def _assert_layouts_agree(outputs, transposed, rows):
    """The normal (32-column blocks) and transposed (32-row blocks) e4m3 d_gate / d_up of one BF16 tensor
    agree within one e4m3 step of the larger block scale (2^-3 relative, 2^-9 x scale absolute), and are finite."""
    named = dict(zip(RAW_OUTPUTS, outputs, strict=True))
    for gradient in ("d_gate", "d_up"):
        normal, normal_scale = _dequantize(named[f"{gradient}_fp8_routed"], named[f"{gradient}_sc_routed"])
        across, across_scale = _dequantize(transposed[f"{gradient}_fp8_t_routed"], transposed[f"{gradient}_sc_t_routed"])
        normal, normal_scale = normal[:rows], normal_scale[:rows]
        across, across_scale = across.t()[:rows], across_scale.t()[:rows]
        assert torch.isfinite(normal).all() and torch.isfinite(across).all(), gradient
        bound = 0.125 * torch.maximum(normal.abs(), across.abs()) + torch.maximum(normal_scale, across_scale) / 512
        worst = ((normal - across).abs() - bound).max().item()
        assert worst <= 0, (gradient, worst)
        assert normal.ne(0).any(), f"{gradient} is zero"


def _relative_l1(reference, actual):
    """Relative L1 distance to ``reference`` over all ranks."""
    sums = torch.stack([(reference.float() - actual.float()).abs().sum(), reference.float().abs().sum()]).double()
    dist.all_reduce(sums)
    return float(sums[0] / sums[1].clamp_min(1e-30))


def _cosine(reference, actual):
    """Cosine similarity with ``reference`` over all ranks."""
    reference, actual = reference.float().flatten(), actual.float().flatten()
    sums = torch.stack([reference @ actual, reference @ reference, actual @ actual]).double()
    dist.all_reduce(sums)
    return float(sums[0] / (sums[1] * sums[2]).sqrt().clamp_min(1e-30))


def _assert_backward(raw, gradients, reference, rows, geometry, swiglu_limit, gate_up_context="bf16", *, upstream=True):
    """Every raw output and peer-written buffer is written and finite, the two e4m3 layouts of d_gate / d_up agree,
    and the public gradients are as close to the BF16 reference's as upstream's on this geometry and clamp limit
    (``UPSTREAM``).  ``upstream=False``: no upstream measurement of this geometry; every gradient at MIN_COSINE."""
    outputs, buffers, state = raw
    for name, actual in zip(RAW_OUTPUTS, outputs, strict=True):
        actual = _written(name, actual, rows)
        assert actual.view(torch.uint8).ne(0).any(), f"{name} was not written"
        if actual.dtype in (torch.bfloat16, torch.float32):
            assert torch.isfinite(actual).all(), name
    if gate_up_context == "bf16":
        _assert_layouts_agree(outputs, state, rows)
    for name, actual in zip(("d_x_routed_buffer", "d_router_weight_buffer"), buffers, strict=True):
        assert actual.view(torch.int32).ne(0).any(), f"{name} was not written"
        assert torch.isfinite(actual).all(), name
    assert len(gradients) == len(reference) == len(GRADIENTS)
    measured = dict(zip(GATE_UP_GRADIENTS, UPSTREAM[geometry, swiglu_limit], strict=True)) if upstream else {}
    for name, exact, actual in zip(GRADIENTS, reference, gradients, strict=True):
        assert exact.shape == actual.shape, name
        min_cosine = MIN_COSINE
        if name in measured:
            upstream_error, upstream_cosine = measured[name]
            error = _relative_l1(exact, actual)
            assert error <= NOT_WORSE * upstream_error + 1e-9, (name, error, upstream_error)
            min_cosine = min(MIN_COSINE, upstream_cosine - COSINE_ROUNDING)
        cosine = _cosine(exact, actual)
        assert cosine >= min_cosine, (name, cosine, min_cosine)


def test_the_cosine_bar_is_lowered_only_where_upstream_is_below_it():
    """``UPSTREAM`` covers every geometry and clamp mode the tests run, and lowers the cosine bar only for the three
    gradients behind the SwiGLU backward at the clamp limit the inputs reach."""
    assert set(UPSTREAM) == {(geometry, limit) for geometry in (*GEOMETRIES, SINGLE_MACROBATCH_GEOMETRY) for limit in (None, CLAMP_LIMIT)}
    for (_, limit), values in UPSTREAM.items():
        for name, (error, cosine) in zip(GATE_UP_GRADIENTS, values, strict=True):
            clamped = limit == CLAMP_LIMIT and name in ("d_x", "d_w_routed_gate", "d_w_routed_up")
            assert (cosine < MIN_COSINE) == clamped, (limit, name, cosine)
            assert 0.96 < cosine <= 1.0 and 0.0 < error < 0.25, (limit, name, error, cosine)


@pytest.mark.parametrize("gate_up_context", ("bf16", "mxfp8"))
def test_cutedsl_mxfp8_backward(context, gate_up_context):
    """Every geometry, both clamp modes, including geometries that replay macrobatches."""
    rank, world_size, device = context
    if world_size != 8:
        pytest.skip("the MXFP8 backward geometries here are EP8")
    if torch.cuda.get_device_capability(device) not in SUPPORTED_CAPABILITIES:
        pytest.skip("the CuTe DSL MXFP8 backward is SM100/SM103 only")
    d_gate_shared = RAW_OUTPUTS.index("d_gate_shared")
    replayed = 0
    for geometry in GEOMETRIES:
        outputs = {}
        for swiglu_limit in (None, CLAMP_LIMIT):
            raw, gradients, reference, rows, macrobatches = _backward(rank, world_size, device, geometry, swiglu_limit, gate_up_context)
            _assert_backward(raw, gradients, reference, rows, geometry, swiglu_limit, gate_up_context)
            outputs[swiglu_limit] = raw[0]
            replayed += macrobatches > 1
        # The clamp is live: the clamped shared-gate gradient differs from the unclamped one.
        assert not torch.equal(outputs[None][d_gate_shared], outputs[CLAMP_LIMIT][d_gate_shared])
    assert replayed, "no geometry exercised the replayed macrobatches"
    # Sizes are runtime arguments: one executor per clamp mode and Gate / Up context, whatever the geometry.
    for is_clamped in (False, True):
        this_variant = (
            ("gate_up_context", gate_up_context),
            ("routed_weights", "grouped"),
            ("native_columnwise", False),
            ("is_situ", False),
            ("has_shared", True),
            ("has_tail", False),
            ("recompute_forward_context", False),
            # the fused main_grad and recompute_forward_context tests compile their own
            ("main_grad", "none", "grouped"),
        )
        assert sum(("is_clamped", is_clamped) in key and all(axis in key for axis in this_variant) for key in mxfp8_bwd_host._EXECUTOR_CACHE) == 1


def _latent_backward_matches_plain(rank, world_size, device, geometry, swiglu_limit, gate_up_context):
    """One latent geometry (workspace.routed_hidden_size != hidden_size): the public MXFP8 backward with
    ``routed_grad_output`` / ``routed_input`` against two ordinary backwards, byte for byte.

    The routed chain (the dX rows pushed into the peers' buffers, the router-weight gradient, the routed weight
    gradients) is compared with the backward run on ``routed_input`` / ``routed_grad_output`` with hidden ==
    routed hidden.  The shared chain (``d_x_shared``, the shared weight gradients) must equal the low-level
    backward run on ``x`` / ``grad_output`` with the shared hidden in both.  One schedule serves the three
    workspaces.
    """
    experts, hidden, routed_hidden, intermediate, topk, tokens, comm, minibatch, macrobatch = geometry
    local_experts = experts // world_size
    config = f.CuteMoKConfig(
        fwd_num_comm_sms=comm,
        bwd_num_comm_sms=comm,
        minibatch_size=minibatch,
        macrobatch_size=macrobatch,
        schedule_capacity_multiplier=1.5,
        gate_up_context=gate_up_context,
    )
    latent = f.create_workspace(
        config, dist.group.WORLD, device=device, num_local_tokens=tokens, hidden_size=hidden, topk=topk, routed_hidden_size=routed_hidden
    )
    in_routed = f.create_workspace(config, dist.group.WORLD, device=device, num_local_tokens=tokens, hidden_size=routed_hidden, topk=topk)
    in_hidden = f.create_workspace(config, dist.group.WORLD, device=device, num_local_tokens=tokens, hidden_size=hidden, topk=topk)
    x, routes, router_weights, shared_gate, shared_up, shared_down, gate_h, up_h, down_h, grad_output = generate_inputs(
        rank, device, experts, local_experts, topk, tokens, hidden, intermediate
    )
    _, _, _, shared_gate_r, shared_up_r, shared_down_r, gate, up, down, _ = generate_inputs(
        rank, device, experts, local_experts, topk, tokens, routed_hidden, intermediate
    )
    generator = torch.Generator(device=device).manual_seed(1234)
    w_down = torch.randn(routed_hidden, hidden, device=device, generator=generator) / hidden**0.5
    w_up = torch.randn(hidden, routed_hidden, device=device, generator=generator) / routed_hidden**0.5
    w_down, w_up = w_down.to(torch.bfloat16), w_up.to(torch.bfloat16)
    routed_input = torch.matmul(x, w_down.t())
    routed_grad_output = torch.matmul(grad_output, w_up)
    schedule = f.build_schedule(latent, config, routes, num_local_experts=local_experts)
    gate_q, up_q, down_q = (tuple(mxfp8_quantize(weight, True, True)) for weight in (gate, up, down))
    gate_qh, up_qh, down_qh = (tuple(mxfp8_quantize(weight, True, True)) for weight in (gate_h, up_h, down_h))
    shared, shared_r = (shared_gate, shared_up, shared_down), (shared_gate_r, shared_up_r, shared_down_r)
    _, ctx_r = f.forward(config, in_routed, schedule, routed_input, router_weights, *shared_r, gate_q[:2], up_q[:2], down_q[:2], swiglu_limit)
    ctx_r = _clone_context(ctx_r)
    _sync_ranks(device)
    _, ctx_h = f.forward(config, in_hidden, schedule, x, router_weights, *shared, gate_qh[:2], up_qh[:2], down_qh[:2], swiglu_limit)
    ctx_h = _clone_context(ctx_h)
    _sync_ranks(device)
    _, ctx = f.forward(config, latent, schedule, x, router_weights, *shared, gate_q[:2], up_q[:2], down_q[:2], swiglu_limit, routed_input=routed_input)
    _sync_ranks(device)
    # The shared chain's reference: the low-level backward with the shared hidden in both.
    raw_h, _, _ = _run_raw(config, in_hidden, schedule, ctx_h, grad_output, x, router_weights, shared, gate_qh, up_qh, down_qh, swiglu_limit, device)
    raw_h = dict(zip(RAW_OUTPUTS, raw_h, strict=True))
    # The routed chain's reference: the public backward with the routed hidden in both.
    in_routed.d_x_routed_buffer.zero_()
    in_routed.d_router_weight_buffer.zero_()
    _sync_ranks(device)
    plain_r = ungated(
        f.backward(config, in_routed, schedule, ctx_r, routed_grad_output, routed_input, router_weights, *shared_r, gate_q, up_q, down_q[2:], swiglu_limit)
    )
    plain_r = tuple(g.clone() for g in plain_r)
    _sync_ranks(device)
    d_x_routed_buffer_r = in_routed.d_x_routed_buffer.clone()
    latent.d_x_routed_buffer.zero_()
    latent.d_router_weight_buffer.zero_()
    _sync_ranks(device)
    grads = ungated(
        f.backward(
            config,
            latent,
            schedule,
            ctx,
            grad_output,
            x,
            router_weights,
            *shared,
            gate_q,
            up_q,
            down_q[2:],
            swiglu_limit,
            routed_grad_output=routed_grad_output,
            routed_input=routed_input,
        )
    )
    _sync_ranks(device)
    (d_x_shared, d_routed_input), d_router_weights, d_w_routed_gate, d_w_routed_up, d_w_routed_down, d_w_shared_gate, d_w_shared_up, d_w_shared_down = grads
    assert tuple(d_x_shared.shape) == (tokens, hidden) and tuple(d_routed_input.shape) == (tokens, routed_hidden)
    ops = f._device_ops(config)
    plain_d_routed_input = ops.bwd_epilogue(torch.zeros(tokens, routed_hidden, dtype=torch.bfloat16, device=device), d_x_routed_buffer_r, schedule.top_experts)
    routed = (
        ("d_routed_input", plain_d_routed_input, d_routed_input),
        ("d_router_weights", plain_r[1], d_router_weights),
        ("d_w_routed_gate", plain_r[2], d_w_routed_gate),
        ("d_w_routed_up", plain_r[3], d_w_routed_up),
        ("d_w_routed_down", plain_r[4], d_w_routed_down),
    )
    assert d_x_routed_buffer_r.view(torch.int32).ne(0).any(), "d_x_routed_buffer was not written"
    assert torch.equal(_bytes(d_x_routed_buffer_r), _bytes(latent.d_x_routed_buffer)), "d_x_routed_buffer"
    for name, expected, actual in routed:
        assert expected.shape == actual.shape and expected.dtype == actual.dtype, name
        assert torch.equal(_bytes(expected), _bytes(actual)), name
    for name, actual in (
        ("d_x_shared", d_x_shared),
        ("d_w_shared_gate", d_w_shared_gate),
        ("d_w_shared_up", d_w_shared_up),
        ("d_w_shared_down", d_w_shared_down),
    ):
        expected = raw_h[name]
        assert expected.shape == actual.shape and expected.dtype == actual.dtype, name
        assert torch.equal(_bytes(expected), _bytes(actual)), name
    return tuple(
        g.clone()
        for g in (
            d_x_shared,
            d_routed_input,
            d_router_weights,
            d_w_routed_gate,
            d_w_routed_up,
            d_w_routed_down,
            d_w_shared_gate,
            d_w_shared_up,
            d_w_shared_down,
        )
    )


@pytest.mark.parametrize("gate_up_context", ("bf16", "mxfp8"))
def test_cutedsl_mxfp8_backward_latent(context, gate_up_context):
    """Latent MoE backward (``functional.backward`` with ``routed_grad_output`` / ``routed_input``, the entry
    ``backward_latent_layer`` wraps) for both Gate / Up context precisions: ``_latent_backward_matches_plain`` on
    the latent geometries, without and with the clamp."""
    rank, world_size, device = context
    if world_size != 8:
        pytest.skip("the MXFP8 backward geometries here are EP8")
    if torch.cuda.get_device_capability(device) not in SUPPORTED_CAPABILITIES:
        pytest.skip("the CuTe DSL MXFP8 backward is SM100/SM103 only")
    mxfp8_bwd_host.clear_persistent_mxfp8_bwd_executor_cache()
    # An ordinary geometry first (routed hidden == hidden): its two executors (clamp off / on) must serve the latent
    # geometries too -- the two hiddens are runtime extents, the latent MoE compiles no variant of its own.
    for swiglu_limit in (None, CLAMP_LIMIT):
        raw, gradients, reference, rows, _ = _backward(rank, world_size, device, GEOMETRIES[0], swiglu_limit, gate_up_context)
        _assert_backward(raw, gradients, reference, rows, GEOMETRIES[0], swiglu_limit, gate_up_context)
    ordinary_executors = dict(mxfp8_bwd_host._EXECUTOR_CACHE)
    assert len(ordinary_executors) == 2
    for geometry in LATENT_GEOMETRIES:
        grads = {limit: _latent_backward_matches_plain(rank, world_size, device, geometry, limit, gate_up_context) for limit in (None, CLAMP_LIMIT)}
        assert not torch.equal(grads[None][6], grads[CLAMP_LIMIT][6]), "d_w_shared_gate"  # the clamp is live
        assert not torch.equal(grads[None][3], grads[CLAMP_LIMIT][3]), "d_w_routed_gate"
    assert mxfp8_bwd_host._EXECUTOR_CACHE.keys() == ordinary_executors.keys()
    assert all(mxfp8_bwd_host._EXECUTOR_CACHE[key] is executor for key, executor in ordinary_executors.items())
    mxfp8_bwd_host.clear_persistent_mxfp8_bwd_executor_cache()  # the other tests count executors by clamp / context
    mxfp8_fwd_host.clear_persistent_mxfp8_executor_cache()


@pytest.mark.parametrize("gate_up_context", ("bf16", "mxfp8"))
def test_cutedsl_mxfp8_backward_single_macrobatch_and_repeat(context, gate_up_context):
    """A geometry without replay (one macrobatch), and the same executor called twice on fresh copies of one
    forward context: the two runs must be byte-identical.  Also the error path of an invalid
    ``gate_up_context``, raised before any barrier."""
    rank, world_size, device = context
    if world_size != 8:
        pytest.skip("the MXFP8 backward geometries here are EP8")
    if torch.cuda.get_device_capability(device) not in SUPPORTED_CAPABILITIES:
        pytest.skip("the CuTe DSL MXFP8 backward is SM100/SM103 only")
    raw, gradients, reference, rows, macrobatches = _backward(rank, world_size, device, SINGLE_MACROBATCH_GEOMETRY, None, gate_up_context)
    assert macrobatches == 1
    _assert_backward(raw, gradients, reference, rows, SINGLE_MACROBATCH_GEOMETRY, None, gate_up_context)
    experts, hidden, intermediate, topk, tokens, comm, minibatch, macrobatch = SINGLE_MACROBATCH_GEOMETRY
    config = f.CuteMoKConfig(
        fwd_num_comm_sms=comm,
        bwd_num_comm_sms=comm,
        minibatch_size=minibatch,
        macrobatch_size=macrobatch,
        schedule_capacity_multiplier=1.5,
        gate_up_context=gate_up_context,
    )
    workspace = f.create_workspace(config, dist.group.WORLD, device=device, num_local_tokens=tokens, hidden_size=hidden, topk=topk)
    x, routes, router_weights, shared_gate, shared_up, shared_down, gate, up, down, d_output = generate_inputs(
        rank, device, experts, experts // world_size, topk, tokens, hidden, intermediate
    )
    schedule = f.build_schedule(workspace, config, routes, num_local_experts=experts // world_size)
    gate_q, up_q, down_q = (tuple(mxfp8_quantize(weight, True, True)) for weight in (gate, up, down))
    shared = (shared_gate, shared_up, shared_down)
    _, ctx = f.forward(config, workspace, schedule, x, router_weights, *shared, gate_q[:2], up_q[:2], down_q[:2], None)
    _sync_ranks(device)
    runs = [_run_raw(config, workspace, schedule, ctx, d_output, x, router_weights, shared, gate_q, up_q, down_q, None, device) for _ in range(2)]
    (outputs_a, buffers_a, _), (outputs_b, buffers_b, _) = runs
    for name, first, second in zip(RAW_OUTPUTS, outputs_a, outputs_b, strict=True):
        assert torch.equal(_bytes(_written(name, first, rows)), _bytes(_written(name, second, rows))), name
    for first, second in zip(buffers_a, buffers_b, strict=True):
        assert torch.equal(_bytes(first), _bytes(second))
    with pytest.raises(ValueError, match="gate_up_context"):
        f.build_schedule(workspace, dataclasses.replace(config, gate_up_context="fp8"), routes, num_local_experts=experts // world_size)


@pytest.mark.parametrize("gate_up_context", ("bf16", "mxfp8"))
def test_cutedsl_mxfp8_backward_recompute_forward_context(context, gate_up_context):
    """``recompute_forward_context``: macrobatch 0 is replayed instead of read from the forward's routed context.  One
    forward, then the low-level backward on a copy of its context and twice with ``recompute_forward_context=True``
    (``_run_raw``: the shared context only, the host's routed scratch poisoned; the second call is the host's
    launch-plan path).  The replay recomputes the forward's x quantization, Gate, Up and Hidden from the same operands,
    so every raw output and both peer-written buffers must be the saved-context run's bytes; the test says so when they
    are merely close.  A geometry with one macrobatch (the whole backward is then replayed) and one with several; the
    communication counters must hold macrobatch 0's x pulls.  A routed context entry passed together with
    ``recompute_forward_context`` is a ``ValueError``."""
    rank, world_size, device = context
    if world_size != 8:
        pytest.skip("the MXFP8 backward geometries here are EP8")
    if torch.cuda.get_device_capability(device) not in SUPPORTED_CAPABILITIES:
        pytest.skip("the CuTe DSL MXFP8 backward is SM100/SM103 only")
    replayed_macrobatches = set()
    for geometry in (SINGLE_MACROBATCH_GEOMETRY, GEOMETRIES[1]):
        experts, hidden, intermediate, topk, tokens, comm, minibatch, macrobatch = geometry
        config = f.CuteMoKConfig(
            fwd_num_comm_sms=comm,
            bwd_num_comm_sms=comm,
            minibatch_size=minibatch,
            macrobatch_size=macrobatch,
            schedule_capacity_multiplier=1.5,
            gate_up_context=gate_up_context,
        )
        workspace = f.create_workspace(config, dist.group.WORLD, device=device, num_local_tokens=tokens, hidden_size=hidden, topk=topk)
        x, routes, router_weights, shared_gate, shared_up, shared_down, gate, up, down, d_output = generate_inputs(
            rank, device, experts, experts // world_size, topk, tokens, hidden, intermediate
        )
        schedule = f.build_schedule(workspace, config, routes, num_local_experts=experts // world_size)
        gate_q, up_q, down_q = (tuple(mxfp8_quantize(weight, True, True)) for weight in (gate, up, down))
        shared = (shared_gate, shared_up, shared_down)
        _, ctx = f.forward(config, workspace, schedule, x, router_weights, *shared, gate_q[:2], up_q[:2], down_q[:2], None)
        _sync_ranks(device)
        num_tokens = int(schedule.num_tokens[0])
        macrobatches = max(1, -(-num_tokens // macrobatch))
        rows = num_tokens - (macrobatches - 1) * macrobatch
        replayed_macrobatches.add(macrobatches > 1)
        saved, *replays = (
            _run_raw(
                config,
                workspace,
                schedule,
                ctx,
                d_output,
                x,
                router_weights,
                shared,
                gate_q,
                up_q,
                down_q,
                None,
                device,
                recompute_forward_context=recompute_forward_context,
            )
            for recompute_forward_context in (False, True, True)
        )
        for call, replayed in enumerate(replays):
            _assert_stream_counters(
                {name: replayed[2][name] for name in STREAM_COUNTERS},
                num_tokens,
                macrobatch,
                minibatch,
                hidden,
                intermediate,
                experts // world_size,
                workspace.schedule_capacity,
                recompute_forward_context=True,
            )
            compared = [
                (name, _written(name, want, rows), _written(name, got, rows)) for name, want, got in zip(RAW_OUTPUTS, saved[0], replayed[0], strict=True)
            ]
            compared += zip(("d_x_routed_buffer", "d_router_weight_buffer"), saved[1], replayed[1], strict=True)
            for name, want, got in compared:
                assert torch.equal(_bytes(want), _bytes(got)), (
                    f"{name} ({macrobatches} macrobatches, recompute_forward_context call {call}): "
                    "not the saved context's result "
                    f"in {(_bytes(want) != _bytes(got)).sum().item()} of {_bytes(want).numel()} bytes "
                    f"(max |difference| {(want.float() - got.float()).abs().max().item():.3e})"
                )
    assert replayed_macrobatches == {False, True}, "one macrobatch and several"
    with pytest.raises(ValueError, match="recompute_forward_context"):
        mxfp8_bwd_host.backward_mxfp8(
            workspace,
            schedule,
            _context_dict(ctx),
            shared_gate,
            tuple(gate_q),
            shared_up,
            tuple(up_q),
            shared_down,
            tuple(down_q[2:]),
            macrobatch_size=config.macrobatch_size,
            minibatch_size=f._bwd_minibatch_size(config),
            num_comm_sms=config.bwd_num_comm_sms,
            recompute_forward_context=True,
        )


def test_cutedsl_mxfp8_backward_two_layers_back_to_back(context):
    """Two layers with different routed weights, the backwards launched back to back on one stream.  The second
    call's TMA descriptors (the routed weights, the context, the state buffers) overwrite the first's in a
    ``tensormaps`` buffer the allocator hands back at the same address, so a descriptor the TMA unit kept from the
    first launch would load the first layer's weights: this is the case the once-per-CTA tensormap-proxy acquire
    exists for.  Each layer's routed Gate / Up weight gradients (which read the routed Down weights through those
    descriptors) must correlate with that layer's BF16 reference and not with the other layer's.  The test records
    both calls' buffer addresses and skips when the allocator did not reuse the block, so a pass means the
    stale-descriptor case was exercised."""
    rank, world_size, device = context
    if world_size != 8:
        pytest.skip("the MXFP8 backward geometries here are EP8")
    if torch.cuda.get_device_capability(device) not in SUPPORTED_CAPABILITIES:
        pytest.skip("the CuTe DSL MXFP8 backward is SM100/SM103 only")
    experts, hidden, intermediate, topk, tokens, comm, minibatch, macrobatch = SINGLE_MACROBATCH_GEOMETRY
    config = f.CuteMoKConfig(
        fwd_num_comm_sms=comm,
        bwd_num_comm_sms=comm,
        minibatch_size=minibatch,
        macrobatch_size=macrobatch,
        schedule_capacity_multiplier=1.5,
        gate_up_context="mxfp8",
    )
    workspace = f.create_workspace(config, dist.group.WORLD, device=device, num_local_tokens=tokens, hidden_size=hidden, topk=topk)
    x, routes, router_weights, shared_gate, shared_up, shared_down, gate, up, down, d_output = generate_inputs(
        rank, device, experts, experts // world_size, topk, tokens, hidden, intermediate
    )
    schedule = f.build_schedule(workspace, config, routes, num_local_experts=experts // world_size)
    shared = (shared_gate, shared_up, shared_down)
    layers, references = [], []
    for shift in (0, 1):  # layer 2: the experts' routed weights rolled by one expert, the output gradient negated
        weights = tuple(torch.roll(weight, shift, dims=0) for weight in (gate, up, down))
        gate_q, up_q, down_q = (tuple(mxfp8_quantize(weight, True, True)) for weight in weights)
        d_y = d_output if shift == 0 else -d_output
        _, ctx = f.forward(config, workspace, schedule, x, router_weights, *shared, gate_q[:2], up_q[:2], down_q[:2], None)
        ctx = _clone_context(ctx)
        _sync_ranks(device)
        layers.append((gate_q, up_q, down_q, d_y, ctx))
        references.append(dict(zip(GRADIENTS, run_reference_bf16(x, routes, router_weights, *shared, *weights, d_y, None)[1:], strict=True)))
    runs = [
        _run_raw(config, workspace, schedule, ctx, d_y, x, router_weights, shared, gate_q, up_q, down_q, None, device)
        for gate_q, up_q, down_q, d_y, ctx in layers
    ]
    for layer, (outputs, _, _) in enumerate(runs):
        named = dict(zip(RAW_OUTPUTS, outputs, strict=True))
        for name in ("d_w_routed_gate", "d_w_routed_up"):
            own = _cosine(references[layer][name], named[name])
            other = _cosine(references[1 - layer][name], named[name])
            assert own >= MIN_COSINE and abs(other) < 0.5, (layer, name, own, other)
    pointers = [state["tensormaps_ptr"] for _, _, state in runs]
    if pointers[0] != pointers[1]:
        pytest.skip(
            f"the allocator did not hand the second backward the first one's tensormaps block "
            f"({pointers[0]:#x} vs {pointers[1]:#x}): the stale-descriptor case was not exercised"
        )


def _native_columnwise(quantized):
    """One ``mxfp8_quantize`` result with its transposed payload in TransformerEngine's layout.

    ``mxfp8_quantize`` physically transposes: for a weight ``(E, N, K)`` it returns the columnwise
    payload as ``(E, K, N)``.  TransformerEngine does not -- its ``columnwise_data`` keeps the
    parameter's own ``(E, N, K)`` shape and only the quantization axis differs, so the same value
    sits at the transposed index.  The scale is already what MOK reads (MCore converts it), and the
    values are identical, so a contiguous transpose of the payload is exactly MCore's input.
    """
    data, scale, data_t, scale_t = quantized
    return (data, scale, data_t.permute(0, 2, 1).contiguous(), scale_t, True)


def test_cutedsl_mxfp8_native_columnwise_backward(context):
    """TransformerEngine's columnwise layout must give the backward the same gradients.

    The dgrads read the transposed weights as an MN-major operand instead of a K-major one: the
    same values, the same scales, the same accumulation order, only a different shared-memory tile
    and one bit of the instruction descriptor.  Nothing about the arithmetic changes, so the two
    runs are compared for byte equality and the test says so when they are merely close.
    """
    rank, world_size, device = context
    if world_size != 8:
        pytest.skip("the MXFP8 geometries here are EP8")
    if torch.cuda.get_device_capability(device) not in SUPPORTED_CAPABILITIES:
        pytest.skip("the CuTe DSL MXFP8 backward is SM100/SM103 only")
    experts, hidden, intermediate, topk, tokens, comm, minibatch, macrobatch = SINGLE_MACROBATCH_GEOMETRY
    config = f.CuteMoKConfig(
        fwd_num_comm_sms=comm, bwd_num_comm_sms=comm, minibatch_size=minibatch, macrobatch_size=macrobatch, schedule_capacity_multiplier=1.5
    )
    workspace = f.create_workspace(config, dist.group.WORLD, device=device, num_local_tokens=tokens, hidden_size=hidden, topk=topk)
    x, routes, router_weights, shared_gate, shared_up, shared_down, gate, up, down, d_output = generate_inputs(
        rank, device, experts, experts // world_size, topk, tokens, hidden, intermediate
    )
    schedule = f.build_schedule(workspace, config, routes, num_local_experts=experts // world_size)
    shared = (shared_gate, shared_up, shared_down)
    gate_q, up_q, down_q = (tuple(mxfp8_quantize(weight, True, True)) for weight in (gate, up, down))
    _, ctx = f.forward(config, workspace, schedule, x, router_weights, *shared, gate_q[:2], up_q[:2], down_q[:2], None)
    _sync_ranks(device)

    def run(gate_w, up_w, down_w):
        gradients = ungated(f.backward(config, workspace, schedule, _clone_context(ctx), d_output, x, router_weights, *shared, gate_w, up_w, down_w, None))
        _sync_ranks(device)
        return tuple(gradient.clone() for gradient in gradients)

    transposed = run(gate_q, up_q, down_q[2:])
    native_gate, native_up, native_down = (_native_columnwise(q) for q in (gate_q, up_q, down_q))
    native = run(native_gate, native_up, (*native_down[2:4], True))

    assert len(transposed) == len(GRADIENTS)
    for name, want, got in zip(GRADIENTS, transposed, native, strict=True):
        assert torch.isfinite(got.float()).all(), f"{name} is not finite"
        assert torch.equal(_bytes(want), _bytes(got)), (
            f"{name}: the MN-major dgrad does not match the K-major one " f"(max |difference| {(want.float() - got.float()).abs().max().item():.3e})"
        )


def test_cutedsl_mxfp8_combined_fc1_backward(context):
    """MCore's combined FC1 through the backward: one [E, 2I, H] parameter passed as Gate and Up.

    The kernel takes every payload and scale whole and reads Up I rows down -- on the rows of the
    rowwise payload and its scale, on the K axis of the transposed ones, where transposing puts
    the combined 2I -- as the C++ port's tile and block offsets do.  Quantization is
    per-row along H and per-32-rows along M, and I is a multiple of both tilings, so the combined
    weight holds exactly the bytes the separate ones do and the gradients must match bit for bit.
    The offset is a runtime argument: the combined call runs the executors the separate weights compiled.
    """
    rank, world_size, device = context
    if world_size != 8:
        pytest.skip("the MXFP8 geometries here are EP8")
    if torch.cuda.get_device_capability(device) not in SUPPORTED_CAPABILITIES:
        pytest.skip("the CuTe DSL MXFP8 backward is SM100/SM103 only")
    experts, hidden, intermediate, topk, tokens, comm, minibatch, macrobatch = SINGLE_MACROBATCH_GEOMETRY
    config = f.CuteMoKConfig(
        fwd_num_comm_sms=comm, bwd_num_comm_sms=comm, minibatch_size=minibatch, macrobatch_size=macrobatch, schedule_capacity_multiplier=1.5
    )
    workspace = f.create_workspace(config, dist.group.WORLD, device=device, num_local_tokens=tokens, hidden_size=hidden, topk=topk)
    x, routes, router_weights, shared_gate, shared_up, shared_down, gate, up, down, d_output = generate_inputs(
        rank, device, experts, experts // world_size, topk, tokens, hidden, intermediate
    )
    schedule = f.build_schedule(workspace, config, routes, num_local_experts=experts // world_size)
    shared = (shared_gate, shared_up, shared_down)
    down_native = _native_columnwise(tuple(mxfp8_quantize(down, True, True)))

    def run(gate_w, up_w):
        # As MCore's runtime does it: the forward's rowwise pair is sliced once and handed over
        # twice, so a combined FC1 reaches MOK as one object and can be recognised as combined.
        gate_forward = gate_w[:2]
        up_forward = gate_forward if up_w is gate_w else up_w[:2]
        _, ctx = f.forward(config, workspace, schedule, x, router_weights, *shared, gate_forward, up_forward, down_native[:2], None)
        _sync_ranks(device)
        gradients = ungated(
            f.backward(config, workspace, schedule, _clone_context(ctx), d_output, x, router_weights, *shared, gate_w, up_w, (*down_native[2:4], True), None)
        )
        _sync_ranks(device)
        return tuple(gradient.clone() for gradient in gradients)

    separate = run(*(_native_columnwise(tuple(mxfp8_quantize(weight, True, True))) for weight in (gate, up)))
    executors = (set(mxfp8_fwd_host._EXECUTOR_CACHE), set(mxfp8_bwd_host._EXECUTOR_CACHE))

    # The same weights quantized as one parameter: Gate rows first, then Up, handed over twice.
    combined_fc1 = torch.cat((gate, up), dim=1).contiguous()
    assert tuple(combined_fc1.shape) == (experts // world_size, 2 * intermediate, hidden)
    combined = _native_columnwise(tuple(mxfp8_quantize(combined_fc1, True, True)))
    merged = run(combined, combined)

    assert len(separate) == len(GRADIENTS)
    for name, want, got in zip(GRADIENTS, separate, merged, strict=True):
        assert torch.equal(_bytes(want), _bytes(got)), (
            f"{name}: the combined FC1 does not match the separate weights " f"(max |difference| {(want.float() - got.float()).abs().max().item():.3e})"
        )
    assert (set(mxfp8_fwd_host._EXECUTOR_CACHE), set(mxfp8_bwd_host._EXECUTOR_CACHE)) == executors, "the combined FC1 compiled executors of its own"
