# Copyright 2026 Cursor Research
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Derived from Mixture-of-Kittens (https://github.com/cursor/mixture-of-kittens, commit 22fc95ae); modified by NVIDIA.

"""The forward megakernels (BF16 and MXFP8 routed weights) against the PyTorch reference.

Run under torchrun on SM100 / SM103, e.g. ``torchrun --standalone --nproc-per-node=8 -m pytest -s
test_forward.py``.  The MXFP8 forward over the test matrix (``utils.shapes`` x ``mok_params`` x
``swiglu_params``; the BF16 forward's matrix is checked with its backward, test_backward.py), the smallest
case and the argument checks of both precisions, the MXFP8 forward's own geometries (EP8), the task -> expert
decode variants and the SwiGLU reference's clamp gradient.
"""

import itertools
from unittest import mock

import pytest
import torch
import torch.distributed as dist

from cudnn.moe_dispatch_mlp_combine import functional
from cudnn.moe_dispatch_mlp_combine import functional as f
from cudnn.moe_dispatch_mlp_combine.cutedsl import persistent_bf16_fwd
from cudnn.moe_dispatch_mlp_combine.cutedsl import persistent_mxfp8_fwd as mxfp8_host
from cudnn.moe_dispatch_mlp_combine.cutedsl._arch import SUPPORTED_CAPABILITIES
from cudnn.moe_dispatch_mlp_combine.cutedsl.persistent_bf16_fwd_contract import EXPERT_TABLE_CAP, TaskDecode
from cudnn.moe_dispatch_mlp_combine.functional import get_workspace
from cudnn.moe_dispatch_mlp_combine.ops import make_routed_scale_storage_table, make_routed_weight_storage_table_mxfp8, mxfp8_quantize

from .utils import (
    BF16_TOLERANCE,
    FORWARD_RESULT_NAMES,
    MXFP8_TOLERANCE,
    capacity_multiplier,
    check_correctness,
    generate_inputs,
    mok_params,
    run_forward_reference_bf16,
    run_fwd_epilogue_reference,
    run_reference_bf16,
    run_swiglu_reference,
    shapes,
    swiglu_params,
)

# --------------------------------------------------------------------------------------------------------------------
# Forward over the test matrix (MXFP8; the BF16 forward's matrix runs with its backward in test_backward.py)


def test_forward_mxfp8(context: tuple[int, int, torch.device]) -> None:
    rank, world_size, device = context
    for shape, mok_param, swiglu_param in itertools.product(shapes(world_size), mok_params(), swiglu_params()):
        shape_name, num_experts, hidden_dim, intermediate_dim, topk, num_local_tokens = shape
        mok_param_name, fwd_num_comm_sms, bwd_num_comm_sms, minibatch_size, macrobatch_size = mok_param
        swiglu_param_name, swiglu_limit = swiglu_param
        assert num_experts % world_size == 0
        num_local_experts = num_experts // world_size
        config = functional.CuteMoKConfig(
            fwd_num_comm_sms=fwd_num_comm_sms,
            bwd_num_comm_sms=bwd_num_comm_sms,
            minibatch_size=minibatch_size,
            macrobatch_size=macrobatch_size,
            schedule_capacity_multiplier=capacity_multiplier(world_size, num_local_tokens, topk, num_local_experts),
        )
        workspace = get_workspace(
            config,
            dist.group.WORLD,
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
            _,
        ) = inputs
        mok_schedule = functional.build_schedule(
            workspace,
            config,
            topk_experts,
            num_local_experts=num_local_experts,
        )
        w_routed_gate_fp8, w_routed_gate_sc, _, _ = mxfp8_quantize(w_routed_gate, True, False)
        w_routed_up_fp8, w_routed_up_sc, _, _ = mxfp8_quantize(w_routed_up, True, False)
        w_routed_down_fp8, w_routed_down_sc, _, _ = mxfp8_quantize(w_routed_down, True, False)

        output, forward_context = functional.forward(
            config,
            workspace,
            mok_schedule,
            x,
            router_weights,
            w_shared_gate,
            w_shared_up,
            w_shared_down,
            (w_routed_gate_fp8, w_routed_gate_sc),
            (w_routed_up_fp8, w_routed_up_sc),
            (w_routed_down_fp8, w_routed_down_sc),
            swiglu_limit,
        )
        (
            reference_combine_buffer,
            reference_gate_shared,
            reference_up_shared,
            reference_hidden_shared,
            reference_y_shared,
        ) = run_forward_reference_bf16(
            x,
            topk_experts,
            w_shared_gate,
            w_shared_up,
            w_shared_down,
            w_routed_gate,
            w_routed_up,
            w_routed_down,
            swiglu_limit,
        )
        reference = (
            run_fwd_epilogue_reference(
                reference_y_shared,
                reference_combine_buffer,
                router_weights,
            ),
            reference_gate_shared,
            reference_up_shared,
            reference_hidden_shared,
        )
        actual = (
            output,
            forward_context.gate_shared,
            forward_context.up_shared,
            forward_context.hidden_shared,
        )
        for name, expected, result in zip(FORWARD_RESULT_NAMES, reference, actual, strict=True):
            check_correctness(
                f"{shape_name}/{mok_param_name}/{swiglu_param_name}/{name}",
                expected,
                result,
                MXFP8_TOLERANCE,
                print_stats=rank == 0,
            )
        # Free this case before the next generate_inputs: at EP1 one case of a 384-expert shape holds
        # about 200 GiB (inputs, quantized weights, results and the reference) on the single GPU.
        del inputs, reference, actual, forward_context, mok_schedule, output, w_routed_gate_fp8
        del w_routed_up_fp8, w_routed_down_fp8, w_routed_gate_sc, w_routed_up_sc, w_routed_down_sc, x
        del topk_experts, router_weights, w_shared_gate, w_shared_up, w_shared_down, w_routed_gate
        del w_routed_up, w_routed_down
        torch.cuda.empty_cache()

    num_experts = world_size
    num_local_experts = 1
    hidden_dim = 256
    intermediate_dim = 256
    topk = 1
    num_local_tokens = 512
    config = functional.CuteMoKConfig(
        fwd_num_comm_sms=2,
        bwd_num_comm_sms=2,
        minibatch_size=256,
        macrobatch_size=256,
        schedule_capacity_multiplier=capacity_multiplier(world_size, num_local_tokens, topk, num_local_experts),
        all_gather_top_experts_chunk_bytes=16,
    )
    workspace = get_workspace(
        config,
        dist.group.WORLD,
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
        _,
    ) = inputs
    mok_schedule = functional.build_schedule(
        workspace,
        config,
        topk_experts,
        num_local_experts=num_local_experts,
    )
    w_routed_gate_fp8, w_routed_gate_sc, _, _ = mxfp8_quantize(w_routed_gate, True, False)
    w_routed_up_fp8, w_routed_up_sc, _, _ = mxfp8_quantize(w_routed_up, True, False)
    w_routed_down_fp8, w_routed_down_sc, _, _ = mxfp8_quantize(w_routed_down, True, False)
    valid_kwargs = {
        "config": config,
        "workspace": workspace,
        "schedule": mok_schedule,
        "x": x,
        "router_weights": router_weights,
        "shared_gate_weights": w_shared_gate,
        "shared_up_weights": w_shared_up,
        "shared_down_weights": w_shared_down,
        "routed_gate_weights": (w_routed_gate_fp8, w_routed_gate_sc),
        "routed_up_weights": (w_routed_up_fp8, w_routed_up_sc),
        "routed_down_weights": (w_routed_down_fp8, w_routed_down_sc),
    }

    output, forward_context = functional.forward(**valid_kwargs)
    (
        reference_combine_buffer,
        reference_gate_shared,
        reference_up_shared,
        reference_hidden_shared,
        reference_y_shared,
    ) = run_forward_reference_bf16(
        x,
        topk_experts,
        w_shared_gate,
        w_shared_up,
        w_shared_down,
        w_routed_gate,
        w_routed_up,
        w_routed_down,
    )
    reference = (
        run_fwd_epilogue_reference(
            reference_y_shared,
            reference_combine_buffer,
            router_weights,
        ),
        reference_gate_shared,
        reference_up_shared,
        reference_hidden_shared,
    )
    actual = (
        output,
        forward_context.gate_shared,
        forward_context.up_shared,
        forward_context.hidden_shared,
    )
    for name, expected, result in zip(FORWARD_RESULT_NAMES, reference, actual, strict=True):
        check_correctness(
            f"All combined minimums/{name}",
            expected,
            result,
            MXFP8_TOLERANCE,
            print_stats=rank == 0,
        )

    invalid_schedule = functional.CuteMoKSchedule(
        peer_rank=mok_schedule.peer_rank[:-256],
        peer_token_idx=mok_schedule.peer_token_idx[:-256],
        num_tokens=mok_schedule.num_tokens,
        tokens_per_expert=mok_schedule.tokens_per_expert,
    )
    for failure_name, overrides, expected_exception in (
        ("config type", {"config": object()}, TypeError),
        ("workspace type", {"workspace": object()}, TypeError),
        ("schedule type", {"schedule": object()}, TypeError),
        ("input dtype", {"x": x.float()}, ValueError),
        ("input shape", {"x": x[:-256]}, ValueError),
        (
            "router weight dtype",
            {"router_weights": router_weights.to(torch.bfloat16)},
            ValueError,
        ),
        (
            "router weight shape",
            {"router_weights": router_weights[:, :0]},
            ValueError,
        ),
        (
            "schedule capacity",
            {"schedule": invalid_schedule},
            ValueError,
        ),
    ):
        try:
            functional.forward(**(valid_kwargs | overrides))
        except expected_exception:
            pass
        else:
            raise AssertionError(f"{failure_name} should have failed")


def test_forward_bf16(context: tuple[int, int, torch.device]) -> None:
    rank, world_size, device = context
    num_experts = world_size
    num_local_experts = 1
    hidden_dim = 256
    intermediate_dim = 256
    topk = 1
    num_local_tokens = 512
    config = functional.CuteMoKConfig(
        fwd_num_comm_sms=2,
        bwd_num_comm_sms=2,
        minibatch_size=256,
        macrobatch_size=256,
        schedule_capacity_multiplier=capacity_multiplier(world_size, num_local_tokens, topk, num_local_experts),
        all_gather_top_experts_chunk_bytes=16,
    )
    workspace = get_workspace(
        config,
        dist.group.WORLD,
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
        _,
    ) = inputs
    mok_schedule = functional.build_schedule(
        workspace,
        config,
        topk_experts,
        num_local_experts=num_local_experts,
    )
    valid_kwargs = {
        "config": config,
        "workspace": workspace,
        "schedule": mok_schedule,
        "x": x,
        "router_weights": router_weights,
        "shared_gate_weights": w_shared_gate,
        "shared_up_weights": w_shared_up,
        "shared_down_weights": w_shared_down,
        "routed_gate_weights": w_routed_gate,
        "routed_up_weights": w_routed_up,
        "routed_down_weights": w_routed_down,
    }

    output, forward_context = functional.forward(**valid_kwargs)
    (
        reference_combine_buffer,
        reference_gate_shared,
        reference_up_shared,
        reference_hidden_shared,
        reference_y_shared,
    ) = run_forward_reference_bf16(
        x,
        topk_experts,
        w_shared_gate,
        w_shared_up,
        w_shared_down,
        w_routed_gate,
        w_routed_up,
        w_routed_down,
    )
    reference = (
        run_fwd_epilogue_reference(
            reference_y_shared,
            reference_combine_buffer,
            router_weights,
        ),
        reference_gate_shared,
        reference_up_shared,
        reference_hidden_shared,
    )
    actual = (
        output,
        forward_context.gate_shared,
        forward_context.up_shared,
        forward_context.hidden_shared,
    )
    for name, expected, result in zip(FORWARD_RESULT_NAMES, reference, actual, strict=True):
        check_correctness(
            f"All combined minimums/{name}",
            expected,
            result,
            BF16_TOLERANCE,
            print_stats=rank == 0,
        )

    invalid_schedule = functional.CuteMoKSchedule(
        peer_rank=mok_schedule.peer_rank[:-256],
        peer_token_idx=mok_schedule.peer_token_idx[:-256],
        num_tokens=mok_schedule.num_tokens,
        tokens_per_expert=mok_schedule.tokens_per_expert,
    )
    for failure_name, overrides, expected_exception in (
        ("config type", {"config": object()}, TypeError),
        ("workspace type", {"workspace": object()}, TypeError),
        ("schedule type", {"schedule": object()}, TypeError),
        ("input dtype", {"x": x.float()}, ValueError),
        ("input shape", {"x": x[:-256]}, ValueError),
        (
            "router weight dtype",
            {"router_weights": router_weights.to(torch.bfloat16)},
            ValueError,
        ),
        (
            "router weight shape",
            {"router_weights": router_weights[:, :0]},
            ValueError,
        ),
        (
            "schedule capacity",
            {"schedule": invalid_schedule},
            ValueError,
        ),
    ):
        try:
            functional.forward(**(valid_kwargs | overrides))
        except expected_exception:
            pass
        else:
            raise AssertionError(f"{failure_name} should have failed")


# --------------------------------------------------------------------------------------------------------------------
# MXFP8 forward
#
# GPU-only EP8 MXFP8 forward through its low-level entry.
#
# Run under torchrun on SM100 / SM103 (8 GPUs).  ``forward_mxfp8`` sees the inputs, schedule, entry
# barrier and buffer copies that ``functional.forward`` gives it.  It returns thirteen tensors with
# ``gate_up_context="mxfp8"`` (the default) and eleven with ``"bf16"`` (the routed Gate / Up context is
# then one BF16 tensor each); both variants run.  The combine buffer is zeroed and the ranks
# synchronized before every run, so a row the kernel failed to send stays zero and the check that the
# routed rows landed is meaningful.  Six geometries (see ``GEOMETRIES``) run through one compiled
# executor per clamp mode, and the clamp limit is one the inputs reach, so the clamped and unclamped
# results must differ.


def _bytes(tensor):
    return tensor.contiguous().view(torch.uint8)


# (experts, hidden, intermediate, top-k, local tokens, comm SMs, minibatch, macrobatch).  Every rank
# receives about tokens x top-k routed rows, so the macrobatch sizes give two, four and two
# macrobatches: the stream's pulls of one macrobatch interleaved with the pushes of the one after it, the
# y_routed_done counts of the macrobatches above 0, and macrobatch 0 (the compared ring rows) as the stream's
# last segment.  H = 768 is three 256-column blocks, a partial last column group (groups of four); H = 1280 is
# five blocks, a group of four and a group of one.  The next two hold every routed row in one macrobatch (pulls,
# then pushes, no y_routed_done arrivals).
GEOMETRIES = (
    (16, 512, 512, 2, 1024, 8, 512, 1024),
    (64, 1024, 768, 4, 2048, 16, 1024, 2048),
    (16, 768, 512, 2, 1024, 8, 512, 1024),
    (16, 1280, 512, 2, 1024, 8, 512, 1024),
    (16, 768, 512, 2, 1024, 8, 512, 4096),
    (64, 1024, 768, 4, 2048, 16, 1024, 16384),
    (16, 512, 512, 1, 512, 8, 256, 1024),  # topk 1: 16 tiles x 1 column group = 32 stream groups < 56 stream warps
)


# With unit-variance inputs and 1/sqrt(H)-scaled weights the Gate / Up activations are of unit
# order, so a limit of 1.0 clamps a large share of them (7.0 would clamp none).
CLAMP_LIMIT = 1.0


# The return order with gate_up_context="mxfp8" (persistent_mxfp8_fwd.PersistentMxfp8State.abi_outputs).
OUTPUTS = (
    "x_routed_t",
    "x_routed_sc_t",
    "gate_shared",
    "gate_routed_fp8",
    "gate_routed_sc",
    "up_shared",
    "up_routed_fp8",
    "up_routed_sc",
    "hidden_shared",
    "hidden_routed_t",
    "hidden_routed_sc_t",
    "y_shared",
    "y_routed",
)


# The return order with gate_up_context="bf16": the routed Gate / Up context one BF16 tensor each.
CUTE_OUTPUTS = (
    "x_routed_t",
    "x_routed_sc_t",
    "gate_shared",
    "gate_routed_bf16",
    "up_shared",
    "up_routed_bf16",
    "hidden_shared",
    "hidden_routed_t",
    "hidden_routed_sc_t",
    "y_shared",
    "y_routed",
)


CUTE_CONTEXT_OUTPUTS = CUTE_OUTPUTS[:9]


CONTEXT_OUTPUTS = OUTPUTS[:11]  # gate_up_context="mxfp8"


def _thirteen(outputs):
    """The eleven outputs of the BF16 context as the thirteen of the e4m3 one, the Gate / Up context quantized."""
    named = dict(zip(CUTE_OUTPUTS, outputs, strict=True))
    for name in ("gate", "up"):
        named[f"{name}_routed_fp8"], named[f"{name}_routed_sc"] = mxfp8_quantize(named[f"{name}_routed_bf16"], True, False)[:2]
    return [named[name] for name in OUTPUTS]


def _sync_ranks(device):
    torch.cuda.synchronize(device)
    dist.barrier()


def _run(config, workspace, schedule, x, router_weights, shared, routed, swiglu_limit, device, gate_up_context="bf16"):
    """The forward through its low-level entry, as ``functional.forward`` runs it.

    Returns the thirteen outputs in ``OUTPUTS`` order (``_thirteen`` for the BF16 context), this rank's combine
    buffer after every peer sent its rows, the epilogue output, and the call's communication counters
    (``x_routed_ready`` with the stream's head word, ``y_routed_done``).  The combine buffer starts zeroed on
    every rank.
    """
    shared_gate, shared_up, shared_down = shared
    gate, up, down = routed
    ops = f._device_ops(config)
    workspace.x_buffer.copy_(x)
    workspace.router_weight_buffer.copy_(router_weights)
    workspace.combine_buffer.zero_()
    _sync_ranks(device)  # every peer's combine buffer is zero before anyone sends
    ops.barrier_all(workspace.barrier_buffer, workspace.barrier_buffer_ptrs, workspace.barrier_buffer_multicast_ptr, workspace.barrier_target)
    # The call's state, from the one function both host paths allocate it with (the full path through
    # prepare_persistent_mxfp8_state, a repeated call through its launch plan).
    states = []
    allocate = mxfp8_host._allocate_state

    def capture(*args, **kwargs):
        states.append(allocate(*args, **kwargs))
        return states[-1]

    with mock.patch.object(mxfp8_host, "_allocate_state", capture):
        outputs = mxfp8_host.forward_mxfp8(
            workspace,
            schedule,
            shared_gate,
            gate,
            shared_up,
            up,
            shared_down,
            down,
            macrobatch_size=config.macrobatch_size,
            minibatch_size=f._fwd_minibatch_size(config),
            swiglu_limit=swiglu_limit,
            num_comm_sms=config.fwd_num_comm_sms,
            gate_up_context=gate_up_context,
        )
    ops.barrier_all(workspace.barrier_buffer, workspace.barrier_buffer_ptrs, workspace.barrier_buffer_multicast_ptr, workspace.barrier_target)
    bf16_context = gate_up_context == "bf16"
    names = CUTE_OUTPUTS if bf16_context else OUTPUTS
    assert len(outputs) == len(names)
    y = ops.fwd_epilogue(outputs[names.index("y_shared")], workspace.combine_buffer, workspace.router_weight_buffer, schedule.top_experts)
    _sync_ranks(device)
    counters = {name: getattr(states[0], name).clone() for name in ("x_routed_ready", "y_routed_done")}
    # The context's storages hold the context only, no buffer of the call.
    context = outputs[: len(CUTE_CONTEXT_OUTPUTS) if bf16_context else len(CONTEXT_OUTPUTS)]
    storages = {t.untyped_storage().data_ptr(): t.untyped_storage().nbytes() for t in context}
    assert sum(storages.values()) == sum(t.nbytes for t in context)
    # A gated forward context keeps y_shared until the backward: its storage holds it alone, not
    # the call's macrobatch-sized y_routed too.
    y_shared = outputs[names.index("y_shared")]
    assert y_shared.untyped_storage().nbytes() == y_shared.nbytes
    if bf16_context:
        assert all(outputs[CUTE_OUTPUTS.index(name)].dtype == torch.bfloat16 for name in ("gate_routed_bf16", "up_routed_bf16"))
        outputs = _thirteen(outputs)
    return [t.clone() for t in outputs], workspace.combine_buffer.clone(), y.clone(), counters


def _stream_blocks(num_tokens, macrobatch, hidden):
    """The forward stream's block count: 2 x (last macrobatch's tiles + full tiles x earlier macrobatches) x column
    groups; the head word ends at it (every group is claimed once, the first ones statically)."""
    last_macro = -(-num_tokens // macrobatch) - 1
    last_tiles = min(macrobatch, num_tokens - last_macro * macrobatch) // 32
    column_blocks = -(-hidden // 256)
    groups = -(-column_blocks // 4)
    return 2 * (last_tiles + last_macro * (macrobatch // 32)) * groups


def _assert_stream_counters(counters, num_tokens, macrobatch, minibatch, hidden):
    """The communication side's arrivals after one forward: ``x_routed_ready[g]`` = the pull blocks of minibatch
    ``g`` (ceil(rows / 32) x ceil(H / 256)), ``y_routed_done[b]`` = 4 x ceil(H / 256) for the 128-row blocks of the
    macrobatches above 0 and 0 for macrobatch 0's (no next macrobatch overwrites y_routed) and the unused tail, and
    the head word (the last element of ``x_routed_ready``) = the stream's block count."""
    column_blocks = -(-hidden // 256)
    x_ready, y_done = counters["x_routed_ready"].tolist(), counters["y_routed_done"].tolist()
    expected = []
    for g in range(len(x_ready) - 1):
        rows = max(0, min(minibatch, num_tokens - g * minibatch))
        expected.append(-(-rows // 32) * column_blocks)
    assert x_ready[:-1] == expected, (x_ready, expected)
    assert x_ready[-1] == _stream_blocks(num_tokens, macrobatch, hidden), (x_ready[-1], num_tokens)
    expected = [4 * column_blocks if macrobatch <= b * 128 < num_tokens else 0 for b in range(len(y_done))]
    assert y_done == expected, (y_done, expected)


def _forward(rank, world_size, device, geometry, swiglu_limit, gate_up_context="bf16"):
    experts, hidden, intermediate, topk, tokens, comm, minibatch, macrobatch = geometry
    config = f.CuteMoKConfig(fwd_num_comm_sms=comm, minibatch_size=minibatch, macrobatch_size=macrobatch, schedule_capacity_multiplier=1.5)
    workspace = f.create_workspace(config, dist.group.WORLD, device=device, num_local_tokens=tokens, hidden_size=hidden, topk=topk)
    x, routes, router_weights, shared_gate, shared_up, shared_down, gate, up, down, _ = generate_inputs(
        rank, device, experts, experts // world_size, topk, tokens, hidden, intermediate
    )
    schedule = f.build_schedule(workspace, config, routes, num_local_experts=experts // world_size)
    routed = [mxfp8_quantize(weight, True, False)[:2] for weight in (gate, up, down)]
    result = _run(config, workspace, schedule, x, router_weights, (shared_gate, shared_up, shared_down), routed, swiglu_limit, device, gate_up_context)
    num_tokens = int(schedule.num_tokens[0])
    _assert_stream_counters(result[3], num_tokens, config.macrobatch_size, f._fwd_minibatch_size(config), hidden)
    return result, min(num_tokens, config.macrobatch_size)


def _written(name, tensor, rows):
    """The part of an output its kernel writes: macrobatch 0's rows of the routed ring buffers
    (the tail is unwritten in both backends), everything else whole."""
    if name in ("x_routed_t", "hidden_routed_t"):  # transposed (columns, macrobatch)
        return tensor[:, :rows]
    if name in ("x_routed_sc_t", "hidden_routed_sc_t"):  # (columns / 128, macrobatch / 128, 32, 16)
        return tensor[:, : rows // 128]
    if name in ("gate_routed_fp8", "up_routed_fp8", "y_routed"):  # (macrobatch, columns)
        return tensor[:rows]
    if name in ("gate_routed_sc", "up_routed_sc"):  # (macrobatch / 128, columns / 128, 32, 16)
        return tensor[: rows // 128]
    return tensor


def _assert_rows_written(result):
    """y_routed's destination: the rows every peer sent into this rank's (zeroed) combine buffer.

    Routes dropped at capacity leave their row zero, so a few are expected; a kernel that sent
    nothing would leave nearly all of them zero.
    """
    _, combine, _, _ = result
    written_rows = combine.view(torch.int16).ne(0).any(dim=1).float().mean().item()
    assert written_rows > 0.9, f"only {written_rows:.3f} of the combine rows were written"


def test_cutedsl_mxfp8_combined_fc1(context):
    """One combined [E, 2I, H] MXFP8 FC1, passed as both Gate and Up.

    The payload halves are strided views, but the scale is not: its tcgen05 layout is expert-major
    over M blocks, so the Up half is not a view of the Gate half.  The kernel takes the whole scale
    and an M-block offset instead, deriving the per-expert stride from the scale's own extent.  The
    result must be what two separate weights give, bit for bit, from the executor the separate weights
    compiled: where Up starts is a runtime argument.
    """
    rank, world_size, device = context
    if world_size != 8:
        pytest.skip("the MXFP8 forward geometries here are EP8")
    if torch.cuda.get_device_capability(device) not in SUPPORTED_CAPABILITIES:
        pytest.skip("the CuTe DSL MXFP8 forward is SM100/SM103 only")
    experts, hidden, intermediate, topk, tokens, comm, minibatch, macrobatch = GEOMETRIES[0]
    local_experts = experts // world_size
    config = f.CuteMoKConfig(fwd_num_comm_sms=comm, minibatch_size=minibatch, macrobatch_size=macrobatch, schedule_capacity_multiplier=1.5)
    workspace = f.create_workspace(config, dist.group.WORLD, device=device, num_local_tokens=tokens, hidden_size=hidden, topk=topk)
    x, routes, router_weights, shared_gate, shared_up, shared_down, gate, up, down, _ = generate_inputs(
        rank, device, experts, local_experts, topk, tokens, hidden, intermediate
    )
    schedule = f.build_schedule(workspace, config, routes, num_local_experts=local_experts)
    shared = (shared_gate, shared_up, shared_down)

    separate = [mxfp8_quantize(weight, True, False)[:2] for weight in (gate, up, down)]
    separate_output, separate_context = f.forward(config, workspace, schedule, x, router_weights, *shared, *separate, None)
    executors = set(mxfp8_host._EXECUTOR_CACHE)

    # The same weights quantized as one parameter: Gate rows first, then Up.
    combined_fc1 = torch.cat((gate, up), dim=1).contiguous()
    assert tuple(combined_fc1.shape) == (local_experts, 2 * intermediate, hidden)
    combined_pair = mxfp8_quantize(combined_fc1, True, False)[:2]
    down_pair = separate[2]
    combined_output, combined_context = f.forward(config, workspace, schedule, x, router_weights, *shared, combined_pair, combined_pair, down_pair, None)

    assert torch.equal(_bytes(separate_output), _bytes(combined_output)), "output"
    for name in ("gate_shared", "up_shared", "hidden_shared"):
        assert torch.equal(_bytes(getattr(separate_context, name)), _bytes(getattr(combined_context, name))), name
    for name in ("gate_routed", "up_routed"):
        for index, (want, got) in enumerate(zip(getattr(separate_context, name), getattr(combined_context, name), strict=True)):
            assert torch.equal(_bytes(want), _bytes(got)), f"{name}[{index}]"
    assert set(mxfp8_host._EXECUTOR_CACHE) == executors, "the combined FC1 compiled an executor of its own"


def _split_mxfp8(payload, scale, local_experts, order=None):
    """A ``SplitRoutedWeight`` over the per-expert slices of one grouped MXFP8 weight.

    Both arms then read the same bytes -- the slices are views, not copies -- so the comparison
    is exact and the only thing under test is how the kernel finds each expert.  An MXFP8 expert
    is a payload *and* a scale, and the scale's tcgen05 layout is expert-major over M blocks, so
    expert e's scale is the row-block slice that MOK would have given it anyway.
    """
    order = tuple(range(local_experts) if order is None else order)
    blocks = scale.shape[0] // local_experts
    payloads = [payload[index] for index in order]
    scales = [scale[index * blocks : (index + 1) * blocks] for index in order]
    return f.SplitRoutedWeight(
        data=payloads[0],
        storage_table=make_routed_weight_storage_table_mxfp8(payloads),
        scale=scales[0],
        scale_storage_table=make_routed_scale_storage_table(scales),
        scale_tensors=tuple(scales),
    )


def _mxfp8_split_case(context):
    """The EP8 geometry, inputs, schedule and grouped MXFP8 weights the split tests compare to."""
    rank, world_size, device = context
    if world_size != 8:
        pytest.skip("the MXFP8 forward geometries here are EP8")
    if torch.cuda.get_device_capability(device) not in SUPPORTED_CAPABILITIES:
        pytest.skip("the CuTe DSL MXFP8 forward is SM100/SM103 only")
    experts, hidden, intermediate, topk, tokens, comm, minibatch, macrobatch = GEOMETRIES[0]
    local_experts = experts // world_size
    assert local_experts >= 2, "a table that reads one expert everywhere would pass vacuously"
    config = f.CuteMoKConfig(fwd_num_comm_sms=comm, minibatch_size=minibatch, macrobatch_size=macrobatch, schedule_capacity_multiplier=1.5)
    workspace = f.create_workspace(config, dist.group.WORLD, device=device, num_local_tokens=tokens, hidden_size=hidden, topk=topk)
    x, routes, router_weights, shared_gate, shared_up, shared_down, gate, up, down, _ = generate_inputs(
        rank, device, experts, local_experts, topk, tokens, hidden, intermediate
    )
    schedule = f.build_schedule(workspace, config, routes, num_local_experts=local_experts)
    return (config, workspace, schedule, x, router_weights, (shared_gate, shared_up, shared_down), (gate, up, down), local_experts)


def _mxfp8_run(case, routed):
    config, workspace, schedule, x, router_weights, shared, _, _ = case
    output, context = f.forward(config, workspace, schedule, x, router_weights, *shared, *routed, None)
    return (output, context.gate_shared, context.up_shared, context.hidden_shared)


def _mxfp8_identical(expected, actual, what):
    """Compare only what is defined everywhere.

    The routed context buffers are macrobatch-sized scratch written where there are tokens, so
    comparing them whole compares uninitialized memory (see test_split_weights.py).
    ``output`` is the combine of every routed expert, so an expert read from the wrong address
    cannot fail to show up in it.
    """
    for name, want, got in zip(("output", "gate_shared", "up_shared", "hidden_shared"), expected, actual, strict=True):
        assert torch.equal(_bytes(want), _bytes(got)), (
            f"{what}: {name} differs; the two arms read the same memory, so this is an addressing " "error, not a numerical one"
        )


def test_cutedsl_mxfp8_split_matches_grouped(context):
    """Split per-expert allocations must give the grouped layout's result, bit for bit."""
    case = _mxfp8_split_case(context)
    local_experts = case[-1]
    grouped = [mxfp8_quantize(weight, True, False)[:2] for weight in case[6]]
    expected = _mxfp8_run(case, grouped)
    split = [_split_mxfp8(payload, scale, local_experts) for payload, scale in grouped]
    _mxfp8_identical(expected, _mxfp8_run(case, split), "split MXFP8 routed weights")


def test_cutedsl_mxfp8_split_follows_the_table(context):
    """Reverse the expert order in the tables; the result must follow them.

    Reading expert 0 everywhere is not the only way to be wrong.  Requiring the split arm to
    reproduce a grouped run over the reversed parameter pins every expert to its own payload
    *and* its own scale, rather than merely to "not expert 0".
    """
    case = _mxfp8_split_case(context)
    local_experts = case[-1]
    reversed_weights = [weight.flip(0).contiguous() for weight in case[6]]
    expected = _mxfp8_run(case, [mxfp8_quantize(w, True, False)[:2] for w in reversed_weights])
    grouped = [mxfp8_quantize(weight, True, False)[:2] for weight in case[6]]
    split = [_split_mxfp8(payload, scale, local_experts, order=reversed(range(local_experts))) for payload, scale in grouped]
    _mxfp8_identical(expected, _mxfp8_run(case, split), "a reversed MXFP8 expert table")


def test_cutedsl_mxfp8_split_combined_fc1(context):
    """One ``[2I, H]`` payload per expert for Gate and Up, which is what MCore owns.

    Two offsets at once, and they are of different kinds: the Up *payload* starts half a
    parameter into each allocation (a byte offset on the address table), while the Up *scale*
    starts ``I / 128`` M blocks into the same scale (a block offset), because the tcgen05 layout
    is expert-major and the second half is not a strided view of the first.
    """
    case = _mxfp8_split_case(context)
    local_experts = case[-1]
    gate, up, down = case[6]
    combined = torch.cat((gate, up), dim=1).contiguous()
    combined_pair = mxfp8_quantize(combined, True, False)[:2]
    down_pair = mxfp8_quantize(down, True, False)[:2]
    expected = _mxfp8_run(case, (combined_pair, combined_pair, down_pair))

    combined_split = _split_mxfp8(*combined_pair, local_experts)
    down_split = _split_mxfp8(*down_pair, local_experts)
    _mxfp8_identical(expected, _mxfp8_run(case, (combined_split, combined_split, down_split)), "a split combined MXFP8 FC1")


@pytest.mark.parametrize("gate_up_context", ("bf16", "mxfp8"))
def test_cutedsl_mxfp8_forward(context, gate_up_context):
    """Every geometry, both clamp modes: the routed rows land, and the clamp changes the result."""
    rank, world_size, device = context
    if world_size != 8:
        pytest.skip("the MXFP8 forward geometries here are EP8")
    if torch.cuda.get_device_capability(device) not in SUPPORTED_CAPABILITIES:
        pytest.skip("the CuTe DSL MXFP8 forward is SM100/SM103 only")
    hidden_shared = OUTPUTS.index("hidden_shared")
    cached_before = set(mxfp8_host._EXECUTOR_CACHE)  # other tests of this module may have compiled this variant already
    for geometry in GEOMETRIES:
        outputs = {}
        for swiglu_limit in (None, CLAMP_LIMIT):
            result, _ = _forward(rank, world_size, device, geometry, swiglu_limit, gate_up_context)
            _assert_rows_written(result)
            outputs[swiglu_limit] = result
        # The clamp is live: the clamped Hidden and the clamped output differ from the unclamped ones.
        assert not torch.equal(outputs[None][0][hidden_shared], outputs[CLAMP_LIMIT][0][hidden_shared])
        assert not torch.equal(outputs[None][2], outputs[CLAMP_LIMIT][2])
    # Sizes are runtime arguments: at most one new executor per clamp mode and Gate / Up context, whatever the geometry.
    # Only the variant this test compiles is counted: the routed-weight layout, SiTU, a shared expert and the
    # tail code are other compile-time axes other tests compile their own executors for.
    this_variant = (("gate_up_context", gate_up_context), ("routed_weights", "grouped"), ("is_situ", False), ("has_shared", True), ("has_tail", False))
    for is_clamped in (False, True):
        assert (
            sum(
                ("is_clamped", is_clamped) in key and all(axis in key for axis in this_variant)
                for key in mxfp8_host._EXECUTOR_CACHE
                if key not in cached_before
            )
            <= 1
        )


# Latent MoE (workspace.routed_hidden_size != hidden_size): (experts, hidden, routed hidden, intermediate, top-k,
# local tokens, comm SMs, minibatch, macrobatch).  The routed chain works in the routed hidden -- a half-filled
# 512-column dispatch tile at 768, wider than the shared hidden at 1280 -- the shared expert in the hidden.
LATENT_GEOMETRIES = (
    (16, 1024, 512, 512, 2, 1024, 8, 512, 1024),
    (64, 1024, 768, 768, 4, 2048, 16, 1024, 2048),
    (16, 512, 1280, 512, 2, 1024, 8, 512, 1024),
    (16, 512, 256, 512, 2, 1024, 8, 512, 1024),  # the minimal routed hidden: one 256-column block
)


ROUTED_CONTEXT = (
    "x_routed_t",
    "x_routed_sc_t",
    "gate_routed_fp8",
    "gate_routed_sc",
    "up_routed_fp8",
    "up_routed_sc",
    "hidden_routed_t",
    "hidden_routed_sc_t",
)


SHARED_CONTEXT = ("gate_shared", "up_shared", "hidden_shared")


def _latent_layer_matches_plain(rank, world_size, device, geometry, swiglu_limit, gate_up_context="bf16"):
    """One latent geometry: ``forward_latent_layer`` with MXFP8 routed weights against two ordinary forwards.

    The routed chain (x quantized in both layouts, the Gate / Up and Hidden context, the rows sent into the peers'
    combine buffers) must equal the forward run on ``routed_input`` with hidden == routed hidden: the same rows,
    weights and quantization.  The shared expert (its Gate / Up / Hidden context, y_shared) must equal the forward
    run on ``x`` with the shared hidden.  The layer's output is the two halves through the BF16 latent GEMMs.  One
    schedule serves the three workspaces (same tokens, top-k and capacity), so the routed rows land in the same
    order.
    """
    experts, hidden, routed_hidden, intermediate, topk, tokens, comm, minibatch, macrobatch = geometry
    local_experts = experts // world_size
    config = f.CuteMoKConfig(
        fwd_num_comm_sms=comm, minibatch_size=minibatch, macrobatch_size=macrobatch, schedule_capacity_multiplier=1.5, gate_up_context=gate_up_context
    )
    latent = f.create_workspace(
        config, dist.group.WORLD, device=device, num_local_tokens=tokens, hidden_size=hidden, topk=topk, routed_hidden_size=routed_hidden
    )
    in_routed = f.create_workspace(config, dist.group.WORLD, device=device, num_local_tokens=tokens, hidden_size=routed_hidden, topk=topk)
    in_hidden = f.create_workspace(config, dist.group.WORLD, device=device, num_local_tokens=tokens, hidden_size=hidden, topk=topk)
    x, routes, router_weights, shared_gate, shared_up, shared_down, gate_h, up_h, down_h, _ = generate_inputs(
        rank, device, experts, local_experts, topk, tokens, hidden, intermediate
    )
    _, _, _, shared_gate_r, shared_up_r, shared_down_r, gate, up, down, _ = generate_inputs(
        rank, device, experts, local_experts, topk, tokens, routed_hidden, intermediate
    )
    generator = torch.Generator(device=device).manual_seed(1234)
    w_down = torch.randn(routed_hidden, hidden, device=device, generator=generator) / hidden**0.5
    w_up = torch.randn(hidden, routed_hidden, device=device, generator=generator) / routed_hidden**0.5
    w_down, w_up = w_down.to(torch.bfloat16), w_up.to(torch.bfloat16)
    routed_input = torch.matmul(x, w_down.t())  # forward_latent_layer's own first GEMM
    schedule = f.build_schedule(latent, config, routes, num_local_experts=local_experts)
    routed = [tuple(mxfp8_quantize(weight, True, False)[:2]) for weight in (gate, up, down)]
    routed_h = [tuple(mxfp8_quantize(weight, True, False)[:2]) for weight in (gate_h, up_h, down_h)]
    # The two references, one hidden each (``_run``: the entry barrier and buffer copies of functional.forward).
    plain_routed, combine_routed, _, _ = _run(
        config, in_routed, schedule, routed_input, router_weights, (shared_gate_r, shared_up_r, shared_down_r), routed, swiglu_limit, device, gate_up_context
    )
    plain_shared, _, _, _ = _run(
        config, in_hidden, schedule, x, router_weights, (shared_gate, shared_up, shared_down), routed_h, swiglu_limit, device, gate_up_context
    )
    plain_routed = dict(zip(OUTPUTS, plain_routed, strict=True))
    plain_shared = dict(zip(OUTPUTS, plain_shared, strict=True))
    # The latent layer through the public entry, its combine buffer zero on every rank before anyone sends.
    latent.combine_buffer.zero_()
    _sync_ranks(device)
    output, layer_context = f.forward_latent_layer(
        config, latent, schedule, x, router_weights, w_down, w_up, shared_gate, shared_up, shared_down, *routed, swiglu_limit
    )
    _sync_ranks(device)
    forward_context = layer_context.mok
    rows = min(int(schedule.num_tokens[0]), macrobatch)
    # The routed chain, byte for byte (the BF16 Gate / Up context quantized as ``_thirteen`` quantizes it).
    x_t, x_sc_t = forward_context.x_routed
    hidden_t, hidden_sc_t = forward_context.hidden_routed
    if gate_up_context == "mxfp8":
        (gate_fp8, gate_sc), (up_fp8, up_sc) = forward_context.gate_routed, forward_context.up_routed
    else:
        gate_fp8, gate_sc = mxfp8_quantize(forward_context.gate_routed, True, False)[:2]
        up_fp8, up_sc = mxfp8_quantize(forward_context.up_routed, True, False)[:2]
    actual_routed = dict(zip(ROUTED_CONTEXT, (x_t, x_sc_t, gate_fp8, gate_sc, up_fp8, up_sc, hidden_t, hidden_sc_t), strict=True))
    for name in ROUTED_CONTEXT:
        expected, actual = plain_routed[name], actual_routed[name]
        assert expected.shape == actual.shape and expected.dtype == actual.dtype, name
        assert torch.equal(_bytes(_written(name, expected, rows)), _bytes(_written(name, actual, rows))), name
    assert torch.equal(_bytes(combine_routed), _bytes(latent.combine_buffer)), "combine_buffer"
    written_rows = combine_routed.view(torch.int16).ne(0).any(dim=1).float().mean().item()
    assert written_rows > 0.9, f"only {written_rows:.3f} of the combine rows were written"
    # The shared expert, byte for byte.
    for name, actual in zip(SHARED_CONTEXT, (forward_context.gate_shared, forward_context.up_shared, forward_context.hidden_shared), strict=True):
        assert torch.equal(_bytes(plain_shared[name]), _bytes(actual)), name
    # The layer's output from the two halves: y_shared + (routed top-k reduction) @ W_up^T.
    ops = f._device_ops(config)
    y_routed = ops.fwd_epilogue(
        torch.zeros(tokens, routed_hidden, dtype=torch.bfloat16, device=device), combine_routed, latent.router_weight_buffer, schedule.top_experts
    )
    assert torch.equal(_bytes(y_routed), _bytes(layer_context.y_routed)), "y_routed"
    expected = torch.addmm(plain_shared["y_shared"], y_routed, w_up.t())
    assert tuple(output.shape) == (tokens, hidden)
    assert torch.equal(_bytes(expected), _bytes(output)), "output"
    return output


@pytest.mark.parametrize("gate_up_context", ("bf16", "mxfp8"))
def test_cutedsl_mxfp8_forward_latent(context, gate_up_context):
    """Latent MoE (``forward_latent_layer`` with MXFP8 routed weights) against the ordinary forward, byte for byte:
    ``_latent_layer_matches_plain`` on the latent geometries, without and with the clamp, both Gate / Up context
    precisions."""
    rank, world_size, device = context
    if world_size != 8:
        pytest.skip("the MXFP8 forward geometries here are EP8")
    if torch.cuda.get_device_capability(device) not in SUPPORTED_CAPABILITIES:
        pytest.skip("the CuTe DSL MXFP8 forward is SM100/SM103 only")
    mxfp8_host.clear_persistent_mxfp8_executor_cache()
    # An ordinary geometry first (routed hidden == hidden): its two executors (clamp off / on) must serve the latent
    # geometries too -- the two hiddens are runtime extents, the latent MoE compiles no variant of its own.
    for swiglu_limit in (None, CLAMP_LIMIT):
        result, _ = _forward(rank, world_size, device, GEOMETRIES[0], swiglu_limit, gate_up_context)
        _assert_rows_written(result)
    ordinary_executors = dict(mxfp8_host._EXECUTOR_CACHE)
    assert len(ordinary_executors) == 2
    for geometry in LATENT_GEOMETRIES:
        outputs = {limit: _latent_layer_matches_plain(rank, world_size, device, geometry, limit, gate_up_context) for limit in (None, CLAMP_LIMIT)}
        assert not torch.equal(outputs[None], outputs[CLAMP_LIMIT])  # the clamp is live
    assert mxfp8_host._EXECUTOR_CACHE.keys() == ordinary_executors.keys()
    assert all(mxfp8_host._EXECUTOR_CACHE[key] is executor for key, executor in ordinary_executors.items())
    mxfp8_host.clear_persistent_mxfp8_executor_cache()  # the non-latent tests count executors by clamp / context


# --------------------------------------------------------------------------------------------------------------------
# BF16 forward task decode variants
#
# The three FWD task->expert decode variants (TaskDecode) give the same forward.
#
# Run under torchrun on SM100 / SM103 (any world size; EP1 covers 512 local experts), e.g.
# ``torchrun --standalone --nproc-per-node=1 -m pytest -s moe_dispatch_mlp_combine/cutedsl/test_forward.py``.
# For every shape the auto-selected variant's output is checked against the PyTorch BF16 reference
# (``run_reference_bf16``) with the BF16 tolerance -- the CUDA backend it was once compared with is
# gone, and a second run of the port is no check -- and every valid forced variant (SMEM_BINARY only when the local experts fit
# the shared-memory prefix) must reproduce the auto variant's outputs bit for bit.


# (name, local experts, experts that receive tokens (per rank), hidden, intermediate, top-k, local tokens).
# With 4096 tokens x top-k 8 = 32768 routed rows per rank the mean rows per receiving expert are
# 32768 / routed experts; a 2048-row minibatch holds 8 row blocks.  Coverage:
#   stride 1 (E <= 256, SMEM_BINARY auto) with one block per expert (256) and with multi-block experts
#   (64: ~2 blocks; 16: ~8 blocks, so experts span minibatch boundaries).  These two use H = 4096 and
#   I = 2048, i.e. 16 column blocks for FC1 and Down: the tile swizzle (supergroups of 8 columns) only
#   differs from row-major order with more than 8 column blocks, and only then does a wrong row-block
#   origin or row count inside a minibatch move tiles;
#   stride 2 with a full last group (512), a short last group (257, 383) and empty experts including a
#   whole empty last group (383 with tokens for the first 300 only; EP1 only, local expert indices);
#   stride 3 / 4 (600 / 769) with a short last group, and 600 with multi-block experts and empty groups.
CASES = (
    ("512 local experts (Qwen3.5 shape, stride 2)", 512, 512, 4096, 1024, 10, 4096),
    ("256 local experts (hidden 6144, stride 1)", 256, 256, 6144, 2048, 8, 2048),
    ("64 local experts, ~2 row blocks each, 16 column blocks (stride 1)", 64, 64, 4096, 2048, 8, 4096),
    ("16 local experts, ~8 row blocks each, spanning minibatches, 16 column blocks (stride 1)", 16, 16, 4096, 2048, 8, 4096),
    ("257 local experts (stride 2, last group of one)", 257, 257, 256, 256, 8, 4096),
    ("383 local experts (stride 2, last group of one)", 383, 383, 256, 256, 8, 4096),
    ("383 local experts, experts 300..382 empty (empty last group)", 383, 300, 256, 256, 8, 4096),
    ("600 local experts (stride 3, groups of 3, last group of three)", 600, 600, 256, 256, 8, 4096),
    ("600 local experts, tokens for the first 100 only (stride 3, multi-block experts, empty groups)", 600, 100, 256, 256, 8, 4096),
    ("769 local experts (stride 4, last group of one)", 769, 769, 256, 256, 8, 4096),
)


def _forward_with_task_decode(config, workspace, schedule, inputs, task_decode):
    """functional.forward with the task->expert decode forced to ``task_decode`` (None = the host's
    choice) by patching the internal selector, the only selection point of forward_bf16."""
    x, _, router_weights, w_shared_gate, w_shared_up, w_shared_down, w_routed_gate, w_routed_up, w_routed_down, _ = inputs
    select = persistent_bf16_fwd.select_task_decode
    forced = select if task_decode is None else (lambda num_local_experts, requested=None: select(num_local_experts, task_decode))
    with mock.patch.object(persistent_bf16_fwd, "select_task_decode", forced):
        output, context = functional.forward(
            config, workspace, schedule, x, router_weights, w_shared_gate, w_shared_up, w_shared_down, w_routed_gate, w_routed_up, w_routed_down, None
        )
    torch.cuda.synchronize(x.device)  # a device fault is reported against this call, not a later one
    return (
        output,
        context.x_routed,
        context.gate_shared,
        context.gate_routed,
        context.up_shared,
        context.up_routed,
        context.hidden_shared,
        context.hidden_routed,
    )


def test_shared_storage_layout(context: tuple[int, int, torch.device]) -> None:
    """The expert table lies after every mbarrier and before the 1024-aligned arena (the module
    asserts it at import); print the layout so the padding after the table is visible."""
    rank, _, device = context
    if torch.cuda.get_device_capability(device) not in SUPPORTED_CAPABILITIES:
        pytest.skip("CuTe DSL backends are SM100/SM103 only")
    from cudnn.moe_dispatch_mlp_combine.cutedsl import _persistent_bf16_fwd_mega as mega

    layout = mega.shared_storage_layout()
    by_name = {name: (offset, size) for name, offset, size in layout}
    table_offset, table_bytes = by_name["expert_cu_end"]
    arena_offset = by_name["arena"][0]
    assert table_bytes == EXPERT_TABLE_CAP * 4
    assert all(offset + size <= table_offset for name, (offset, size) in by_name.items() if "mbarrier" in name)
    assert table_offset + table_bytes <= arena_offset and arena_offset % 1024 == 0
    # The decoded-task record ring (CLC warp -> other warps): every field has one Int32 word, the
    # stages are 16-byte aligned and lie with the barriers before the table.
    record_offset, record_bytes = by_name["task_info"]
    assert len(set(mega.TASK_INFO_FIELDS)) == len(mega.TASK_INFO_FIELDS) <= mega.TASK_INFO_WORDS
    assert record_bytes == mega.TASK_INFO_STAGES * mega.TASK_INFO_WORDS * 4 and record_offset % 16 == 0
    assert record_offset + record_bytes <= table_offset
    if rank == 0:
        for name, offset, size in layout:
            print(f"SMEM {name}: offset {offset} size {size} end {offset + size}")
        print(f"SMEM padding after expert_cu_end: {arena_offset - table_offset - table_bytes} B before the arena at {arena_offset}")


def test_task_decode_variants_agree(context: tuple[int, int, torch.device]) -> None:
    rank, world_size, device = context
    if torch.cuda.get_device_capability(device) not in SUPPORTED_CAPABILITIES:
        pytest.skip("CuTe DSL backends are SM100/SM103 only")
    for name, num_local_experts, routed_local_experts, hidden_dim, intermediate_dim, topk, num_local_tokens in CASES:
        if routed_local_experts != num_local_experts and world_size != 1:
            continue  # restricted routing is expressed in local expert indices
        num_experts = num_local_experts * world_size
        # Schedule capacity: every local expert is padded to a 256-row block, so the padded row count
        # can reach E_local * 256 + routed rows (the CUDA scheduler traps above the capacity).  The
        # capacity factor is max(2, ceil(world * multiplier)) times T * top-k.
        routed_rows = num_local_tokens * topk
        capacity_factor = -(-(num_local_experts * 256 + routed_rows * world_size) // routed_rows) + 1
        geometry = dict(fwd_num_comm_sms=30, bwd_num_comm_sms=14, macrobatch_size=8192, schedule_capacity_multiplier=capacity_factor / world_size)
        dsl_config = functional.CuteMoKConfig(fwd_minibatch_size=2048, bwd_minibatch_size=2048, **geometry)
        workspace = get_workspace(dsl_config, dist.group.WORLD, device=device, num_local_tokens=num_local_tokens, hidden_size=hidden_dim, topk=topk)
        # Routing over the first routed_local_experts * world_size experts only; the weights
        # cover all num_local_experts, so the remaining experts receive no rows.
        inputs = generate_inputs(rank, device, routed_local_experts * world_size, num_local_experts, topk, num_local_tokens, hidden_dim, intermediate_dim)
        topk_experts = inputs[1]
        dsl_schedule = functional.build_schedule(workspace, dsl_config, topk_experts, num_local_experts=num_local_experts)

        auto = persistent_bf16_fwd.select_task_decode(num_local_experts)
        expected_auto = TaskDecode.SMEM_BINARY if num_local_experts <= EXPERT_TABLE_CAP else TaskDecode.SMEM_GROUP
        assert auto is expected_auto, (name, num_local_experts, auto)
        stride = -(-num_local_experts // EXPERT_TABLE_CAP)
        expected_output = run_reference_bf16(*inputs)[0]
        reference = [t.clone() for t in _forward_with_task_decode(dsl_config, workspace, dsl_schedule, inputs, None)]
        check_correctness(f"{name}/auto={auto.name} vs reference", expected_output, reference[0], BF16_TOLERANCE, print_stats=rank == 0)
        for variant in TaskDecode:
            if variant is TaskDecode.SMEM_BINARY and num_local_experts > EXPERT_TABLE_CAP:
                with pytest.raises(ValueError):
                    persistent_bf16_fwd.select_task_decode(num_local_experts, variant)
                continue
            result = _forward_with_task_decode(dsl_config, workspace, dsl_schedule, inputs, variant)
            for index, (want, got) in enumerate(zip(reference, result, strict=True)):
                assert torch.equal(want, got), f"{name}/{variant.name}: output {index} differs from {auto.name}"
            if rank == 0:
                print(f"{name}: {variant.name} == {auto.name} (E_local={num_local_experts}, stride {stride})")
        del workspace, inputs, reference, result
        torch.cuda.empty_cache()


# --------------------------------------------------------------------------------------------------------------------
# MXFP8 forward task decode variants
#
# The MXFP8 forward's task -> expert decode variants (TaskDecode) give the same bytes.
#
# Run under torchrun on SM100 / SM103 (8 GPUs), like ``test_forward.py`` whose entries it
# reuses.  The MXFP8 forward compiles one kernel per decode variant (the BF16 forward's
# ``TaskDecode``: the shared expert table searched per expert for E <= EXPERT_TABLE_CAP local experts,
# per expert group above it, or the linear scan).  For every geometry the host's choice is checked, and
# every forced variant must reproduce its thirteen outputs (the BF16 Gate / Up context quantized by
# ``mxfp8_quantize``, as ``test_forward.py`` does), the combine buffer and the epilogue
# output byte for byte.  Both Gate / Up context precisions run.  A variant is forced by patching the
# host's selector, the only selection point of ``forward_mxfp8``.  Each geometry's executors are
# cleared afterwards, so the other MXFP8 tests' executor counts do not depend on test order.


# (name, experts, hidden, intermediate, top-k, local tokens, comm SMs, minibatch, macrobatch); the
# local expert count is experts / world size (8): 16 and 64 with several row blocks per expert, 256 at
# the table's capacity (stride 1), 257 and 384 above it (stride 2, a short and a full last group).
MXFP8_CASES = (
    ("16 local experts, several row blocks each", 128, 1024, 768, 4, 4096, 16, 2048, 8192),
    ("64 local experts, 9 column blocks (the tile swizzle differs from row-major order)", 512, 2304, 2304, 4, 2048, 16, 1024, 4096),
    ("64 local experts (Qwen-sized)", 512, 1024, 512, 4, 2048, 16, 1024, 4096),
    ("256 local experts (the table's capacity, stride 1)", 2048, 512, 512, 2, 4096, 16, 1024, 8192),
    ("257 local experts (stride 2, last group of one)", 2056, 512, 512, 2, 4096, 16, 1024, 8192),
    ("384 local experts (stride 2, full last group)", 3072, 512, 512, 2, 4096, 16, 1024, 8192),
)


def _mxfp8_forward_with_task_decode(config, workspace, schedule, inputs, task_decode, device, gate_up_context="bf16"):
    """One backend's forward through its low-level entry; ``task_decode`` forces the MXFP8 kernel's
    decode variant (None = the host's choice)."""
    x, router_weights, shared, routed = inputs
    select = mxfp8_host.select_task_decode
    if task_decode is None:
        forced = select
    else:

        def forced(num_local_experts, requested=None):
            return select(num_local_experts, task_decode)

    with mock.patch.object(mxfp8_host, "select_task_decode", forced):
        return _run(config, workspace, schedule, x, router_weights, shared, routed, None, device, gate_up_context)[:3]


def _assert_same(name, want, got, rows):
    outputs_want, combine_want, y_want = want
    outputs_got, combine_got, y_got = got
    for output, expected, actual in zip(OUTPUTS, outputs_want, outputs_got, strict=True):
        assert torch.equal(_bytes(_written(output, expected, rows)), _bytes(_written(output, actual, rows))), f"{name}: {output}"
    assert torch.equal(_bytes(combine_want), _bytes(combine_got)), f"{name}: combine_buffer"
    assert torch.equal(_bytes(y_want), _bytes(y_got)), f"{name}: output"


@pytest.mark.parametrize("gate_up_context", ("bf16", "mxfp8"))
def test_cutedsl_mxfp8_task_decode_variants_agree(context, gate_up_context):
    rank, world_size, device = context
    if world_size != 8:
        pytest.skip("the MXFP8 forward geometries here are EP8")
    if torch.cuda.get_device_capability(device) not in SUPPORTED_CAPABILITIES:
        pytest.skip("the CuTe DSL MXFP8 forward is SM100/SM103 only")
    mxfp8_host.clear_persistent_mxfp8_executor_cache()
    for name, experts, hidden, intermediate, topk, tokens, comm, minibatch, macrobatch in MXFP8_CASES:
        local_experts = experts // world_size
        # Every local expert is padded to a 256-row block, so the schedule must hold
        # local_experts * 256 rows plus the routed rows of every rank.
        routed_rows = tokens * topk
        capacity_factor = -(-(local_experts * 256 + routed_rows * world_size) // routed_rows) + 1
        config = f.CuteMoKConfig(
            fwd_num_comm_sms=comm, minibatch_size=minibatch, macrobatch_size=macrobatch, schedule_capacity_multiplier=capacity_factor / world_size
        )
        workspace = f.create_workspace(config, dist.group.WORLD, device=device, num_local_tokens=tokens, hidden_size=hidden, topk=topk)
        x, routes, router_weights, shared_gate, shared_up, shared_down, gate, up, down, _ = generate_inputs(
            rank, device, experts, local_experts, topk, tokens, hidden, intermediate
        )
        schedule = f.build_schedule(workspace, config, routes, num_local_experts=local_experts)
        inputs = (x, router_weights, (shared_gate, shared_up, shared_down), [mxfp8_quantize(weight, True, False)[:2] for weight in (gate, up, down)])
        rows = min(int(schedule.num_tokens[0]), config.macrobatch_size)

        auto = mxfp8_host.select_task_decode(local_experts)
        expected = TaskDecode.SMEM_BINARY if local_experts <= EXPERT_TABLE_CAP else TaskDecode.SMEM_GROUP
        assert auto is expected, (name, local_experts, auto)
        reference = _mxfp8_forward_with_task_decode(config, workspace, schedule, inputs, None, device, gate_up_context)
        for variant in TaskDecode:
            if variant is TaskDecode.SMEM_BINARY and local_experts > EXPERT_TABLE_CAP:
                with pytest.raises(ValueError):
                    mxfp8_host.select_task_decode(local_experts, variant)
                continue
            result = _mxfp8_forward_with_task_decode(config, workspace, schedule, inputs, variant, device, gate_up_context)
            _assert_same(f"{name}/{variant.name} vs {auto.name}", reference, result, rows)
            if rank == 0:
                print(f"{name}: {variant.name} == {auto.name} (E_local={local_experts})")
        del workspace, inputs, reference, result
        mxfp8_host.clear_persistent_mxfp8_executor_cache()
        torch.cuda.empty_cache()


# --------------------------------------------------------------------------------------------------------------------
# SwiGLU reference
#
# Pin MOK's inclusive clamp subgradient independently of torch.clamp backward.


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("component", ["gate", "up"])
def test_clamped_swiglu_reference_boundary_gradient(context, dtype, component):
    device = context[2]
    limit = 4.0
    lo = torch.tensor(-limit, device=device, dtype=dtype)
    hi = -lo
    negative_inf = torch.full_like(lo, -float("inf"))
    positive_inf = -negative_inf
    # Include both exact boundaries and their nearest representable neighbors.
    values = torch.stack(
        (
            2 * lo,
            torch.nextafter(lo, negative_inf),
            lo,
            torch.nextafter(lo, positive_inf),
            torch.zeros_like(lo),
            torch.nextafter(hi, negative_inf),
            hi,
            torch.nextafter(hi, positive_inf),
            2 * hi,
        )
    )
    gate = (values if component == "gate" else torch.ones_like(values)).requires_grad_()
    up = (values if component == "up" else torch.ones_like(values)).requires_grad_()
    dy = torch.tensor([1, -2, 0.5, 2, -1, 0.25, -0.5, 4, 1], device=device, dtype=dtype)
    output = run_swiglu_reference(gate, up, limit)
    d_gate, d_up = torch.autograd.grad(output, (gate, up), dy)

    # torch.clamp is used only for forward values, never its backward rule.
    gate_value = gate.detach().float().clamp(max=limit)
    up_value = up.detach().float().clamp(min=-limit, max=limit)
    silu_value = torch.nn.functional.silu(gate_value)
    expected_output = (silu_value * up_value).to(dtype)
    torch.testing.assert_close(output, expected_output, rtol=0, atol=0)

    # Hand-written masks pin the inclusive boundary contract, not a version probe.
    if component == "gate":
        gate_mask = [1, 1, 1, 1, 1, 1, 1, 0, 0]
        up_mask = [1] * 9
    else:
        gate_mask = [1] * 9
        up_mask = [0, 0, 1, 1, 1, 1, 1, 0, 0]
    sigmoid = torch.sigmoid(gate_value)
    dsilu = sigmoid * (1 + gate_value * (1 - sigmoid))
    expected_gate = (dy.float() * up_value * dsilu * torch.tensor(gate_mask, device=device)).to(dtype)
    expected_up = (dy.float() * silu_value * torch.tensor(up_mask, device=device)).to(dtype)
    tolerance = dict(rtol=2e-6, atol=1e-7) if dtype == torch.float32 else dict(rtol=0, atol=0)
    torch.testing.assert_close(d_gate, expected_gate, **tolerance)
    torch.testing.assert_close(d_up, expected_up, **tolerance)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_unclamped_swiglu_reference_unchanged(context, dtype):
    gate = torch.tensor([-5, -4, 0, 4, 5], device=context[2], dtype=dtype, requires_grad=True)
    up = torch.tensor([-4, 2, 1, 4, 5], device=context[2], dtype=dtype, requires_grad=True)
    actual = run_swiglu_reference(gate, up)
    expected = torch.nn.functional.silu(gate) * up
    actual_grads = torch.autograd.grad(actual.sum(), (gate, up))
    expected_grads = torch.autograd.grad(expected.sum(), (gate, up))
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    for actual_grad, expected_grad in zip(actual_grads, expected_grads):
        torch.testing.assert_close(actual_grad, expected_grad, rtol=0, atol=0)
