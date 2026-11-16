# Copyright 2026 Cursor Research
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Derived from Mixture-of-Kittens (https://github.com/cursor/mixture-of-kittens, commit 22fc95ae); modified by NVIDIA.

"""The standalone operators of ``cudnn.moe_dispatch_mlp_combine.ops`` and their kernels: all-gather of the routes, the schedule,
MXFP8 quantization, the forward and backward epilogues and the routed-row staging; and the fused megakernels
through the C++ port's op signatures.

Op tests run under torchrun on SM100 / SM103 (``context``); the kernel tests of the scheduler, the epilogues and
the quantizer need one GPU, e.g. ``python -m pytest moe_dispatch_mlp_combine/cutedsl/test_ops.py``.
"""

from __future__ import annotations

import itertools
import math
import unittest
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist

from cudnn.moe_dispatch_mlp_combine import functional, ops
from cudnn.moe_dispatch_mlp_combine.cutedsl._arch import SUPPORTED_CAPABILITIES, TARGET_ARCH, TARGET_CAPABILITY
from cudnn.moe_dispatch_mlp_combine.functional import get_workspace
from cudnn.moe_dispatch_mlp_combine.ops import all_gather_top_experts, barrier_all, bwd_epilogue, fwd_epilogue, mxfp8_quantize, schedule

from .utils import (
    capacity_multiplier,
    check_correctness,
    generate_inputs,
    generate_topk_experts,
    mok_params,
    run_all_gather_top_experts_reference,
    run_bwd_epilogue_reference,
    run_forward_reference_bf16,
    run_fwd_epilogue_reference,
    run_mxfp8_quantize_reference,
    run_reference_bf16,
    run_schedule_reference,
    shapes,
)

# --------------------------------------------------------------------------------------------------------------------
# The C++ port's fused-op signatures over the CuTe DSL entries
#
# The C++ port's fused-op ABI over the CuTe DSL entries, for the op tests below.
#
# The C++ build exposes its megakernels as four torch ops taking raw buffers, pointer lists and
# schedule tensors (``mok.ops.dispatch_mlp_swiglu_combine_{fwd,bwd}_{bf16,mxfp8}`` at d59188a).  This
# port has no such ops: ``functional`` calls ``forward_bf16`` / ``forward_mxfp8`` / ``backward_bf16`` /
# ``backward_mxfp8`` with a workspace and a schedule.  These adapters take exactly the C++ ops'
# arguments, build the workspace / schedule / context views those entries read, call them and return
# the outputs in the C++ ops' order, so the C++ op tests run here unchanged.
#
# Two things the C++ signatures do not carry are derived, not invented:
#
# * ``valid_tokens_per_expert`` (the BF16 forward's per-expert unpadded row count, the CuTe DSL
#   scheduler's fifth output): the scheduler marks every unrouted entry of the schedule -1, so it is
#   the number of entries with ``peer_rank >= 0`` in each expert's padded segment.  test_ops'
#   test_schedule checks that this derivation reproduces the scheduler's own output.
# * the workspace's rank: the position of the local buffer in its pointer list.
#
# The C++ ops' SiTU convention -- ``situ_beta1 == situ_beta2 == 0`` means SwiGLU -- maps to
# ``situ_beta=None``.  The C++ BF16 backward returns fifteen tensors where this one has thirteen (no
# materialized dHidden, the SwiGLU backward runs in the Down dgrad epilogue); those two slots are None.


def _rank_of(pointers, buffer):
    try:
        return list(pointers).index(buffer.data_ptr())
    except ValueError:
        return -1  # rejected by the entries' ownership check, as a malformed call should be


def _situ(beta1, beta2):
    return None if beta1 == 0.0 and beta2 == 0.0 else (beta1, beta2)


def _valid_tokens_per_expert(peer_rank, num_tokens, tokens_per_expert):
    """Unpadded routed rows per local expert: the schedule entries with a peer, per padded segment."""
    rows = int(num_tokens.reshape(-1)[0].item())
    segment_ends = torch.cumsum(tokens_per_expert.to(torch.int64), 0)
    positions = torch.arange(rows, device=peer_rank.device)
    expert = torch.searchsorted(segment_ends, positions, right=True)
    routed = peer_rank[:rows] >= 0
    return torch.bincount(expert[routed], minlength=tokens_per_expert.numel()).to(torch.int32)


def _forward_workspace(x, x_ptrs, combine_buffer, combine_buffer_ptrs, topk, schedule_capacity):
    return SimpleNamespace(
        device=x.device,
        ep_size=len(x_ptrs),
        ep_rank=_rank_of(x_ptrs, x),
        num_local_tokens=x.shape[0],
        hidden_size=x.shape[-1],
        topk=topk,
        schedule_capacity=schedule_capacity,
        x_buffer=x,
        x_buffer_ptrs=list(x_ptrs),
        combine_buffer=combine_buffer,
        combine_buffer_ptrs=list(combine_buffer_ptrs),
    )


def _backward_workspace(
    x,
    x_ptrs,
    d_y_buffer,
    d_y_buffer_ptrs,
    d_x_routed_buffer,
    d_x_routed_buffer_ptrs,
    router_weight_buffer,
    router_weight_buffer_ptrs,
    d_router_weight_buffer,
    d_router_weight_buffer_ptrs,
    topk,
    schedule_capacity,
):
    return SimpleNamespace(
        device=x.device,
        ep_size=len(x_ptrs),
        ep_rank=_rank_of(x_ptrs, x),
        num_local_tokens=x.shape[0],
        hidden_size=x.shape[-1],
        topk=topk,
        schedule_capacity=schedule_capacity,
        x_buffer=x,
        x_buffer_ptrs=list(x_ptrs),
        d_y_buffer=d_y_buffer,
        d_y_buffer_ptrs=list(d_y_buffer_ptrs),
        d_x_routed_buffer=d_x_routed_buffer,
        d_x_routed_buffer_ptrs=list(d_x_routed_buffer_ptrs),
        router_weight_buffer=router_weight_buffer,
        router_weight_buffer_ptrs=list(router_weight_buffer_ptrs),
        d_router_weight_buffer=d_router_weight_buffer,
        d_router_weight_buffer_ptrs=list(d_router_weight_buffer_ptrs),
        _cutedsl_bwd_preparation=None,
    )


def _schedule(peer_rank, peer_token_idx, num_tokens, tokens_per_expert, valid=None):
    return SimpleNamespace(
        peer_rank=peer_rank, peer_token_idx=peer_token_idx, num_tokens=num_tokens, tokens_per_expert=tokens_per_expert, valid_tokens_per_expert=valid
    )


def dispatch_mlp_swiglu_combine_fwd_bf16(
    x,
    x_ptrs,
    combine_buffer,
    combine_buffer_ptrs,
    w_shared_gate,
    w_routed_gate,
    w_shared_up,
    w_routed_up,
    w_shared_down,
    w_routed_down,
    schedule_peer_rank,
    schedule_peer_token_idx,
    num_tokens,
    tokens_per_expert,
    topk,
    swiglu_limit,
    num_comm_sms,
    macrobatch_size,
    minibatch_size,
    w_routed_gate_storage_table=None,
    w_routed_up_storage_table=None,
    w_routed_down_storage_table=None,
    situ_beta1=0.0,
    situ_beta2=0.0,
):
    """C++ ``dispatch_mlp_swiglu_combine_fwd_bf16``: the nine outputs, x_routed .. y_routed."""
    from cudnn.moe_dispatch_mlp_combine.cutedsl.persistent_bf16_fwd import forward_bf16

    workspace = _forward_workspace(x, x_ptrs, combine_buffer, combine_buffer_ptrs, topk, schedule_peer_rank.numel())
    valid = _valid_tokens_per_expert(schedule_peer_rank, num_tokens, tokens_per_expert) if schedule_peer_rank.ndim == 1 else None
    schedule = _schedule(schedule_peer_rank, schedule_peer_token_idx, num_tokens, tokens_per_expert, valid)
    tables = None if w_routed_gate_storage_table is None else (w_routed_gate_storage_table, w_routed_up_storage_table, w_routed_down_storage_table)
    return forward_bf16(
        workspace,
        schedule,
        w_shared_gate,
        w_routed_gate,
        w_shared_up,
        w_routed_up,
        w_shared_down,
        w_routed_down,
        macrobatch_size=macrobatch_size,
        minibatch_size=minibatch_size,
        swiglu_limit=swiglu_limit,
        num_comm_sms=num_comm_sms,
        routed_storage_tables=tables,
        situ_beta=_situ(situ_beta1, situ_beta2),
    )


def dispatch_mlp_swiglu_combine_fwd_mxfp8(
    x,
    x_ptrs,
    combine_buffer,
    combine_buffer_ptrs,
    w_shared_gate,
    w_routed_gate,
    w_routed_gate_sc,
    w_shared_up,
    w_routed_up,
    w_routed_up_sc,
    w_shared_down,
    w_routed_down,
    w_routed_down_sc,
    schedule_peer_rank,
    schedule_peer_token_idx,
    num_tokens,
    tokens_per_expert,
    topk,
    swiglu_limit,
    num_comm_sms,
    macrobatch_size,
    minibatch_size,
    w_routed_gate_storage_table=None,
    w_routed_up_storage_table=None,
    w_routed_down_storage_table=None,
    w_routed_gate_sc_storage_table=None,
    w_routed_up_sc_storage_table=None,
    w_routed_down_sc_storage_table=None,
    situ_beta1=0.0,
    situ_beta2=0.0,
):
    """C++ ``dispatch_mlp_swiglu_combine_fwd_mxfp8``: the thirteen outputs, x_fp8_t_routed .. y_routed."""
    from cudnn.moe_dispatch_mlp_combine.cutedsl.persistent_mxfp8_fwd import forward_mxfp8

    workspace = _forward_workspace(x, x_ptrs, combine_buffer, combine_buffer_ptrs, topk, schedule_peer_rank.numel())
    schedule = _schedule(schedule_peer_rank, schedule_peer_token_idx, num_tokens, tokens_per_expert)
    tables = (
        None
        if w_routed_gate_storage_table is None
        else (
            w_routed_gate_storage_table,
            w_routed_up_storage_table,
            w_routed_down_storage_table,
            w_routed_gate_sc_storage_table,
            w_routed_up_sc_storage_table,
            w_routed_down_sc_storage_table,
        )
    )
    return forward_mxfp8(
        workspace,
        schedule,
        w_shared_gate,
        (w_routed_gate, w_routed_gate_sc),
        w_shared_up,
        (w_routed_up, w_routed_up_sc),
        w_shared_down,
        (w_routed_down, w_routed_down_sc),
        macrobatch_size=macrobatch_size,
        minibatch_size=minibatch_size,
        swiglu_limit=swiglu_limit,
        num_comm_sms=num_comm_sms,
        routed_storage_tables=tables,
        situ_beta=_situ(situ_beta1, situ_beta2),
    )


def dispatch_mlp_swiglu_combine_bwd_bf16(
    d_y_buffer,
    d_y_buffer_ptrs,
    d_x_routed_buffer,
    d_x_routed_buffer_ptrs,
    router_weight_buffer,
    router_weight_buffer_ptrs,
    d_router_weight_buffer,
    d_router_weight_buffer_ptrs,
    w_shared_gate,
    w_routed_gate,
    w_shared_up,
    w_routed_up,
    w_shared_down,
    w_routed_down,
    x_routed,
    gate_shared,
    gate_routed,
    up_shared,
    up_routed,
    hidden_shared,
    hidden_routed,
    x,
    x_ptrs,
    schedule_peer_rank,
    schedule_peer_token_idx,
    num_tokens,
    tokens_per_expert,
    topk,
    swiglu_limit,
    num_comm_sms,
    macrobatch_size,
    minibatch_size,
    shared_grad_output=None,
    situ_beta1=0.0,
    situ_beta2=0.0,
):
    """C++ ``dispatch_mlp_swiglu_combine_bwd_bf16``: the fifteen outputs (d_hidden_* are None here)."""
    from cudnn.moe_dispatch_mlp_combine.cutedsl.persistent_bf16_bwd import backward_bf16

    workspace = _backward_workspace(
        x,
        x_ptrs,
        d_y_buffer,
        d_y_buffer_ptrs,
        d_x_routed_buffer,
        d_x_routed_buffer_ptrs,
        router_weight_buffer,
        router_weight_buffer_ptrs,
        d_router_weight_buffer,
        d_router_weight_buffer_ptrs,
        topk,
        schedule_peer_rank.numel(),
    )
    schedule = _schedule(schedule_peer_rank, schedule_peer_token_idx, num_tokens, tokens_per_expert)
    context = SimpleNamespace(
        x_routed=x_routed,
        gate_shared=gate_shared,
        gate_routed=gate_routed,
        up_shared=up_shared,
        up_routed=up_routed,
        hidden_shared=hidden_shared,
        hidden_routed=hidden_routed,
    )
    raw = backward_bf16(
        workspace,
        schedule,
        context,
        w_shared_gate,
        w_routed_gate,
        w_shared_up,
        w_routed_up,
        w_shared_down,
        w_routed_down,
        macrobatch_size=macrobatch_size,
        minibatch_size=minibatch_size,
        num_comm_sms=num_comm_sms,
        swiglu_limit=swiglu_limit,
        d_y_shared=shared_grad_output,
        situ_beta=_situ(situ_beta1, situ_beta2),
    )
    # raw13: d_x_shared, d_x_routed, d_gate_shared, d_gate_routed, d_up_shared, d_up_routed,
    # d_y_routed, then the six weight gradients shared/routed per matrix.
    return (*raw[:6], None, None, raw[6], *raw[7:])


def dispatch_mlp_swiglu_combine_bwd_mxfp8(
    d_y_buffer,
    d_y_buffer_ptrs,
    d_x_routed_buffer,
    d_x_routed_buffer_ptrs,
    router_weight_buffer,
    router_weight_buffer_ptrs,
    d_router_weight_buffer,
    d_router_weight_buffer_ptrs,
    w_shared_gate,
    w_routed_gate_T,
    w_routed_gate_T_sc,
    w_shared_up,
    w_routed_up_T,
    w_routed_up_T_sc,
    w_shared_down,
    w_routed_down_T,
    w_routed_down_T_sc,
    x_fp8_t_routed,
    x_sc_t_routed,
    gate_shared,
    gate_fp8_routed,
    gate_sc_routed,
    up_shared,
    up_fp8_routed,
    up_sc_routed,
    hidden_shared,
    hidden_fp8_t_routed,
    hidden_sc_t_routed,
    x,
    x_ptrs,
    w_routed_gate,
    w_routed_gate_sc,
    w_routed_up,
    w_routed_up_sc,
    schedule_peer_rank,
    schedule_peer_token_idx,
    num_tokens,
    tokens_per_expert,
    topk,
    swiglu_limit,
    num_comm_sms,
    macrobatch_size,
    minibatch_size,
    routed_weights_are_native_columnwise=False,
    shared_grad_output=None,
    situ_beta1=0.0,
    situ_beta2=0.0,
):
    """C++ ``dispatch_mlp_swiglu_combine_bwd_mxfp8``: the eighteen outputs, in its order."""
    from cudnn.moe_dispatch_mlp_combine.cutedsl.persistent_mxfp8_bwd import backward_mxfp8

    workspace = _backward_workspace(
        x,
        x_ptrs,
        d_y_buffer,
        d_y_buffer_ptrs,
        d_x_routed_buffer,
        d_x_routed_buffer_ptrs,
        router_weight_buffer,
        router_weight_buffer_ptrs,
        d_router_weight_buffer,
        d_router_weight_buffer_ptrs,
        topk,
        schedule_peer_rank.numel(),
    )
    schedule = _schedule(schedule_peer_rank, schedule_peer_token_idx, num_tokens, tokens_per_expert)
    context = dict(
        x_fp8_t_routed=x_fp8_t_routed,
        x_sc_t_routed=x_sc_t_routed,
        gate_shared=gate_shared,
        gate_fp8_routed=gate_fp8_routed,
        gate_sc_routed=gate_sc_routed,
        up_shared=up_shared,
        up_fp8_routed=up_fp8_routed,
        up_sc_routed=up_sc_routed,
        hidden_shared=hidden_shared,
        hidden_fp8_t_routed=hidden_fp8_t_routed,
        hidden_sc_t_routed=hidden_sc_t_routed,
    )
    return backward_mxfp8(
        workspace,
        schedule,
        context,
        w_shared_gate,
        (w_routed_gate, w_routed_gate_sc, w_routed_gate_T, w_routed_gate_T_sc),
        w_shared_up,
        (w_routed_up, w_routed_up_sc, w_routed_up_T, w_routed_up_T_sc),
        w_shared_down,
        (w_routed_down_T, w_routed_down_T_sc),
        macrobatch_size=macrobatch_size,
        minibatch_size=minibatch_size,
        num_comm_sms=num_comm_sms,
        swiglu_limit=swiglu_limit,
        native_columnwise=routed_weights_are_native_columnwise,
        d_y_shared=shared_grad_output,
        situ_beta=_situ(situ_beta1, situ_beta2),
    )


# --------------------------------------------------------------------------------------------------------------------
# Operators (the C++ port's op tests)
#
# The fused megakernel ops run their smallest case and their argument checks here; the test matrix runs
# through the functional layer (test_forward.py, test_backward.py), on the same kernels.
#
# Op-level tests: the C++ port's tests/test_ops.py (d59188a), on the CuTe DSL kernels.
#
# Every test body is the C++ one.  The fused megakernels are reached through the adapters above,
# which takes the C++ ops' arguments; the two recompute_forward_context op tests are not here: this
# package has no such op (``functional.recompute_forward_context`` leaves the routed recompute to the backward
# kernel, see test_recompute_forward_context.py).  test_schedule also checks the
# scheduler's fifth output, which the C++ scheduler does not have.
# The schedule tables are sized with ``utils.capacity_multiplier`` instead of the C++ tests' fixed 1.5: on one GPU a rank
# owns every expert, their padding to 256 rows overflows a 1.5 table and the scheduler traps (as upstream).


EXACT_TOLERANCE = (0.0, 0.0)


BF16_TOLERANCE = (0.5, 0.01)


MXFP8_TOLERANCE = (1.0, 0.1)


FORWARD_RESULT_NAMES = (
    "combine_buffer",
    "gate_shared",
    "up_shared",
    "hidden_shared",
    "y_shared",
)


BACKWARD_RESULT_NAMES = (
    "d_x",
    "d_router_weights",
    "d_w_routed_gate",
    "d_w_routed_up",
    "d_w_routed_down",
    "d_w_shared_gate",
    "d_w_shared_up",
    "d_w_shared_down",
)


MXFP8_RESULT_NAMES = (
    "normal",
    "normal_scales",
    "transposed",
    "transposed_scales",
)


def test_all_gather_top_experts(context: tuple[int, int, torch.device]) -> None:
    rank, world_size, device = context
    for shape in shapes(world_size):
        shape_name, num_experts, hidden_dim, _, topk, num_local_tokens = shape
        config = functional.CuteMoKConfig(schedule_capacity_multiplier=capacity_multiplier(world_size, num_local_tokens, topk, num_experts // world_size))
        workspace = get_workspace(
            config,
            dist.group.WORLD,
            device=device,
            num_local_tokens=num_local_tokens,
            hidden_size=hidden_dim,
            topk=topk,
        )
        topk_experts = generate_topk_experts(rank, device, num_experts, topk, num_local_tokens).to(torch.int32)

        all_gather_top_experts(
            topk_experts,
            workspace.all_gather_top_experts_buffer,
            workspace.all_gather_top_experts_buffer_multicast_ptr,
            rank,
            config.all_gather_top_experts_chunk_bytes,
        )
        barrier_all(workspace.barrier_buffer, workspace.barrier_buffer_ptrs, workspace.barrier_buffer_multicast_ptr, workspace.barrier_target)

        reference = run_all_gather_top_experts_reference(topk_experts)
        check_correctness(
            shape_name,
            reference,
            workspace.all_gather_top_experts_buffer,
            EXACT_TOLERANCE,
            print_stats=rank == 0,
        )

    num_experts = world_size
    hidden_dim = 256
    topk = 1
    num_local_tokens = 512
    config = functional.CuteMoKConfig(schedule_capacity_multiplier=capacity_multiplier(world_size, num_local_tokens, topk, num_experts // world_size))
    workspace = get_workspace(
        config,
        dist.group.WORLD,
        device=device,
        num_local_tokens=num_local_tokens,
        hidden_size=hidden_dim,
        topk=topk,
    )
    topk_experts = generate_topk_experts(rank, device, num_experts, topk, num_local_tokens).to(torch.int32)

    all_gather_top_experts(
        topk_experts,
        workspace.all_gather_top_experts_buffer,
        workspace.all_gather_top_experts_buffer_multicast_ptr,
        rank,
        16,
    )
    barrier_all(workspace.barrier_buffer, workspace.barrier_buffer_ptrs, workspace.barrier_buffer_multicast_ptr, workspace.barrier_target)
    reference = run_all_gather_top_experts_reference(topk_experts)
    check_correctness(
        "Minimum chunk bytes",
        reference,
        workspace.all_gather_top_experts_buffer,
        EXACT_TOLERANCE,
        print_stats=rank == 0,
    )

    valid_kwargs = {
        "top_experts": topk_experts,
        "all_gather_top_experts_buffer": workspace.all_gather_top_experts_buffer,
        "all_gather_top_experts_buffer_multicast_ptr": workspace.all_gather_top_experts_buffer_multicast_ptr,
        "rank": rank,
        "chunk_bytes": 16,
    }
    for failure_name, overrides, expected_exception in (
        ("top_experts dtype", {"top_experts": topk_experts.to(torch.int64)}, ValueError),
        (
            "all-gather buffer dtype",
            {"all_gather_top_experts_buffer": workspace.all_gather_top_experts_buffer.to(torch.int64)},
            TypeError,
        ),
        (
            "multicast pointer",
            {"all_gather_top_experts_buffer_multicast_ptr": 0},
            TypeError,
        ),
        ("rank", {"rank": world_size}, ValueError),
        ("chunk alignment", {"chunk_bytes": 15}, ValueError),
        ("chunk divisibility", {"chunk_bytes": 48}, ValueError),
    ):
        try:
            all_gather_top_experts(**(valid_kwargs | overrides))
        except expected_exception:
            pass
        else:
            raise AssertionError(f"{failure_name} should have failed")


def test_schedule(context: tuple[int, int, torch.device]) -> None:
    rank, world_size, device = context
    for shape in shapes(world_size):
        shape_name, num_experts, _, _, topk, num_local_tokens = shape
        assert num_experts % world_size == 0
        num_local_experts = num_experts // world_size
        multiplier = capacity_multiplier(world_size, num_local_tokens, topk, num_local_experts)
        schedule_capacity = num_local_tokens * topk * max(2, math.ceil(world_size * multiplier))
        topk_experts = generate_topk_experts(rank, device, num_experts, topk, num_local_tokens).to(torch.int32)
        route_counts = torch.arange(num_local_tokens, device=device) % (topk + 1)
        route_valid = torch.arange(topk, device=device).unsqueeze(0) < route_counts.unsqueeze(1)
        topk_experts.masked_fill_(~route_valid, -1)
        topk_all = run_all_gather_top_experts_reference(topk_experts)

        actual = schedule(topk_all, num_local_experts, schedule_capacity, rank)
        reference = run_schedule_reference(topk_all, num_local_experts, schedule_capacity, rank)
        valid_tokens = reference[0] >= 0
        # The port's fifth output, the unpadded rows per expert, is what the adapters above derive from the
        # first four for the BF16 forward adapter.
        assert len(actual) == 5
        assert torch.equal(actual[4], _valid_tokens_per_expert(actual[0], actual[2], actual[3])), shape_name
        for name, expected, result in (
            ("peer_rank", reference[0], actual[0]),
            ("peer_token_idx", reference[1][valid_tokens], actual[1][valid_tokens]),
            ("num_tokens", reference[2], actual[2]),
            ("tokens_per_expert", reference[3], actual[3]),
        ):
            check_correctness(
                f"{shape_name}/{name}",
                expected,
                result,
                EXACT_TOLERANCE,
                print_stats=rank == 0,
            )

    num_local_experts = 1
    num_local_tokens = 512
    topk = 1
    schedule_capacity = num_local_tokens * topk
    route_pattern = torch.arange(num_local_tokens, dtype=torch.int32, device=device).remainder(world_size)
    topk_all = route_pattern.view(1, num_local_tokens, topk).expand(world_size, -1, -1).contiguous()

    actual = schedule(topk_all, num_local_experts, schedule_capacity, rank)
    reference = run_schedule_reference(topk_all, num_local_experts, schedule_capacity, rank)
    valid_tokens = reference[0] >= 0
    for name, expected, result in (
        ("peer_rank", reference[0], actual[0]),
        ("peer_token_idx", reference[1][valid_tokens], actual[1][valid_tokens]),
        ("num_tokens", reference[2], actual[2]),
        ("tokens_per_expert", reference[3], actual[3]),
    ):
        check_correctness(
            f"Minimum schedule capacity/{name}",
            expected,
            result,
            EXACT_TOLERANCE,
            print_stats=rank == 0,
        )

    valid_kwargs = {
        "topk_all": topk_all,
        "num_local_experts": num_local_experts,
        "schedule_capacity": schedule_capacity,
        "rank": rank,
    }
    for failure_name, overrides, expected_exception in (
        ("topk_all rank", {"topk_all": topk_all.squeeze(-1)}, ValueError),
        ("local experts", {"num_local_experts": 0}, ValueError),
        ("schedule capacity zero", {"schedule_capacity": 0}, ValueError),
        ("schedule capacity alignment", {"schedule_capacity": 128}, ValueError),
        ("schedule capacity size", {"schedule_capacity": 256}, ValueError),
        ("rank", {"rank": world_size}, ValueError),
    ):
        try:
            schedule(**(valid_kwargs | overrides))
        except expected_exception:
            pass
        else:
            raise AssertionError(f"{failure_name} should have failed")


def test_mxfp8_quantize(context: tuple[int, int, torch.device]) -> None:
    rank, world_size, device = context
    for shape in shapes(world_size):
        shape_name, num_experts, hidden_dim, intermediate_dim, _, num_local_tokens = shape
        assert num_experts % world_size == 0
        num_local_experts = num_experts // world_size
        generator = torch.Generator(device=device).manual_seed(1234 + rank)
        input_shapes = (
            ("2D", (num_local_tokens, hidden_dim)),
            ("3D", (num_local_experts, intermediate_dim, hidden_dim)),
        )
        for input_name, input_shape in input_shapes:
            x_bf16 = torch.randn(
                input_shape,
                generator=generator,
                device=device,
                dtype=torch.bfloat16,
            )
            actual = mxfp8_quantize(x_bf16, True, True)
            reference = run_mxfp8_quantize_reference(x_bf16, True, True)
            for name, expected, result in zip(MXFP8_RESULT_NAMES, reference, actual, strict=True):
                assert expected is not None
                assert result is not None
                if expected.dtype == torch.float8_e4m3fn:
                    expected = expected.view(torch.uint8)
                    result = result.view(torch.uint8)
                check_correctness(
                    f"{shape_name}/{input_name}/{name}",
                    expected,
                    result,
                    EXACT_TOLERANCE,
                    print_stats=rank == 0,
                )

    generator = torch.Generator(device=device).manual_seed(5678 + rank)
    x_bf16 = torch.randn(128, 128, generator=generator, device=device, dtype=torch.bfloat16)
    for layout_name, return_normal, return_transposed in (
        ("Normal only", True, False),
        ("Transposed only", False, True),
    ):
        actual = mxfp8_quantize(x_bf16, return_normal, return_transposed)
        reference = run_mxfp8_quantize_reference(x_bf16, return_normal, return_transposed)
        for name, expected, result in zip(MXFP8_RESULT_NAMES, reference, actual, strict=True):
            if expected is None:
                assert result is None
                continue
            assert result is not None
            if expected.dtype == torch.float8_e4m3fn:
                expected = expected.view(torch.uint8)
                result = result.view(torch.uint8)
            check_correctness(
                f"{layout_name}/{name}",
                expected,
                result,
                EXACT_TOLERANCE,
                print_stats=rank == 0,
            )

    for failure_name, x, return_normal, return_transposed in (
        ("input rank", x_bf16.flatten(), True, True),
        (
            "row divisibility",
            torch.randn(127, 128, device=device, dtype=torch.bfloat16),
            True,
            True,
        ),
        (
            "column divisibility",
            torch.randn(128, 127, device=device, dtype=torch.bfloat16),
            True,
            True,
        ),
        ("missing layout", x_bf16, False, False),
    ):
        try:
            mxfp8_quantize(x, return_normal, return_transposed)
        except ValueError:
            pass
        else:
            raise AssertionError(f"{failure_name} should have failed")


def test_dispatch_mlp_swiglu_combine_fwd_mxfp8(
    context: tuple[int, int, torch.device],
) -> None:
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
        schedule_capacity_multiplier=capacity_multiplier(world_size, num_local_tokens, topk, num_experts // world_size),
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
        _,
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
    workspace.x_buffer.copy_(x)
    barrier_all(workspace.barrier_buffer, workspace.barrier_buffer_ptrs, workspace.barrier_buffer_multicast_ptr, workspace.barrier_target)
    valid_kwargs = {
        "x": workspace.x_buffer,
        "x_ptrs": workspace.x_buffer_ptrs,
        "combine_buffer": workspace.combine_buffer,
        "combine_buffer_ptrs": workspace.combine_buffer_ptrs,
        "w_shared_gate": w_shared_gate,
        "w_routed_gate": w_routed_gate_fp8,
        "w_routed_gate_sc": w_routed_gate_sc,
        "w_shared_up": w_shared_up,
        "w_routed_up": w_routed_up_fp8,
        "w_routed_up_sc": w_routed_up_sc,
        "w_shared_down": w_shared_down,
        "w_routed_down": w_routed_down_fp8,
        "w_routed_down_sc": w_routed_down_sc,
        "schedule_peer_rank": mok_schedule.peer_rank,
        "schedule_peer_token_idx": mok_schedule.peer_token_idx,
        "num_tokens": mok_schedule.num_tokens,
        "tokens_per_expert": mok_schedule.tokens_per_expert,
        "topk": topk,
        "swiglu_limit": None,
        "num_comm_sms": 2,
        "macrobatch_size": 256,
        "minibatch_size": 256,
    }

    outputs = dispatch_mlp_swiglu_combine_fwd_mxfp8(**valid_kwargs)
    barrier_all(workspace.barrier_buffer, workspace.barrier_buffer_ptrs, workspace.barrier_buffer_multicast_ptr, workspace.barrier_target)
    actual = (
        workspace.combine_buffer,
        outputs[2],
        outputs[5],
        outputs[8],
        outputs[11],
    )
    reference = run_forward_reference_bf16(
        x,
        topk_experts,
        w_shared_gate,
        w_shared_up,
        w_shared_down,
        w_routed_gate,
        w_routed_up,
        w_routed_down,
    )
    for name, expected, result in zip(FORWARD_RESULT_NAMES, reference, actual, strict=True):
        check_correctness(
            f"All combined minimums/{name}",
            expected,
            result,
            MXFP8_TOLERANCE,
            print_stats=rank == 0,
        )

    for failure_name, overrides, expected_exception in (
        ("input rank", {"x": workspace.x_buffer.unsqueeze(0)}, ValueError),
        ("top-k", {"topk": 0}, ValueError),
        ("communication SMs", {"num_comm_sms": 1}, ValueError),
        ("minibatch alignment", {"minibatch_size": 128}, ValueError),
        ("macrobatch multiple", {"macrobatch_size": 384}, ValueError),
        ("pointer EP size", {"x_ptrs": workspace.x_buffer_ptrs[:-1]}, ValueError),
    ):
        try:
            dispatch_mlp_swiglu_combine_fwd_mxfp8(**(valid_kwargs | overrides))
        except expected_exception:
            pass
        else:
            raise AssertionError(f"{failure_name} should have failed")


def test_dispatch_mlp_swiglu_combine_fwd_bf16(
    context: tuple[int, int, torch.device],
) -> None:
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
        schedule_capacity_multiplier=capacity_multiplier(world_size, num_local_tokens, topk, num_experts // world_size),
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
        _,
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
    workspace.x_buffer.copy_(x)
    barrier_all(workspace.barrier_buffer, workspace.barrier_buffer_ptrs, workspace.barrier_buffer_multicast_ptr, workspace.barrier_target)
    valid_kwargs = {
        "x": workspace.x_buffer,
        "x_ptrs": workspace.x_buffer_ptrs,
        "combine_buffer": workspace.combine_buffer,
        "combine_buffer_ptrs": workspace.combine_buffer_ptrs,
        "w_shared_gate": w_shared_gate,
        "w_routed_gate": w_routed_gate,
        "w_shared_up": w_shared_up,
        "w_routed_up": w_routed_up,
        "w_shared_down": w_shared_down,
        "w_routed_down": w_routed_down,
        "schedule_peer_rank": mok_schedule.peer_rank,
        "schedule_peer_token_idx": mok_schedule.peer_token_idx,
        "num_tokens": mok_schedule.num_tokens,
        "tokens_per_expert": mok_schedule.tokens_per_expert,
        "topk": topk,
        "swiglu_limit": None,
        "num_comm_sms": 2,
        "macrobatch_size": 256,
        "minibatch_size": 256,
    }

    outputs = dispatch_mlp_swiglu_combine_fwd_bf16(**valid_kwargs)
    barrier_all(workspace.barrier_buffer, workspace.barrier_buffer_ptrs, workspace.barrier_buffer_multicast_ptr, workspace.barrier_target)
    actual = (
        workspace.combine_buffer,
        outputs[1],
        outputs[3],
        outputs[5],
        outputs[7],
    )
    reference = run_forward_reference_bf16(
        x,
        topk_experts,
        w_shared_gate,
        w_shared_up,
        w_shared_down,
        w_routed_gate,
        w_routed_up,
        w_routed_down,
    )
    for name, expected, result in zip(FORWARD_RESULT_NAMES, reference, actual, strict=True):
        check_correctness(
            f"All combined minimums/{name}",
            expected,
            result,
            BF16_TOLERANCE,
            print_stats=rank == 0,
        )

    for failure_name, overrides, expected_exception in (
        ("input rank", {"x": workspace.x_buffer.unsqueeze(0)}, ValueError),
        ("top-k", {"topk": 0}, ValueError),
        ("communication SMs", {"num_comm_sms": 1}, ValueError),
        ("minibatch alignment", {"minibatch_size": 128}, ValueError),
        ("macrobatch multiple", {"macrobatch_size": 384}, ValueError),
        ("pointer EP size", {"x_ptrs": workspace.x_buffer_ptrs[:-1]}, ValueError),
    ):
        try:
            dispatch_mlp_swiglu_combine_fwd_bf16(**(valid_kwargs | overrides))
        except expected_exception:
            pass
        else:
            raise AssertionError(f"{failure_name} should have failed")


def test_dispatch_mlp_swiglu_combine_bwd_mxfp8(
    context: tuple[int, int, torch.device],
) -> None:
    """Against the BF16 reference within ``MXFP8_TOLERANCE``."""
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
        schedule_capacity_multiplier=capacity_multiplier(world_size, num_local_tokens, topk, num_experts // world_size),
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
        d_output,
    ) = inputs
    mok_schedule = functional.build_schedule(
        workspace,
        config,
        topk_experts,
        num_local_experts=num_local_experts,
    )
    (
        w_routed_gate_fp8,
        w_routed_gate_sc,
        w_routed_gate_t_fp8,
        w_routed_gate_t_sc,
    ) = mxfp8_quantize(w_routed_gate, True, True)
    (
        w_routed_up_fp8,
        w_routed_up_sc,
        w_routed_up_t_fp8,
        w_routed_up_t_sc,
    ) = mxfp8_quantize(w_routed_up, True, True)
    (
        w_routed_down_fp8,
        w_routed_down_sc,
        w_routed_down_t_fp8,
        w_routed_down_t_sc,
    ) = mxfp8_quantize(w_routed_down, True, True)
    workspace.x_buffer.copy_(x)
    barrier_all(workspace.barrier_buffer, workspace.barrier_buffer_ptrs, workspace.barrier_buffer_multicast_ptr, workspace.barrier_target)
    (
        x_fp8_t_routed,
        x_sc_t_routed,
        gate_shared,
        gate_fp8_routed,
        gate_sc_routed,
        up_shared,
        up_fp8_routed,
        up_sc_routed,
        hidden_shared,
        hidden_fp8_t_routed,
        hidden_sc_t_routed,
        _,
        _,
    ) = dispatch_mlp_swiglu_combine_fwd_mxfp8(
        workspace.x_buffer,
        workspace.x_buffer_ptrs,
        workspace.combine_buffer,
        workspace.combine_buffer_ptrs,
        w_shared_gate,
        w_routed_gate_fp8,
        w_routed_gate_sc,
        w_shared_up,
        w_routed_up_fp8,
        w_routed_up_sc,
        w_shared_down,
        w_routed_down_fp8,
        w_routed_down_sc,
        mok_schedule.peer_rank,
        mok_schedule.peer_token_idx,
        mok_schedule.num_tokens,
        mok_schedule.tokens_per_expert,
        topk,
        None,
        2,
        256,
        256,
    )
    barrier_all(workspace.barrier_buffer, workspace.barrier_buffer_ptrs, workspace.barrier_buffer_multicast_ptr, workspace.barrier_target)
    workspace.d_y_buffer.copy_(d_output)
    workspace.x_buffer.copy_(x)
    workspace.router_weight_buffer.copy_(router_weights)
    barrier_all(workspace.barrier_buffer, workspace.barrier_buffer_ptrs, workspace.barrier_buffer_multicast_ptr, workspace.barrier_target)
    valid_kwargs = {
        "d_y_buffer": workspace.d_y_buffer,
        "d_y_buffer_ptrs": workspace.d_y_buffer_ptrs,
        "d_x_routed_buffer": workspace.d_x_routed_buffer,
        "d_x_routed_buffer_ptrs": workspace.d_x_routed_buffer_ptrs,
        "router_weight_buffer": workspace.router_weight_buffer,
        "router_weight_buffer_ptrs": workspace.router_weight_buffer_ptrs,
        "d_router_weight_buffer": workspace.d_router_weight_buffer,
        "d_router_weight_buffer_ptrs": workspace.d_router_weight_buffer_ptrs,
        "w_shared_gate": w_shared_gate,
        "w_routed_gate_T": w_routed_gate_t_fp8,
        "w_routed_gate_T_sc": w_routed_gate_t_sc,
        "w_shared_up": w_shared_up,
        "w_routed_up_T": w_routed_up_t_fp8,
        "w_routed_up_T_sc": w_routed_up_t_sc,
        "w_shared_down": w_shared_down,
        "w_routed_down_T": w_routed_down_t_fp8,
        "w_routed_down_T_sc": w_routed_down_t_sc,
        "x_fp8_t_routed": x_fp8_t_routed,
        "x_sc_t_routed": x_sc_t_routed,
        "gate_shared": gate_shared,
        "gate_fp8_routed": gate_fp8_routed,
        "gate_sc_routed": gate_sc_routed,
        "up_shared": up_shared,
        "up_fp8_routed": up_fp8_routed,
        "up_sc_routed": up_sc_routed,
        "hidden_shared": hidden_shared,
        "hidden_fp8_t_routed": hidden_fp8_t_routed,
        "hidden_sc_t_routed": hidden_sc_t_routed,
        "x": workspace.x_buffer,
        "x_ptrs": workspace.x_buffer_ptrs,
        "w_routed_gate": w_routed_gate_fp8,
        "w_routed_gate_sc": w_routed_gate_sc,
        "w_routed_up": w_routed_up_fp8,
        "w_routed_up_sc": w_routed_up_sc,
        "schedule_peer_rank": mok_schedule.peer_rank,
        "schedule_peer_token_idx": mok_schedule.peer_token_idx,
        "num_tokens": mok_schedule.num_tokens,
        "tokens_per_expert": mok_schedule.tokens_per_expert,
        "topk": topk,
        "swiglu_limit": None,
        "num_comm_sms": 2,
        "macrobatch_size": 256,
        "minibatch_size": 256,
    }

    outputs = dispatch_mlp_swiglu_combine_bwd_mxfp8(**valid_kwargs)
    barrier_all(workspace.barrier_buffer, workspace.barrier_buffer_ptrs, workspace.barrier_buffer_multicast_ptr, workspace.barrier_target)
    actual = (
        run_bwd_epilogue_reference(outputs[0], workspace.d_x_routed_buffer),
        workspace.d_router_weight_buffer.clone(),
        outputs[13],
        outputs[15],
        outputs[17],
        outputs[12],
        outputs[14],
        outputs[16],
    )
    reference = run_reference_bf16(*inputs)[1:]
    for name, expected, result in zip(BACKWARD_RESULT_NAMES, reference, actual, strict=True):
        check_correctness(
            f"All combined minimums/{name}",
            expected,
            result,
            MXFP8_TOLERANCE,
            print_stats=rank == 0,
        )

    for failure_name, overrides, expected_exception in (
        ("input rank", {"x": workspace.x_buffer.unsqueeze(0)}, ValueError),
        ("top-k", {"topk": 0}, ValueError),
        ("communication SMs", {"num_comm_sms": 1}, ValueError),
        ("minibatch alignment", {"minibatch_size": 128}, ValueError),
        ("macrobatch multiple", {"macrobatch_size": 384}, ValueError),
        ("pointer EP size", {"x_ptrs": workspace.x_buffer_ptrs[:-1]}, ValueError),
        (
            "gradient output shape",
            {"d_y_buffer": workspace.d_y_buffer[:, :-1]},
            ValueError,
        ),
    ):
        try:
            dispatch_mlp_swiglu_combine_bwd_mxfp8(**(valid_kwargs | overrides))
        except expected_exception:
            pass
        else:
            raise AssertionError(f"{failure_name} should have failed")


def test_dispatch_mlp_swiglu_combine_bwd_bf16(
    context: tuple[int, int, torch.device],
) -> None:
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
        schedule_capacity_multiplier=capacity_multiplier(world_size, num_local_tokens, topk, num_experts // world_size),
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
        d_output,
    ) = inputs
    mok_schedule = functional.build_schedule(
        workspace,
        config,
        topk_experts,
        num_local_experts=num_local_experts,
    )
    workspace.x_buffer.copy_(x)
    barrier_all(workspace.barrier_buffer, workspace.barrier_buffer_ptrs, workspace.barrier_buffer_multicast_ptr, workspace.barrier_target)
    (
        x_routed,
        gate_shared,
        gate_routed,
        up_shared,
        up_routed,
        hidden_shared,
        hidden_routed,
        _,
        _,
    ) = dispatch_mlp_swiglu_combine_fwd_bf16(
        workspace.x_buffer,
        workspace.x_buffer_ptrs,
        workspace.combine_buffer,
        workspace.combine_buffer_ptrs,
        w_shared_gate,
        w_routed_gate,
        w_shared_up,
        w_routed_up,
        w_shared_down,
        w_routed_down,
        mok_schedule.peer_rank,
        mok_schedule.peer_token_idx,
        mok_schedule.num_tokens,
        mok_schedule.tokens_per_expert,
        topk,
        None,
        2,
        256,
        256,
    )
    barrier_all(workspace.barrier_buffer, workspace.barrier_buffer_ptrs, workspace.barrier_buffer_multicast_ptr, workspace.barrier_target)
    workspace.d_y_buffer.copy_(d_output)
    workspace.x_buffer.copy_(x)
    workspace.router_weight_buffer.copy_(router_weights)
    barrier_all(workspace.barrier_buffer, workspace.barrier_buffer_ptrs, workspace.barrier_buffer_multicast_ptr, workspace.barrier_target)
    valid_kwargs = {
        "d_y_buffer": workspace.d_y_buffer,
        "d_y_buffer_ptrs": workspace.d_y_buffer_ptrs,
        "d_x_routed_buffer": workspace.d_x_routed_buffer,
        "d_x_routed_buffer_ptrs": workspace.d_x_routed_buffer_ptrs,
        "router_weight_buffer": workspace.router_weight_buffer,
        "router_weight_buffer_ptrs": workspace.router_weight_buffer_ptrs,
        "d_router_weight_buffer": workspace.d_router_weight_buffer,
        "d_router_weight_buffer_ptrs": workspace.d_router_weight_buffer_ptrs,
        "w_shared_gate": w_shared_gate,
        "w_routed_gate": w_routed_gate,
        "w_shared_up": w_shared_up,
        "w_routed_up": w_routed_up,
        "w_shared_down": w_shared_down,
        "w_routed_down": w_routed_down,
        "x_routed": x_routed,
        "gate_shared": gate_shared,
        "gate_routed": gate_routed,
        "up_shared": up_shared,
        "up_routed": up_routed,
        "hidden_shared": hidden_shared,
        "hidden_routed": hidden_routed,
        "x": workspace.x_buffer,
        "x_ptrs": workspace.x_buffer_ptrs,
        "schedule_peer_rank": mok_schedule.peer_rank,
        "schedule_peer_token_idx": mok_schedule.peer_token_idx,
        "num_tokens": mok_schedule.num_tokens,
        "tokens_per_expert": mok_schedule.tokens_per_expert,
        "topk": topk,
        "swiglu_limit": None,
        "num_comm_sms": 2,
        "macrobatch_size": 256,
        "minibatch_size": 256,
    }

    outputs = dispatch_mlp_swiglu_combine_bwd_bf16(**valid_kwargs)
    barrier_all(workspace.barrier_buffer, workspace.barrier_buffer_ptrs, workspace.barrier_buffer_multicast_ptr, workspace.barrier_target)
    actual = (
        run_bwd_epilogue_reference(outputs[0], workspace.d_x_routed_buffer),
        workspace.d_router_weight_buffer.clone(),
        outputs[10],
        outputs[12],
        outputs[14],
        outputs[9],
        outputs[11],
        outputs[13],
    )
    reference = run_reference_bf16(*inputs)[1:]
    for name, expected, result in zip(BACKWARD_RESULT_NAMES, reference, actual, strict=True):
        check_correctness(
            f"All combined minimums/{name}",
            expected,
            result,
            BF16_TOLERANCE,
            print_stats=rank == 0,
        )

    for failure_name, overrides, expected_exception in (
        ("input rank", {"x": workspace.x_buffer.unsqueeze(0)}, ValueError),
        ("top-k", {"topk": 0}, ValueError),
        ("communication SMs", {"num_comm_sms": 1}, ValueError),
        ("minibatch alignment", {"minibatch_size": 128}, ValueError),
        ("macrobatch multiple", {"macrobatch_size": 384}, ValueError),
        ("pointer EP size", {"x_ptrs": workspace.x_buffer_ptrs[:-1]}, ValueError),
        (
            "gradient output shape",
            {"d_y_buffer": workspace.d_y_buffer[:, :-1]},
            ValueError,
        ),
    ):
        try:
            dispatch_mlp_swiglu_combine_bwd_bf16(**(valid_kwargs | overrides))
        except expected_exception:
            pass
        else:
            raise AssertionError(f"{failure_name} should have failed")


def test_fwd_epilogue(context: tuple[int, int, torch.device]) -> None:
    rank, world_size, device = context
    for shape in shapes(world_size):
        shape_name, _, hidden_dim, _, topk, num_local_tokens = shape
        generator = torch.Generator(device=device).manual_seed(1234 + rank)
        y_shared = torch.randn(
            num_local_tokens,
            hidden_dim,
            generator=generator,
            device=device,
            dtype=torch.bfloat16,
        )
        combine_buffer = torch.randn(
            num_local_tokens * topk,
            hidden_dim,
            generator=generator,
            device=device,
            dtype=torch.bfloat16,
        )
        topk_weights = torch.softmax(
            torch.randn(
                num_local_tokens,
                topk,
                generator=generator,
                device=device,
            ),
            dim=-1,
        )
        route_counts = torch.arange(num_local_tokens, device=device) % (topk + 1)
        route_valid = torch.arange(topk, device=device).unsqueeze(0) < route_counts.unsqueeze(1)
        top_experts = torch.zeros(num_local_tokens, topk, dtype=torch.int32, device=device).masked_fill(~route_valid, -1)
        combine_buffer.view(num_local_tokens, topk, hidden_dim).masked_fill_(~route_valid.unsqueeze(2), float("nan"))
        topk_weights.masked_fill_(~route_valid, 0.0)
        actual = fwd_epilogue(y_shared, combine_buffer, topk_weights, top_experts)
        reference = run_fwd_epilogue_reference(y_shared, combine_buffer, topk_weights, top_experts)
        check_correctness(
            shape_name,
            reference,
            actual,
            BF16_TOLERANCE,
            print_stats=rank == 0,
        )

    num_local_tokens = 512
    hidden_dim = 256
    shared_memory_limit = torch.cuda.get_device_properties(device).shared_memory_per_block_optin
    topk = (shared_memory_limit - 5120) // 4112
    generator = torch.Generator(device=device).manual_seed(5678 + rank)
    y_shared = torch.randn(num_local_tokens, hidden_dim, generator=generator, device=device, dtype=torch.bfloat16)
    combine_buffer = torch.randn(num_local_tokens * topk, hidden_dim, generator=generator, device=device, dtype=torch.bfloat16)
    topk_weights = torch.softmax(
        torch.randn(num_local_tokens, topk, generator=generator, device=device),
        dim=-1,
    )
    top_experts = torch.zeros(num_local_tokens, topk, dtype=torch.int32, device=device)

    actual = fwd_epilogue(y_shared, combine_buffer, topk_weights, top_experts)
    reference = run_fwd_epilogue_reference(y_shared, combine_buffer, topk_weights, top_experts)
    check_correctness(
        "Maximum executable top-k",
        reference,
        actual,
        BF16_TOLERANCE,
        print_stats=rank == 0,
    )

    valid_kwargs = {
        "y_shared": y_shared,
        "combine_buffer": combine_buffer,
        "topk_weights": topk_weights,
        "top_experts": top_experts,
    }
    for failure_name, overrides, expected_exception in (
        ("shared output rank", {"y_shared": y_shared.unsqueeze(0)}, ValueError),
        ("local token minimum", {"y_shared": y_shared[:256]}, ValueError),
        ("hidden alignment", {"y_shared": y_shared[:, :128]}, ValueError),
        (
            "top-k minimum",
            {
                "topk_weights": topk_weights[:, :0],
                "combine_buffer": combine_buffer[:0],
            },
            ValueError,
        ),
        (
            "top-k maximum",
            {"topk_weights": topk_weights.new_empty(num_local_tokens, 256)},
            ValueError,
        ),
        (
            "combine shape",
            {"combine_buffer": combine_buffer[:-1]},
            ValueError,
        ),
    ):
        try:
            fwd_epilogue(**(valid_kwargs | overrides))
        except expected_exception:
            pass
        else:
            raise AssertionError(f"{failure_name} should have failed")


def test_bwd_epilogue(context: tuple[int, int, torch.device]) -> None:
    rank, world_size, device = context
    for shape in shapes(world_size):
        shape_name, _, hidden_dim, _, topk, num_local_tokens = shape
        generator = torch.Generator(device=device).manual_seed(1234 + rank)
        d_x_shared = torch.randn(
            num_local_tokens,
            hidden_dim,
            generator=generator,
            device=device,
            dtype=torch.bfloat16,
        )
        d_x_routed_buffer = torch.randn(
            num_local_tokens * topk,
            hidden_dim,
            generator=generator,
            device=device,
            dtype=torch.bfloat16,
        )
        route_counts = torch.arange(num_local_tokens, device=device) % (topk + 1)
        route_valid = torch.arange(topk, device=device).unsqueeze(0) < route_counts.unsqueeze(1)
        top_experts = torch.zeros(num_local_tokens, topk, dtype=torch.int32, device=device).masked_fill(~route_valid, -1)
        d_x_routed_buffer.view(num_local_tokens, topk, hidden_dim).masked_fill_(~route_valid.unsqueeze(2), float("nan"))
        actual = bwd_epilogue(d_x_shared, d_x_routed_buffer, top_experts)
        reference = run_bwd_epilogue_reference(d_x_shared, d_x_routed_buffer, top_experts)
        check_correctness(
            shape_name,
            reference,
            actual,
            BF16_TOLERANCE,
            print_stats=rank == 0,
        )

    num_local_tokens = 512
    hidden_dim = 256
    topk = 55
    generator = torch.Generator(device=device).manual_seed(5678 + rank)
    d_x_shared = torch.randn(num_local_tokens, hidden_dim, generator=generator, device=device, dtype=torch.bfloat16)
    d_x_routed_buffer = torch.randn(num_local_tokens * topk, hidden_dim, generator=generator, device=device, dtype=torch.bfloat16)
    top_experts = torch.zeros(num_local_tokens, topk, dtype=torch.int32, device=device)

    actual = bwd_epilogue(d_x_shared, d_x_routed_buffer, top_experts)
    reference = run_bwd_epilogue_reference(d_x_shared, d_x_routed_buffer, top_experts)
    check_correctness(
        "Large top-k",
        reference,
        actual,
        BF16_TOLERANCE,
        print_stats=rank == 0,
    )

    valid_kwargs = {
        "d_x_shared": d_x_shared,
        "d_x_routed_buffer": d_x_routed_buffer,
        "top_experts": top_experts,
    }
    for failure_name, overrides, expected_exception in (
        (
            "shared gradient rank",
            {"d_x_shared": d_x_shared.unsqueeze(0)},
            ValueError,
        ),
        (
            "local token minimum",
            {"d_x_shared": d_x_shared[:256]},
            ValueError,
        ),
        (
            "hidden alignment",
            {"d_x_shared": d_x_shared[:, :128]},
            ValueError,
        ),
        (
            "routed gradient rank",
            {"d_x_routed_buffer": d_x_routed_buffer.flatten()},
            ValueError,
        ),
        (
            "top-k minimum",
            {"d_x_routed_buffer": d_x_routed_buffer[:0]},
            ValueError,
        ),
        (
            "routed gradient shape",
            {"d_x_routed_buffer": d_x_routed_buffer[:-1]},
            ValueError,
        ),
    ):
        try:
            bwd_epilogue(**(valid_kwargs | overrides))
        except expected_exception:
            pass
        else:
            raise AssertionError(f"{failure_name} should have failed")


def test_copy_routed_rows(context) -> None:
    """The routed-row staging of the functional layer (C++ port's test_copy_routed_rows; the CuTe DSL op
    raises ValueError where the C++ entry point raises RuntimeError)."""
    rank, _, device = context
    generator = torch.Generator(device=device).manual_seed(4321 + rank)
    for num_local_tokens, hidden_dim, topk in ((512, 2048, 8), (768, 2304, 40)):
        top_experts = torch.randint(0, 64, (num_local_tokens, topk), generator=generator, device=device, dtype=torch.int32)
        # Route counts 0..topk+1 per row: rows without any route keep their stale buffer contents
        route_counts = torch.arange(num_local_tokens, device=device) % (topk + 2)
        top_experts.masked_fill_(torch.arange(topk, device=device).unsqueeze(0) >= route_counts.unsqueeze(1), -1)
        routed = (top_experts >= 0).any(dim=1)
        x = torch.randn(num_local_tokens, hidden_dim, generator=generator, device=device, dtype=torch.bfloat16)
        grad_output = torch.randn_like(x)
        router_weights = torch.rand(num_local_tokens, topk, generator=generator, device=device)
        x_buffer = torch.full_like(x, 3.0)
        d_y_buffer = torch.full_like(x, 5.0)
        router_weight_buffer = torch.full_like(router_weights, -1.0)

        ops.copy_routed_rows(top_experts, [x], [x_buffer], router_weights, router_weight_buffer)
        assert torch.equal(x_buffer[routed], x[routed])
        assert bool((x_buffer[~routed] == 3.0).all())
        assert torch.equal(router_weight_buffer, router_weights)

        x_buffer.fill_(3.0)
        router_weight_buffer.fill_(-1.0)
        ops.copy_routed_rows(top_experts, [grad_output, x], [d_y_buffer, x_buffer], router_weights, router_weight_buffer)
        assert torch.equal(d_y_buffer[routed], grad_output[routed])
        assert bool((d_y_buffer[~routed] == 5.0).all())
        assert torch.equal(x_buffer[routed], x[routed])
        assert bool((x_buffer[~routed] == 3.0).all())
        assert torch.equal(router_weight_buffer, router_weights)

    misaligned = torch.empty(x.numel() + 1, device=device, dtype=torch.bfloat16)[1:].view_as(x)
    for failure_name, sources, destinations, weights in (
        ("routes dtype", [x], [x_buffer], (router_weights, router_weight_buffer)),
        ("buffer shape", [x], [x_buffer[:-1]], (router_weights, router_weight_buffer)),
        ("buffer dtype", [x], [x_buffer.float()], (router_weights, router_weight_buffer)),
        ("misaligned source", [misaligned], [x_buffer], (router_weights, router_weight_buffer)),
        ("weights shape", [x], [x_buffer], (router_weights[:, :1].contiguous(), router_weight_buffer)),
    ):
        routes = top_experts.long() if failure_name == "routes dtype" else top_experts
        with pytest.raises((RuntimeError, ValueError)):
            ops.copy_routed_rows(routes, sources, destinations, *weights)


# --------------------------------------------------------------------------------------------------------------------
# Schedule through the functional layer


def test_build_schedule(context: tuple[int, int, torch.device]) -> None:
    rank, world_size, device = context
    for shape, params in itertools.product(shapes(world_size), mok_params()):
        shape_name, num_experts, hidden_dim, _, topk, num_local_tokens = shape
        params_name, fwd_num_comm_sms, bwd_num_comm_sms, minibatch_size, macrobatch_size = params
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
        topk_experts = generate_topk_experts(rank, device, num_experts, topk, num_local_tokens)

        actual = functional.build_schedule(
            workspace,
            config,
            topk_experts,
            num_local_experts=num_local_experts,
        )
        topk_all = run_all_gather_top_experts_reference(topk_experts.to(torch.int32))
        reference = run_schedule_reference(
            topk_all,
            num_local_experts,
            workspace.schedule_capacity,
            rank,
        )
        valid_tokens = reference[0] >= 0
        for name, expected, result in (
            ("peer_rank", reference[0], actual.peer_rank),
            ("peer_token_idx", reference[1][valid_tokens], actual.peer_token_idx[valid_tokens]),
            ("num_tokens", reference[2], actual.num_tokens),
            ("tokens_per_expert", reference[3], actual.tokens_per_expert),
        ):
            check_correctness(
                f"{shape_name}/{params_name}/{name}",
                expected,
                result,
                EXACT_TOLERANCE,
                print_stats=rank == 0,
            )

    num_experts = world_size
    num_local_experts = 1
    hidden_dim = 256
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
    topk_experts = generate_topk_experts(rank, device, num_experts, topk, num_local_tokens)

    actual = functional.build_schedule(
        workspace,
        config,
        topk_experts,
        num_local_experts=num_local_experts,
    )
    topk_all = run_all_gather_top_experts_reference(topk_experts.to(torch.int32))
    reference = run_schedule_reference(
        topk_all,
        num_local_experts,
        workspace.schedule_capacity,
        rank,
    )
    valid_tokens = reference[0] >= 0
    for name, expected, result in (
        ("peer_rank", reference[0], actual.peer_rank),
        ("peer_token_idx", reference[1][valid_tokens], actual.peer_token_idx[valid_tokens]),
        ("num_tokens", reference[2], actual.num_tokens),
        ("tokens_per_expert", reference[3], actual.tokens_per_expert),
    ):
        check_correctness(
            f"All combined minimums/{name}",
            expected,
            result,
            EXACT_TOLERANCE,
            print_stats=rank == 0,
        )

    valid_kwargs = {
        "workspace": workspace,
        "config": config,
        "top_experts": topk_experts,
        "num_local_experts": num_local_experts,
    }
    for failure_name, overrides, expected_exception in (
        ("workspace type", {"workspace": object()}, TypeError),
        ("config type", {"config": object()}, TypeError),
        (
            "forward communication SMs",
            {
                "config": functional.CuteMoKConfig(
                    fwd_num_comm_sms=1,
                    bwd_num_comm_sms=2,
                    minibatch_size=256,
                    macrobatch_size=256,
                    all_gather_top_experts_chunk_bytes=16,
                )
            },
            ValueError,
        ),
        (
            "backward communication SMs",
            {
                "config": functional.CuteMoKConfig(
                    fwd_num_comm_sms=2,
                    bwd_num_comm_sms=1,
                    minibatch_size=256,
                    macrobatch_size=256,
                    all_gather_top_experts_chunk_bytes=16,
                )
            },
            ValueError,
        ),
        (
            "minibatch alignment",
            {
                "config": functional.CuteMoKConfig(
                    fwd_num_comm_sms=2,
                    bwd_num_comm_sms=2,
                    minibatch_size=128,
                    macrobatch_size=256,
                    all_gather_top_experts_chunk_bytes=16,
                )
            },
            ValueError,
        ),
        (
            "macrobatch multiple",
            {
                "config": functional.CuteMoKConfig(
                    fwd_num_comm_sms=2,
                    bwd_num_comm_sms=2,
                    minibatch_size=256,
                    macrobatch_size=384,
                    all_gather_top_experts_chunk_bytes=16,
                )
            },
            ValueError,
        ),
        (
            "all-gather chunk alignment",
            {
                "config": functional.CuteMoKConfig(
                    fwd_num_comm_sms=2,
                    bwd_num_comm_sms=2,
                    minibatch_size=256,
                    macrobatch_size=256,
                    all_gather_top_experts_chunk_bytes=15,
                )
            },
            ValueError,
        ),
        (
            # int32 and int64 are both accepted now -- MCore's routing map is int32 -- so the
            # rejected dtype has to be one that is neither.
            "top experts dtype",
            {"top_experts": topk_experts.to(torch.int16)},
            TypeError,
        ),
        (
            "top experts shape",
            {"top_experts": topk_experts[:, :0]},
            ValueError,
        ),
        ("local experts", {"num_local_experts": 0}, ValueError),
    ):
        try:
            functional.build_schedule(**(valid_kwargs | overrides))
        except expected_exception:
            pass
        else:
            raise AssertionError(f"{failure_name} should have failed")


# --------------------------------------------------------------------------------------------------------------------
# Scheduler kernel
#
# CuTe DSL local scheduler against the PyTorch reference, entry for entry.
#
# ``utils.run_schedule_reference`` is the reference: the same gathered top-k must give the same padded
# expert counts, total, peer ranks (every capacity entry, -1 on padding rows and the tail) and, for
# the routed entries, peer token indices in the thread-major order.  Geometries cover route
# counts that are not a multiple of the 1024-thread stride, more routes per CUDA thread than one
# register batch of the schedule kernel holds, local expert counts that are not a multiple of the
# kernel's expert group and above 256, EP 1 / 4 / 8 / 16 / 64, first and last rank, skewed and empty
# experts, top-k 1, a single token, the capacity exactly at the padded total, and repeated calls whose
# outputs reuse the freed allocations of the previous call.
#
# One GPU, no distributed setup: ``python -m pytest moe_dispatch_mlp_combine/cutedsl/test_ops.py``.


def _blackwell() -> bool:
    return torch.cuda.is_available() and torch.cuda.get_device_capability() in SUPPORTED_CAPABILITIES


def _uniform(world, tokens, experts, topk, seed):
    generator = torch.Generator(device="cuda").manual_seed(seed)
    scores = torch.rand(world, tokens, experts, device="cuda", generator=generator)
    return scores.topk(topk, dim=-1).indices.to(torch.int32).contiguous()


def _reference(topk_all, num_local_experts, capacity, rank):
    """The PyTorch model of the schedule, which has no geometry restrictions of its own."""
    from .utils import run_schedule_reference

    return run_schedule_reference(topk_all, num_local_experts, capacity, rank)


@unittest.skipUnless(_blackwell(), "the CuTe DSL scheduler is SM100/SM103 only")
class SchedulerMatchesReference(unittest.TestCase):

    @staticmethod
    def _capacity(topk_all, num_local_experts, capacity):
        # Default: enough for every route plus a full padding tile per local expert (the scheduler traps otherwise).
        world, tokens, topk = topk_all.shape
        return world * tokens * topk + num_local_experts * 256 if capacity is None else capacity

    def _case(self, topk_all: torch.Tensor, num_local_experts: int, rank: int, capacity: int | None = None) -> None:
        from cudnn.moe_dispatch_mlp_combine.cutedsl.scheduler import scheduler

        capacity = self._capacity(topk_all, num_local_experts, capacity)
        self._compare(topk_all, num_local_experts, rank, capacity, scheduler(topk_all, num_local_experts, capacity, rank))

    def _compare(self, topk_all, num_local_experts, rank, capacity, actual) -> None:
        expected = _reference(topk_all, num_local_experts, capacity, rank)
        torch.cuda.synchronize()
        routed = expected[0] >= 0
        # The fifth output: the unpadded rows of each local expert, i.e. the routed rows (peer rank
        # >= 0) of its padded segment in the CUDA schedule.
        device = topk_all.device
        expert_of_row = torch.repeat_interleave(torch.arange(num_local_experts, device=device), expected[3].to(torch.int64))
        valid_rows = torch.zeros(num_local_experts, dtype=torch.int32, device=device).index_add_(0, expert_of_row, routed[: int(expected[2])].to(torch.int32))
        self.assertEqual(tuple(actual[0].shape), (capacity,))
        for name, want, got in (
            ("peer_rank", expected[0], actual[0]),  # every capacity entry: routed, padding (-1), tail (-1)
            ("peer_token_idx", expected[1][routed], actual[1][routed]),
            ("num_tokens", expected[2], actual[2]),
            ("tokens_per_expert", expected[3], actual[3]),
            ("valid_tokens_per_expert", valid_rows, actual[4]),
        ):
            differing = (want != got).nonzero()
            if differing.numel():
                first = int(differing[0, 0])
                self.fail(f"{name}: {differing.shape[0]}/{want.numel()} entries differ; first at {first}: " f"CUDA {int(want[first])} vs DSL {int(got[first])}")

    def test_uniform(self):
        """Uniform random routing: Qwen / DeepSeek-like expert counts and others (9 and 20 local experts), EP 1 / 4 / 8 / 16 / 64;
        (4, 65536, 256, 10) has 640 routes per CUDA thread, more than one register batch of the schedule kernel."""
        for world, tokens, experts, topk in [
            (8, 8192, 512, 10),
            (8, 3072, 384, 6),
            (8, 1024, 64, 8),
            (4, 4352, 36, 3),
            (1, 768, 20, 2),
            (8, 20480, 384, 6),
            (16, 1024, 256, 6),
            (64, 512, 128, 4),
            (4, 65536, 256, 10),
        ]:
            topk_all = _uniform(world, tokens, experts, topk, seed=tokens)
            for rank in sorted({0, world - 1}):
                with self.subTest(world=world, tokens=tokens, experts=experts, topk=topk, rank=rank):
                    self._case(topk_all, experts // world, rank)

    def test_more_than_256_local_experts(self):
        """E per rank above 256 (the mega kernels' grouped decode): EP1 with 512 local experts, EP4 with 288, EP8 with 288 (the
        multi-node shape on one GPU), and EP1 with 1024 (more local experts than the schedule kernel's block has threads, so its
        block scan over the padded counts runs a second round).  Each needs more than one pass of the schedule kernel's expert
        tables, which are refilled per group."""
        for world, tokens, experts, topk in [(1, 2048, 512, 8), (4, 1024, 1152, 6), (8, 512, 2304, 4), (1, 1024, 1024, 4)]:
            topk_all = _uniform(world, tokens, experts, topk, seed=experts)
            with self.subTest(world=world, experts=experts):
                self._case(topk_all, experts // world, world - 1)

    def test_many_rows_and_many_experts(self):
        """128 local experts (several expert groups of the schedule kernel) with 320 routes per CUDA thread (several
        register batches): the per-column state must carry across both loops."""
        topk_all = _uniform(4, 32768, 512, 10, seed=7)
        self._case(topk_all, 128, 1)

    def test_op_matches_reference(self):
        """``cudnn.moe_dispatch_mlp_combine.ops.schedule`` equals the reference entry for entry, and its outputs do not alias."""
        from cudnn.moe_dispatch_mlp_combine.ops import schedule

        world, tokens, experts, topk = 8, 1024, 256, 6
        topk_all = _uniform(world, tokens, experts, topk, seed=11)
        capacity = world * tokens * topk + (experts // world) * 256
        expected = _reference(topk_all, experts // world, capacity, 2)
        actual = schedule(topk_all, experts // world, capacity, 2)
        torch.cuda.synchronize()
        routed = expected[0] >= 0
        for name, want, got in (
            ("peer_rank", expected[0], actual[0]),
            ("peer_token_idx", expected[1][routed], actual[1][routed]),
            ("num_tokens", expected[2], actual[2]),
            ("tokens_per_expert", expected[3], actual[3]),
        ):
            self.assertTrue(torch.equal(want, got), name)
        # The two per-capacity-entry outputs must be separate buffers; the three small per-expert
        # counts may share one allocation.  (The stricter "no output aliases another" was a
        # torch.library custom-op requirement, and schedule is a plain function now.)
        peer_rank, peer_token_idx = actual[0], actual[1]
        self.assertNotEqual(
            int(peer_rank.untyped_storage().data_ptr()), int(peer_token_idx.untyped_storage().data_ptr()), "peer_rank and peer_token_idx share a buffer"
        )

    def test_skewed_and_empty_experts(self):
        """Every route of every peer on one expert of rank 0; ranks without routes; a single local expert;
        every route on one rank spread over its experts; the capacity exactly at the padded total."""
        world, tokens, topk = 8, 2304, 1
        topk_all = torch.full((world, tokens, topk), 3, dtype=torch.int32, device="cuda")
        for rank in (0, 5):
            with self.subTest(rank=rank):
                self._case(topk_all, 8, rank)
        with self.subTest(case="capacity exactly at the padded total"):
            # Rank 0 receives all 8 * 2304 routes on expert 3: 18432 rows, already a multiple of 256.
            self._case(topk_all, 8, 0, capacity=world * tokens * topk)
        pattern = torch.arange(tokens, dtype=torch.int32, device="cuda").remainder(world)
        with self.subTest(case="one local expert, round robin"):
            self._case(pattern.view(1, tokens, 1).expand(world, -1, -1).contiguous(), 1, 2)
        with self.subTest(case="all routes to rank 6, spread over its 12 experts"):
            experts = 12
            routes = (torch.arange(world * tokens * 4, dtype=torch.int32, device="cuda") * 7).remainder(experts) + 6 * experts
            topk_all = routes.view(world, tokens, 4)
            for rank in (6, 0):
                self._case(topk_all, experts, rank)

    def test_single_token_topk_one(self):
        """T = 1 and top-k 1 (below the CUDA op's host minimum; the raw binding is the reference)."""
        world, experts = 8, 8
        topk_all = torch.tensor([[[5]], [[13]], [[5]], [[7]], [[0]], [[5]], [[63]], [[6]]], dtype=torch.int32, device="cuda")
        # Rank 0 receives experts 0, 5, 6, 7 (4 padded tiles = 1024 rows), rank 7 one route on expert 63.
        for rank, capacity in ((0, 1024), (7, 256)):
            with self.subTest(rank=rank):
                self._case(topk_all, experts, rank, capacity=capacity)

    def test_repeated_calls_reuse_freed_allocations(self):
        """Two calls of the same geometry in one process with different routings: the second call's outputs come from the
        caching allocator's reuse of the first call's freed buffers, so any entry the kernels do not write (a count, a
        padding row, the tail) would carry the first routing's values."""
        from cudnn.moe_dispatch_mlp_combine.cutedsl.scheduler import scheduler

        world, tokens, experts, topk = 8, 4096, 128, 6
        dense = _uniform(world, tokens, experts, topk, seed=1)
        capacity = world * tokens * topk + (experts // world) * 256
        # Sparse: only peer 2's first 100 tokens route to rank 3 (expert 50 of its 16), everything else elsewhere.
        sparse = torch.full((world, tokens, topk), 7, dtype=torch.int32, device="cuda")  # expert 7 is on rank 0
        sparse[2, :100, 0] = 50
        first = scheduler(dense, experts // world, capacity, 3)
        torch.cuda.synchronize()
        pointers = tuple(int(t.data_ptr()) for t in first)
        del first
        second = scheduler(sparse, experts // world, capacity, 3)
        torch.cuda.synchronize()
        self.assertEqual(
            tuple(int(t.data_ptr()) for t in second), pointers, "the second call did not reuse the first call's buffers; the test no longer covers stale data"
        )
        self._compare(sparse, experts // world, 3, capacity, second)


class SchedulerHost(unittest.TestCase):
    def test_output_and_scratch_sections_are_16_byte_aligned(self):
        """The consumers require 16-byte aligned schedule tensors; the output and scratch sections keep that for every E / EP."""
        from cudnn.moe_dispatch_mlp_combine.cutedsl.scheduler import output_sections, scratch_sections

        for experts, world in [(1, 1), (3, 8), (48, 8), (64, 8), (512, 1), (288, 4), (9, 64)]:
            total, padded, valid, output_size = output_sections(experts)
            min_counts, counts, chunk_counts, column_counts, scratch_size = scratch_sections(experts, world)
            with self.subTest(experts=experts, world=world):
                self.assertEqual((total, min_counts), (0, 0))
                for offset in (padded, valid, output_size, counts, chunk_counts, column_counts):
                    self.assertEqual(offset % 4, 0)
                self.assertGreaterEqual(valid - padded, experts)
                self.assertGreaterEqual(output_size - valid, experts)
                self.assertGreaterEqual(counts - min_counts, experts)
                self.assertGreaterEqual(chunk_counts - counts, experts * world)
                self.assertGreaterEqual(column_counts - chunk_counts, experts * world * 32)
                self.assertGreaterEqual(scratch_size - column_counts, experts * world * 1024)


@unittest.skipUnless(torch.cuda.is_available(), "the limits are read from the current device")
class SchedulerLimits(unittest.TestCase):
    def test_shared_memory_limits_at_their_exact_boundaries(self):
        """The host validation (no launch) accepts the largest E and EP the two shared-memory formulas allow on this device and
        rejects the next value with a ValueError naming the limit: 128 * E <= the opt-in shared memory per block (the count
        histogram; 1816 on B300 / GB300), and 256 * EP + 4 * E <= the bytes the schedule kernel's static block leaves for its
        runtime tables (EP 134 at E = 8)."""
        from cudnn.moe_dispatch_mlp_combine.cutedsl import scheduler as module
        from cudnn.moe_dispatch_mlp_combine.cutedsl.scheduler_schedule import _FIXED_BYTES, _GROUP, _SMEM_BYTES

        budget = torch.cuda.get_device_properties(torch.cuda.current_device()).shared_memory_per_block_optin
        max_experts = budget // 128
        routes = torch.zeros((1, 512, 2), dtype=torch.int32, device="cuda")
        module._validate_inputs(routes, max_experts, 1024, 0)  # accepted: returns the validated geometry
        with self.assertRaisesRegex(ValueError, "count histogram"):
            module._validate_inputs(routes, max_experts + 1, 1024, 0)
        experts = 8
        max_ranks = (_SMEM_BYTES - _FIXED_BYTES - 4 * experts) // (4 * _GROUP)
        module._validate_inputs(torch.zeros((max_ranks, 8, 1), dtype=torch.int32, device="cuda"), experts, 2048, 0)
        with self.assertRaisesRegex(ValueError, "schedule kernel's tables"):
            module._validate_inputs(torch.zeros((max_ranks + 1, 8, 1), dtype=torch.int32, device="cuda"), experts, 2048, 0)
        self.assertEqual((max_experts, max_ranks), (budget // 128, 134))


# --------------------------------------------------------------------------------------------------------------------
# Forward epilogue kernel
#
# CuTe DSL BF16 forward epilogue against the PyTorch reference, bit for bit.
#
# ``mok::fwd_epilogue`` is the reference: y_shared + sum_k w_k * combine_k, FP32 accumulation in route
# order and one BF16 rounding.  Geometries cover hidden sizes whose last 1024-column tile holds 256, 768 or
# 1024 columns, top-k 1, 2, 4, 6, 7, 8, 9, 10, 12, 13, 16, 17, 18, 24, 26, 27 and 40 (both sides of every resident-CTA
# band edge on GB300, six down to one CTA per SM, and the two load rounds of top-k + 1 > 32), tile counts that are not
# a multiple of the persistent grid, CTAs with a single tile, and CTAs that reuse both slots many times.  Random normal data: subnormal results are not
# covered (the CUDA build flushes them to zero, the CuTe DSL kernel does not).
#
# One GPU, no distributed setup: ``python -m pytest moe_dispatch_mlp_combine/cutedsl/test_ops.py``.


def _inputs(tokens, hidden, topk, seed, missing_route_fraction=0.0):
    """Inputs of one epilogue call.

    ``missing_route_fraction`` marks that share of the route slots unused (-1) and poisons their
    combine rows with NaN: the kernel must skip them, and weighting by zero would not -- the
    symmetric buffer those rows come from is allocated, not zeroed.
    """
    generator = torch.Generator(device="cuda").manual_seed(seed)
    y_shared = torch.randn(tokens, hidden, device="cuda", generator=generator).to(torch.bfloat16)
    combine = torch.randn(tokens * topk, hidden, device="cuda", generator=generator).to(torch.bfloat16)
    weights = torch.rand(tokens, topk, device="cuda", generator=generator)
    routes = torch.zeros(tokens, topk, device="cuda", dtype=torch.int32)
    if missing_route_fraction:
        unused = torch.rand(routes.shape, generator=generator, device="cuda") < missing_route_fraction
        routes = routes.masked_fill(unused, -1)
        combine.view(tokens, topk, hidden)[unused] = float("nan")
    return y_shared, combine, weights, routes


@unittest.skipUnless(_blackwell(), "the CuTe DSL forward epilogue is SM100/SM103 only")
class FwdEpilogueMatchesReference(unittest.TestCase):

    def assert_close(self, expected, actual, what=""):
        """Within the suite's BF16 tolerance of the reference: max abs error and relative L1.

        The kernel and the PyTorch model both accumulate the top-k terms in FP32 and round once,
        but not in the same order, and FP32 addition is not associative.  A per-element relative
        bound is the wrong model here: the terms are signed and cancel, so an output can sit five
        orders of magnitude below the terms that produced it and carry a large relative error from
        an absolute one that is tiny.  BF16_TOLERANCE bounds exactly that pair.  (utils.py's
        get_error_stats all-reduces its statistics; this test is single-GPU, so the same two
        statistics are computed locally.)

        Byte equality is asserted where it does hold, between repeated calls of the kernel
        (assert_bitwise) -- the determinism MCore's replay test needs.
        """
        from .utils import BF16_TOLERANCE

        max_abs, max_relative_l1 = BF16_TOLERANCE
        difference = (expected.float() - actual.float()).abs()
        abs_error_max = difference.max().item()
        reference_sum = expected.float().abs().sum()
        relative_l1 = 0.0 if reference_sum == 0 else (difference.sum() / reference_sum).item()
        self.assertLessEqual(abs_error_max, max_abs, f"{what}max abs error")
        self.assertLessEqual(relative_l1, max_relative_l1, f"{what}relative L1 error")

    def assert_bitwise(self, expected, actual, what=""):
        differing = (expected.view(torch.int16) != actual.view(torch.int16)).nonzero()
        if differing.numel():
            t, c = (int(v) for v in differing[0])
            self.fail(f"{what}{differing.shape[0]} elements differ; first at ({t}, {c}): " f"{float(expected[t, c])} vs {float(actual[t, c])}")

    def test_random(self):
        from cudnn.moe_dispatch_mlp_combine.cutedsl.fwd_epilogue import fwd_epilogue_bf16
        from .utils import run_fwd_epilogue_reference

        # Tiles per CTA on a 152-SM GB300 (grid = min(tiles, SMs x CTAs per SM); top-1 to top-7 run 6 CTAs per SM,
        # top-8 5, top-10 4, top-16 3, top-18 and top-24 2, top-40 1): (512, 256, *) 512 tiles, one per CTA;
        # (512, 1280, 10) 1024 tiles, 1-2 per CTA; (2560, 4096, 10) 10240 tiles, 16-17 per CTA, both slots reused
        # many times; (1024, 7168, 6) 7168 tiles, 7-8 per CTA at 6 CTAs per SM; the 768 / 1280 / 2560-token cases
        # have tile counts that are not a multiple of the grid.  (512, 1792, 16): three CTAs per SM, 2-3 tiles per
        # CTA; (512, 4096, 18) and (768, 3072, 24): two CTAs per SM, 6-8 tiles per CTA; (512, 1280, 40): one CTA per
        # SM, two load rounds per tile (41 vectors), 6-7 tiles per CTA.
        for tokens, hidden, topk in [
            (512, 4096, 10),
            (1024, 7168, 6),
            (768, 7168, 4),
            (512, 2304, 1),
            (1280, 1280, 8),
            (512, 1792, 16),
            (768, 3328, 2),
            (512, 256, 10),
            (512, 256, 1),
            (512, 1280, 10),
            (2560, 4096, 10),
            (512, 4096, 18),
            (768, 3072, 24),
            (512, 1280, 40),
            # both sides of every resident-CTA band edge on GB300 (6/5 at top-k 7/8, 5/4 at
            # 9/10, 4/3 at 12/13, 3/2 at 17/18, 2/1 at 26/27): 2048 tiles, 2-14 per CTA
            (512, 4096, 7),
            (512, 4096, 8),
            (512, 4096, 9),
            (512, 4096, 12),
            (512, 4096, 13),
            (512, 4096, 17),
            (512, 4096, 26),
            (512, 4096, 27),
        ]:
            with self.subTest(tokens=tokens, hidden=hidden, topk=topk):
                y_shared, combine, weights, routes = _inputs(tokens, hidden, topk, tokens + hidden + topk)
                expected = run_fwd_epilogue_reference(y_shared, combine, weights, routes)
                actual = fwd_epilogue_bf16(y_shared, combine, weights, routes)
                self.assert_close(expected, actual)
                del expected, actual, y_shared, combine, weights, routes

    def test_unused_route_slots_are_skipped(self):
        """A slot marked -1 contributes nothing, even when its combine row is NaN.

        This is the property that makes weighting by zero insufficient: the dispatch never writes
        the combine rows of a token's unused slots, and the symmetric buffer they live in is
        allocated rather than zeroed, so whatever is there may not be finite.
        """
        from cudnn.moe_dispatch_mlp_combine.cutedsl.fwd_epilogue import fwd_epilogue_bf16
        from .utils import run_fwd_epilogue_reference

        for tokens, hidden, topk in [(512, 1024, 8), (512, 4096, 10), (512, 256, 1)]:
            with self.subTest(tokens=tokens, hidden=hidden, topk=topk):
                y_shared, combine, weights, routes = _inputs(tokens, hidden, topk, tokens + hidden + topk, missing_route_fraction=0.4)
                self.assertTrue((routes < 0).any(), "the case has no unused slot")
                self.assertTrue(torch.isnan(combine.float()).any(), "no poisoned row to skip")
                actual = fwd_epilogue_bf16(y_shared, combine, weights, routes)
                self.assertTrue(torch.isfinite(actual.float()).all(), "a poisoned row of an unused slot reached the output")
                self.assert_close(run_fwd_epilogue_reference(y_shared, combine, weights, routes), actual)

    def test_token_with_no_routes_returns_the_shared_term(self):
        """Every slot of a token may be unused; its output is the shared term alone."""
        from cudnn.moe_dispatch_mlp_combine.cutedsl.fwd_epilogue import fwd_epilogue_bf16

        tokens, hidden, topk = 512, 1024, 6
        y_shared, combine, weights, _ = _inputs(tokens, hidden, topk, seed=7)
        combine.fill_(float("nan"))
        routes = torch.full((tokens, topk), -1, device="cuda", dtype=torch.int32)
        self.assert_bitwise(y_shared, fwd_epilogue_bf16(y_shared, combine, weights, routes))

    def test_repeated_calls_same_shape(self):
        """Two calls of one shape with different data: the second result must not carry anything from the first
        (stale ring slots, weights or a reused output buffer)."""
        from cudnn.moe_dispatch_mlp_combine.cutedsl.fwd_epilogue import fwd_epilogue_bf16
        from .utils import run_fwd_epilogue_reference

        tokens, hidden, topk = 1024, 4096, 6
        for seed in (11, 12):
            y_shared, combine, weights, routes = _inputs(tokens, hidden, topk, seed)
            expected = run_fwd_epilogue_reference(y_shared, combine, weights, routes)
            actual = fwd_epilogue_bf16(y_shared, combine, weights, routes)
            self.assert_close(expected, actual, what=f"seed {seed}: ")
            # The same call twice must give the same bytes: nothing carried over from the first.
            self.assert_bitwise(actual, fwd_epilogue_bf16(y_shared, combine, weights, routes), what=f"seed {seed} replay: ")
            del expected, actual, y_shared, combine, weights, routes

    def test_routed_only_is_the_zero_shared_sum(self):
        """No shared output (a routed-only layer, C++ ``fwd_epilogue_routed_kernel``): the
        accumulator starts at zero, which is the sum with an all-zero shared term bit for bit --
        0 + x is x in FP32 and the route order is the same.  A token without routes gives exact
        zeros, and an unused slot's NaN row is still skipped."""
        from cudnn.moe_dispatch_mlp_combine.cutedsl.fwd_epilogue import fwd_epilogue_bf16
        from .utils import run_fwd_epilogue_reference

        for tokens, hidden, topk, missing in [(512, 4096, 10, 0.0), (768, 7168, 4, 0.3), (512, 256, 1, 0.0), (512, 1280, 40, 0.2)]:
            with self.subTest(tokens=tokens, hidden=hidden, topk=topk, missing=missing):
                y_shared, combine, weights, routes = _inputs(tokens, hidden, topk, tokens + hidden + topk, missing_route_fraction=missing)
                zero = torch.zeros_like(y_shared)
                routed_only = fwd_epilogue_bf16(None, combine, weights, routes)
                self.assert_bitwise(fwd_epilogue_bf16(zero, combine, weights, routes), routed_only)
                self.assert_close(run_fwd_epilogue_reference(zero, combine, weights, routes), routed_only)
        tokens, hidden, topk = 512, 1024, 6
        _, combine, weights, _ = _inputs(tokens, hidden, topk, seed=8)
        combine.fill_(float("nan"))
        routes = torch.full((tokens, topk), -1, device="cuda", dtype=torch.int32)
        self.assert_bitwise(torch.zeros(tokens, hidden, dtype=torch.bfloat16, device="cuda"), fwd_epilogue_bf16(None, combine, weights, routes))
        gate = torch.ones(tokens, 1, dtype=torch.bfloat16, device="cuda")
        with self.assertRaisesRegex(ValueError, "requires a shared output"):
            fwd_epilogue_bf16(None, combine, weights, routes, gate)

    def test_routed_only_refill_of_a_token_without_routes(self):
        """Routed-only, a ring slot refilled with a tile that loads nothing (its token has no used route slot):
        the slot's barrier then expects zero bytes, which must not complete its next phase while other warps
        still wait for the current one (the kernel would not return).  (2560, 4096, 4): 11-12 tiles per CTA on
        GB300, every other token without routes, so half of the refills are such tiles; repeated, because the
        ordering that hangs is rare."""
        from cudnn.moe_dispatch_mlp_combine.cutedsl.fwd_epilogue import fwd_epilogue_bf16

        tokens, hidden, topk = 2560, 4096, 4
        y_shared, combine, weights, routes = _inputs(tokens, hidden, topk, seed=9)
        routes[::2] = -1
        combine.view(tokens, topk, hidden)[::2] = float("nan")
        expected = fwd_epilogue_bf16(torch.zeros_like(y_shared), combine, weights, routes)
        for _ in range(20):
            routed_only = fwd_epilogue_bf16(None, combine, weights, routes)
            torch.cuda.synchronize()
        self.assert_bitwise(expected, routed_only)
        self.assertFalse(routed_only[::2].float().abs().sum().item(), "a token without routes is not zero")

    def test_one_executor_serves_every_shape(self):
        """T, H, top-k, tile and slot counts are runtime parameters: alternating shapes compile one executor."""
        from cudnn.moe_dispatch_mlp_combine.cutedsl import fwd_epilogue as module

        module.clear_fwd_epilogue_executor_cache()
        for tokens, hidden, topk in [(512, 1024, 2), (768, 2048, 6), (512, 1024, 2), (1024, 4096, 10), (512, 1024, 24)]:
            y_shared, combine, weights, routes = _inputs(tokens, hidden, topk, tokens + topk)
            module.fwd_epilogue_bf16(y_shared, combine, weights, routes)
        torch.cuda.synchronize()
        self.assertEqual(len(module._EXECUTOR_CACHE), 1)


# --------------------------------------------------------------------------------------------------------------------
# Backward epilogue kernel
#
# CuTe DSL BF16 backward epilogue: the PyTorch reference, and the unused route slots.
#
# The backward sums the routed gradients with no weight, so a slot a token does not use cannot be
# cancelled by a zero -- it has to be skipped.  Its row of ``d_x_routed_buffer`` was never written
# and the symmetric buffer is allocated rather than zeroed, so the rows are poisoned with NaN here:
# an implementation that reads them fails.
#
# One GPU, no distributed setup: ``python -m pytest moe_dispatch_mlp_combine/cutedsl/test_ops.py``.


def _supported() -> bool:
    return torch.cuda.is_available() and torch.cuda.get_device_capability() == TARGET_CAPABILITY


def _bwd_epilogue_inputs(tokens, hidden, topk, seed, missing_route_fraction=0.0):
    generator = torch.Generator(device="cuda").manual_seed(seed)
    d_x_shared = torch.randn(tokens, hidden, device="cuda", generator=generator).to(torch.bfloat16)
    routed = torch.randn(tokens * topk, hidden, device="cuda", generator=generator).to(torch.bfloat16)
    routes = torch.zeros(tokens, topk, device="cuda", dtype=torch.int32)
    if missing_route_fraction:
        unused = torch.rand(routes.shape, generator=generator, device="cuda") < missing_route_fraction
        routes = routes.masked_fill(unused, -1)
        routed.view(tokens, topk, hidden)[unused] = float("nan")
    return d_x_shared, routed, routes


@unittest.skipUnless(_supported(), f"the CuTe DSL backward epilogue is compiled for {TARGET_ARCH}")
class BwdEpilogueMatchesReference(unittest.TestCase):

    def assert_close(self, expected, actual, what=""):
        from .utils import BF16_TOLERANCE

        max_abs, max_relative_l1 = BF16_TOLERANCE
        difference = (expected.float() - actual.float()).abs()
        reference_sum = expected.float().abs().sum()
        relative_l1 = 0.0 if reference_sum == 0 else (difference.sum() / reference_sum).item()
        self.assertLessEqual(difference.max().item(), max_abs, f"{what}max abs error")
        self.assertLessEqual(relative_l1, max_relative_l1, f"{what}relative L1 error")

    def test_random(self):
        from cudnn.moe_dispatch_mlp_combine.cutedsl.bwd_epilogue import bwd_epilogue_bf16
        from .utils import run_bwd_epilogue_reference

        for tokens, hidden, topk in [(512, 1024, 8), (512, 4096, 10), (1024, 7168, 6), (512, 256, 1)]:
            with self.subTest(tokens=tokens, hidden=hidden, topk=topk):
                d_x_shared, routed, routes = _bwd_epilogue_inputs(tokens, hidden, topk, tokens + hidden + topk)
                self.assert_close(run_bwd_epilogue_reference(d_x_shared, routed, routes), bwd_epilogue_bf16(d_x_shared, routed, routes))

    def test_unused_route_slots_are_skipped(self):
        from cudnn.moe_dispatch_mlp_combine.cutedsl.bwd_epilogue import bwd_epilogue_bf16
        from .utils import run_bwd_epilogue_reference

        for tokens, hidden, topk in [(512, 1024, 8), (512, 4096, 10)]:
            with self.subTest(tokens=tokens, hidden=hidden, topk=topk):
                d_x_shared, routed, routes = _bwd_epilogue_inputs(tokens, hidden, topk, tokens + hidden, missing_route_fraction=0.4)
                self.assertTrue((routes < 0).any(), "the case has no unused slot")
                actual = bwd_epilogue_bf16(d_x_shared, routed, routes)
                self.assertTrue(torch.isfinite(actual.float()).all(), "a poisoned row of an unused slot reached the output")
                self.assert_close(run_bwd_epilogue_reference(d_x_shared, routed, routes), actual)

    def test_token_with_no_routes_returns_the_shared_term(self):
        from cudnn.moe_dispatch_mlp_combine.cutedsl.bwd_epilogue import bwd_epilogue_bf16

        tokens, hidden, topk = 512, 1024, 6
        d_x_shared, routed, _ = _bwd_epilogue_inputs(tokens, hidden, topk, seed=7)
        routed.fill_(float("nan"))
        routes = torch.full((tokens, topk), -1, device="cuda", dtype=torch.int32)
        actual = bwd_epilogue_bf16(d_x_shared, routed, routes)
        self.assertTrue(torch.equal(d_x_shared.view(torch.int16), actual.view(torch.int16)))

    def test_routed_only_is_the_zero_shared_sum(self):
        """No shared gradient (a routed-only layer, C++ ``bwd_epilogue_kernel<false>``): the
        accumulator starts at zero, the sum with an all-zero shared term bit for bit; a token
        without routes gives exact zeros, and unused slots' NaN rows are skipped."""
        from cudnn.moe_dispatch_mlp_combine.cutedsl.bwd_epilogue import bwd_epilogue_bf16
        from .utils import run_bwd_epilogue_reference

        for tokens, hidden, topk, missing in [(512, 1024, 8, 0.0), (512, 4096, 10, 0.4), (1024, 7168, 6, 0.0), (512, 256, 1, 0.0)]:
            with self.subTest(tokens=tokens, hidden=hidden, topk=topk, missing=missing):
                d_x_shared, routed, routes = _bwd_epilogue_inputs(tokens, hidden, topk, tokens + hidden + topk, missing_route_fraction=missing)
                zero = torch.zeros_like(d_x_shared)
                routed_only = bwd_epilogue_bf16(None, routed, routes)
                self.assertTrue(torch.equal(bwd_epilogue_bf16(zero, routed, routes).view(torch.int16), routed_only.view(torch.int16)))
                self.assert_close(run_bwd_epilogue_reference(zero, routed, routes), routed_only)
        tokens, hidden, topk = 512, 1024, 6
        _, routed, _ = _bwd_epilogue_inputs(tokens, hidden, topk, seed=9)
        routed.fill_(float("nan"))
        routes = torch.full((tokens, topk), -1, device="cuda", dtype=torch.int32)
        actual = bwd_epilogue_bf16(None, routed, routes)
        self.assertTrue(torch.equal(actual.view(torch.int16), torch.zeros_like(actual).view(torch.int16)))


# --------------------------------------------------------------------------------------------------------------------
# Epilogues over more than the maximum grid y tokens
#
# Miscellaneous tests.


def test_epilogues_support_more_than_max_grid_y_tokens(
    context: tuple[int, int, torch.device],
) -> None:
    _, _, device = context
    num_local_tokens = 2 * (65_535 + 1)
    hidden_dim = 256
    shared = torch.full(
        (num_local_tokens, hidden_dim),
        1.0,
        device=device,
        dtype=torch.bfloat16,
    )
    routed = torch.full_like(shared, 2.0)
    topk_weights = torch.full(
        (num_local_tokens, 1),
        0.25,
        device=device,
        dtype=torch.float32,
    )

    top_experts = torch.zeros((num_local_tokens, 1), device=device, dtype=torch.int32)

    output = ops.fwd_epilogue(shared, routed, topk_weights, top_experts)
    assert torch.all(output == 1.5)

    d_x = ops.bwd_epilogue(shared, routed, top_experts)
    assert torch.all(d_x == 3.0)

    # The same inputs with the single slot unused: both epilogues return the shared term alone.
    unused = torch.full_like(top_experts, -1)
    assert torch.all(ops.fwd_epilogue(shared, routed, topk_weights, unused) == 1.0)
    assert torch.all(ops.bwd_epilogue(shared, routed, unused) == 1.0)


# --------------------------------------------------------------------------------------------------------------------
# MXFP8 quantization kernel
#
# CuTe DSL MXFP8 quantization against the PyTorch reference, byte for byte.
#
# ``mok::mxfp8_quantize`` (``csrc/mxfp8.cuh``) is the reference.  MXFP8's end-to-end
# tolerance is loose enough to hide a wrong scale rounding, so the port's quantization
# is pinned here at the byte level instead: same input, identical payload and scale
# bytes in both the normal and the transposed layout.
#
# One GPU, no distributed setup: ``pytest moe_dispatch_mlp_combine/cutedsl/test_ops.py``.


requires_blackwell = pytest.mark.skipif(
    not (torch.cuda.is_available() and torch.cuda.get_device_capability() in SUPPORTED_CAPABILITIES),
    reason="the CuTe DSL MXFP8 quantizer is SM100/SM103 only",
)


OUTPUT_NAMES = ("x_fp8", "x_sc", "x_fp8_t", "x_sc_t")


def _quantize_reference(x, return_normal=True, return_transposed=True):
    """The PyTorch model of the quantization, not another call into the kernel under test."""
    from .utils import run_mxfp8_quantize_reference

    return run_mxfp8_quantize_reference(x, return_normal, return_transposed)


def _ported(x, return_normal=True, return_transposed=True):
    from cudnn.moe_dispatch_mlp_combine.cutedsl.mxfp8_quantize import mxfp8_quantize

    return mxfp8_quantize(x, return_normal, return_transposed)


def _assert_same_bytes(name, expected, actual):
    assert tuple(actual.shape) == tuple(expected.shape), f"{name}: shape"
    assert actual.dtype == expected.dtype, f"{name}: dtype"
    expected_bytes = expected.reshape(-1).view(torch.uint8)
    actual_bytes = actual.reshape(-1).view(torch.uint8)
    differing = expected_bytes != actual_bytes
    count = int(differing.sum().item())
    if count:
        first = int(differing.nonzero()[0].item())
        pytest.fail(
            f"{name}: {count}/{expected_bytes.numel()} bytes differ; first at "
            f"{first}: reference {int(expected_bytes[first])} vs kernel {int(actual_bytes[first])}"
        )


def _assert_same_quantization(x):
    for name, expected, actual in zip(OUTPUT_NAMES, _quantize_reference(x), _ported(x), strict=True):
        _assert_same_bytes(name, expected, actual)


@requires_blackwell
@pytest.mark.parametrize("shape", [(128, 128), (128, 512), (384, 256), (1024, 4096), (3, 256, 384)])
def test_random(shape):
    """Random normal activations over several geometries, 2-D and 3-D."""
    torch.manual_seed(0)
    _assert_same_quantization(torch.randn(shape, dtype=torch.bfloat16, device="cuda"))


@requires_blackwell
def test_block_boundaries():
    """Values that exercise the scale rounding, the floor and e4m3 saturation.

    One row per pattern, every value exactly representable in BF16 (the kernel's input
    type, so the pattern survives the conversion): zeros (amax 0 takes the 1e-12 floor),
    exact powers of two (the e8m0 exponent is exact), e4m3's 448 maximum and its BF16
    neighbours 446 and 450, magnitudes around the scale floor, a block whose amax sits one
    BF16 ulp below a power of two (127.5, where round-toward-+inf and round-to-nearest
    disagree), and two ramps that mix magnitudes within a block.
    """
    exact_rows = [
        torch.zeros(128),
        torch.full((128,), 448.0),
        torch.full((128,), 450.0),
        torch.full((128,), 446.0),
        2.0 ** torch.arange(-20, 108, dtype=torch.float32),
        torch.full((128,), 2.0**-40),
        torch.full((128,), 2.0**-14),
        torch.full((128,), 127.5),
    ]
    for row in exact_rows:
        assert torch.equal(row.to(torch.bfloat16).float(), row), "pattern is not BF16-exact"
    rows = exact_rows + [
        torch.linspace(-448.0, 448.0, 128),
        torch.linspace(-1e-6, 1e-6, 128),
    ]
    block = torch.stack(rows)
    # 128 rows: the patterns, their negatives and both column-reversed, repeated.
    variants = torch.cat([block, -block, block.flip(1), -block.flip(1)])
    x = variants.repeat(-(-128 // variants.shape[0]), 1)[:128]
    _assert_same_quantization(x.to(torch.bfloat16).cuda().contiguous())


@requires_blackwell
def test_single_layouts():
    """``return_normal`` / ``return_transposed`` select exactly one pair each."""
    torch.manual_seed(1)
    x = torch.randn((256, 256), dtype=torch.bfloat16, device="cuda")
    both = _ported(x)

    normal = _ported(x, True, False)
    assert normal[2] is None and normal[3] is None
    _assert_same_bytes("x_fp8", both[0], normal[0])
    _assert_same_bytes("x_sc", both[1], normal[1])

    transposed = _ported(x, False, True)
    assert transposed[0] is None and transposed[1] is None
    _assert_same_bytes("x_fp8_t", both[2], transposed[2])
    _assert_same_bytes("x_sc_t", both[3], transposed[3])
