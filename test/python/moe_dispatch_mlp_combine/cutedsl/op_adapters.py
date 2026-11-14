# Copyright 2026 Cursor Research
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Derived from Mixture-of-Kittens (https://github.com/cursor/mixture-of-kittens, commit 22fc95ae); modified by NVIDIA.

"""The C++ port's fused-op ABI over the CuTe DSL entries, for test_ops.py.

The C++ build exposes its megakernels as four torch ops taking raw buffers, pointer lists and
schedule tensors (``mok.ops.dispatch_mlp_swiglu_combine_{fwd,bwd}_{bf16,mxfp8}`` at d59188a).  This
port has no such ops: ``functional`` calls ``forward_bf16`` / ``forward_mxfp8`` / ``backward_bf16`` /
``backward_mxfp8`` with a workspace and a schedule.  These adapters take exactly the C++ ops'
arguments, build the workspace / schedule / context views those entries read, call them and return
the outputs in the C++ ops' order, so the C++ op tests run here unchanged.

Two things the C++ signatures do not carry are derived, not invented:

* ``valid_tokens_per_expert`` (the BF16 forward's per-expert unpadded row count, the CuTe DSL
  scheduler's fifth output): the scheduler marks every unrouted entry of the schedule -1, so it is
  the number of entries with ``peer_rank >= 0`` in each expert's padded segment.  test_ops'
  test_schedule checks that this derivation reproduces the scheduler's own output.
* the workspace's rank: the position of the local buffer in its pointer list.

The C++ ops' SiTU convention -- ``situ_beta1 == situ_beta2 == 0`` means SwiGLU -- maps to
``situ_beta=None``.  The C++ BF16 backward returns fifteen tensors where this one has thirteen (no
materialized dHidden, the SwiGLU backward runs in the Down dgrad epilogue); those two slots are None.
"""

from types import SimpleNamespace

import torch


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
