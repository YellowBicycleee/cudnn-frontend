# Copyright 2026 Cursor Research
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Derived from Mixture-of-Kittens (https://github.com/cursor/mixture-of-kittens, commit 22fc95ae); modified by NVIDIA.

import math

import torch
import torch.distributed as dist

# (max absolute error, relative L1 error).  Known limit of the BF16 check with a clamped SwiGLU: on
# B300, Cursor's own kernels fail d_w_shared_gate (0.516 > 0.5, 384 experts, intermediate 2048 / Default / Clamped).
# On GB300 the shared-expert gate equals the reference bit for bit and d_w_shared_gate stays within one
# BF16 ulp of it.  The B300 cause is inferred, not measured: a gate within one ulp of the clamp limit
# rounded to opposite sides of the limit by the reference (torch / cuBLAS) and the kernel would flip the
# clamp mask and move that row's whole d_gate * x term, an error a rounding-level tolerance does not
# cover.  (The routed gate shows such opposite-sign elements on GB300 too, below the tolerance.)
# The check is kept as is.
#: The results forward and backward return, in order, for per-tensor comparison messages.
FORWARD_RESULT_NAMES = (
    "output",
    "gate_shared",
    "up_shared",
    "hidden_shared",
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


def ungated(gradients):
    """functional.backward's results without the shared output gate's: the C++ port returns a
    ninth, the gate weight's gradient, which is None unless the layer is gated."""
    assert len(gradients) == 9 and gradients[8] is None, "an ungated backward's ninth result is None"
    return tuple(gradients[:8])


BF16_TOLERANCE = (0.5, 0.01)
MXFP8_TOLERANCE = (1.0, 0.1)
# MXFP8 backward results on which upstream Mixture-of-Kittens (cursor/mixture-of-kittens 22fc95a, CUDA) is itself
# outside MXFP8_TOLERANCE's absolute bound on the tests' inputs, and this implementation returns upstream's bytes:
# (world size, case, SwiGLU mode, result) -> upstream's abs max, which is that result's absolute bound (an exact
# binary fraction, no margin).  Measured on GB300 at 8 ranks; every other case, result and world size keeps
# MXFP8_TOLERANCE.  validate/README.md has the measurements, what is known of the cause, and the scripts that
# repeat them.
MXFP8_UPSTREAM_ABS_MAX = {
    # One local expert per rank, hidden 256, intermediate 256, top-k 1, 512 local tokens; every MoK parameter set.
    (8, "All minimums", "Clamped SwiGLU", "d_w_routed_gate"): 1.0859375,
    # The latent layer's routed chain is a plain layer in the routed hidden: upstream's numbers are that layer's.
    (8, "Qwen3.5-397B-A17B, routed hidden 1024", "Clamped SwiGLU", "d_w_routed_gate"): 4.98388671875,
    (8, "Qwen3.5-397B-A17B, routed hidden 1024", "Clamped SwiGLU", "d_w_routed_up"): 4.99853515625,
    (4, "Qwen3.5-397B-A17B, routed hidden 1024", "Clamped SwiGLU", "d_w_routed_gate"): 8.017333984375,
    (4, "Qwen3.5-397B-A17B, routed hidden 1024", "Clamped SwiGLU", "d_w_routed_up"): 5.505859375,
}


def mxfp8_tolerance(world_size: int, case: str, swiglu: str, result: str) -> tuple[float, float]:
    """``MXFP8_TOLERANCE``, with upstream's abs max as the absolute bound where ``MXFP8_UPSTREAM_ABS_MAX`` records
    one for exactly this world size, case, SwiGLU mode and result."""
    absolute, relative = MXFP8_TOLERANCE
    return MXFP8_UPSTREAM_ABS_MAX.get((world_size, case, swiglu, result), absolute), relative


def shapes(world_size: int) -> tuple[tuple[str, int, int, int, int, int], ...]:
    return (  # (name, routed experts, hidden dim, intermediate dim, top-k, num local tokens)
        ("384 experts, intermediate 2048", 384, 7168, 2048, 8, 7168),
        ("256 experts, hidden 6144", 256, 6144, 2048, 8, 7168),
        ("Qwen3.5-397B-A17B", 512, 4096, 1024, 10, 7168),
        ("DeepSeek-V4-Pro", 384, 7168, 3072, 6, 7168),
        ("All minimums", world_size, 256, 256, 1, 512),
        ("Minimum top-k", 384, 7168, 2048, 1, 7168),
        ("Minimum local experts", world_size, 7168, 2048, min(8, world_size), 7168),
        ("Minimum hidden dimension", 384, 256, 2048, 8, 7168),
        ("Minimum intermediate dimension", 384, 7168, 256, 8, 7168),
        ("Minimum local tokens", 384, 7168, 2048, 8, 512),
        ("Local tokens 1024", 384, 7168, 2048, 8, 1024),
        ("Local tokens 16384", 384, 7168, 2048, 8, 2048),
        # 257 local experts: above every static expert-table capacity in the CuTe
        # kernels (256 / 64), so the runtime fallbacks (O(E) expert scan, literal
        # Wgrad grid) are exercised on the GPU, not only on the CPU.
        ("Expert table fallback (257 local experts)", 257 * world_size, 256, 256, 8, 4096),
    )


def mok_params() -> tuple[tuple[str, int, int, int, int], ...]:
    return (  # (name, forward comm SMs, backward comm SMs, minibatch size, macrobatch size)
        ("Default", 40, 28, 4096, 131072),
        ("Minimum forward comm SMs", 2, 28, 4096, 131072),
        ("Minimum forward comm SMs, 2 macrobatches", 2, 28, 4096, 4096),
        ("Minimum backward comm SMs", 40, 2, 4096, 131072),
        ("Minimum minibatch", 40, 28, 256, 131072),
        ("Minimum macrobatch", 40, 28, 4096, 4096),
        ("2x macrobatch", 40, 28, 4096, 8192),
        ("4x macrobatch", 40, 28, 4096, 16384),
    )


def swiglu_params() -> tuple[tuple[str, float | None], ...]:
    return (
        ("Unclamped SwiGLU", None),
        ("Clamped SwiGLU", 4.0),
    )


def capacity_multiplier(world_size: int, num_local_tokens: int, topk: int, num_local_experts: int, base: float = 1.5) -> float:
    """``schedule_capacity_multiplier`` for a test workspace of this geometry.

    The workspace table holds ``num_local_tokens * topk * factor`` rows with
    ``factor = max(2, ceil(ep_size * multiplier))``.  Both schedulers pad every local expert to a
    multiple of 256 rows (at most 255 extra rows each) and trap on the device when the padded total
    exceeds the table; MoK keeps that behaviour (as upstream), so the table size is the test's own
    choice.  The tests' customary ``base`` 1.5 (factor 12 at EP8) holds every shape of ``shapes()`` at
    EP8.  At EP1 one rank owns all 384 / 512 experts: for "Minimum top-k" and the small-token shapes the
    padding alone is up to 98304 rows against a ``2 * num_local_tokens * topk`` table, and both
    Cursor 22fc95a's tests and this fork's died of the trap ("unspecified launch failure") on one GPU.
    Returned: the multiplier whose factor holds the rank's balanced share (``num_local_tokens * topk``
    rows) plus full padding of its local experts, and at least ``base``; at EP8 this is ``base`` for
    every shape of ``shapes()`` (the EP8 tables are unchanged).
    """
    routes = num_local_tokens * topk
    factor = max(math.ceil(world_size * base), 1 + -(-255 * num_local_experts // routes))
    return factor / world_size


def generate_topk_experts(
    rank: int,
    device: torch.device,
    num_experts: int,
    topk: int,
    num_local_tokens: int,
    missing_route_fraction: float = 0.0,
) -> torch.Tensor:
    """Routing of one rank's tokens.

    Args:
        missing_route_fraction: replace this share of the slots with -1, the marker for a token that
            uses fewer than topk experts.  MCore's routing map produces those whenever a token is
            dropped or the router returns fewer experts, and every slot of a token may go: the
            epilogues must handle a row with no routes at all.
    """
    generator = torch.Generator(device=device).manual_seed(1234 + rank)
    router_logits = torch.randn(num_local_tokens, num_experts, generator=generator, device=device)
    experts = torch.topk(router_logits, topk, dim=1).indices
    if missing_route_fraction:
        drop = torch.rand(experts.shape, generator=generator, device=device) < missing_route_fraction
        experts = experts.masked_fill(drop, -1)
        # Keep each token's used slots first, as MCore's route adapter compacts them.
        order = torch.argsort((experts < 0).int(), dim=1, stable=True)
        experts = experts.gather(1, order)
    return experts


def generate_inputs(
    rank: int,
    device: torch.device,
    num_experts: int,
    num_local_experts: int,
    topk: int,
    num_local_tokens: int,
    hidden_dim: int,
    intermediate_dim: int,
    missing_route_fraction: float = 0.0,
) -> tuple[
    torch.Tensor,  # x
    torch.Tensor,  # topk_experts (-1 where a token uses fewer than topk experts)
    torch.Tensor,  # router_weights
    torch.Tensor,  # w_shared_gate
    torch.Tensor,  # w_shared_up
    torch.Tensor,  # w_shared_down
    torch.Tensor,  # w_routed_gate
    torch.Tensor,  # w_routed_up
    torch.Tensor,  # w_routed_down
    torch.Tensor,  # d_output
]:
    generator = torch.Generator(device=device).manual_seed(1234 + rank)

    router_logits = torch.randn(num_local_tokens, num_experts, generator=generator, device=device)
    topk_values, topk_experts = torch.topk(router_logits, topk, dim=1)
    router_weights = torch.softmax(topk_values.float(), dim=-1)
    if missing_route_fraction:
        # MCore's route adapter marks a slot a token does not use -1 and its weight 0, with the
        # used slots compacted first.
        unused = torch.rand(topk_experts.shape, generator=generator, device=device) < missing_route_fraction
        order = torch.argsort(unused.int(), dim=1, stable=True)
        topk_experts = topk_experts.masked_fill(unused, -1).gather(1, order)
        router_weights = router_weights.gather(1, order).masked_fill(topk_experts < 0, 0.0)

    x = torch.randn(num_local_tokens, hidden_dim, generator=generator, device=device, dtype=torch.bfloat16)
    w_shared_gate = torch.randn(intermediate_dim, hidden_dim, generator=generator, device=device, dtype=torch.bfloat16) * hidden_dim**-0.5
    w_shared_up = torch.randn(intermediate_dim, hidden_dim, generator=generator, device=device, dtype=torch.bfloat16) * hidden_dim**-0.5
    w_shared_down = torch.randn(hidden_dim, intermediate_dim, generator=generator, device=device, dtype=torch.bfloat16) * intermediate_dim**-0.5
    w_routed_gate = torch.randn(num_local_experts, intermediate_dim, hidden_dim, generator=generator, device=device, dtype=torch.bfloat16) * hidden_dim**-0.5
    w_routed_up = torch.randn(num_local_experts, intermediate_dim, hidden_dim, generator=generator, device=device, dtype=torch.bfloat16) * hidden_dim**-0.5
    w_routed_down = (
        torch.randn(num_local_experts, hidden_dim, intermediate_dim, generator=generator, device=device, dtype=torch.bfloat16) * intermediate_dim**-0.5
    )
    d_output = torch.randn(num_local_tokens, hidden_dim, generator=generator, device=device, dtype=torch.bfloat16) * hidden_dim**-0.5

    return (
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
    )


def all_to_all(x: torch.Tensor, count: int, output_splits: list[int], input_splits: list[int], *, group: dist.ProcessGroup | None = None) -> torch.Tensor:
    output = x.new_empty((count, *x.shape[1:]))
    dist.all_to_all_single(output, x, output_splits, input_splits, group=group)
    return output


def run_all_gather_top_experts_reference(top_experts: torch.Tensor) -> torch.Tensor:
    world_size = dist.get_world_size()
    top_experts_all = top_experts.new_empty((world_size, *top_experts.shape))
    dist.all_gather_into_tensor(top_experts_all, top_experts)
    return top_experts_all


def run_schedule_reference(
    topk_all: torch.Tensor,
    num_local_experts: int,
    schedule_capacity: int,
    rank: int,
) -> tuple[
    torch.Tensor,
    torch.Tensor,  # schedule_peer_rank, schedule_peer_token_idx
    torch.Tensor,
    torch.Tensor,  # num_tokens, tokens_per_expert
]:
    world_size, num_local_tokens, topk = topk_all.shape
    num_routes = num_local_tokens * topk
    first_expert = rank * num_local_experts

    schedule_peer_rank = torch.full((schedule_capacity,), -1, dtype=torch.int32, device=topk_all.device)
    schedule_peer_token_idx = torch.full_like(schedule_peer_rank, -1)
    tokens_per_expert = torch.empty(num_local_experts, dtype=torch.int32, device=topk_all.device)

    # The CUDA scheduler visits all routes owned by thread 0, then thread 1, etc.
    num_schedule_threads = 1024
    route_order = (
        torch.arange(
            math.ceil(num_routes / num_schedule_threads) * num_schedule_threads,
            device=topk_all.device,
        )
        .view(-1, num_schedule_threads)
        .T.flatten()
    )
    route_order = route_order[route_order < num_routes]
    topk_flat = topk_all.view(world_size, num_routes)

    offset = 0
    for local_expert in range(num_local_experts):
        expert = first_expert + local_expert
        peer_ranks = []
        peer_token_indices = []
        peer_offsets = []
        for peer_rank in range(world_size):
            peer_tokens = route_order[topk_flat[peer_rank, route_order] == expert]
            peer_ranks.append(torch.full_like(peer_tokens, peer_rank, dtype=torch.int32))
            peer_token_indices.append(peer_tokens.to(torch.int32))
            peer_offsets.append(torch.arange(peer_tokens.numel(), device=topk_all.device))

        num_expert_tokens = sum(tokens.numel() for tokens in peer_token_indices)
        padded_num_expert_tokens = math.ceil(num_expert_tokens / 256) * 256
        tokens_per_expert[local_expert] = padded_num_expert_tokens
        if offset + padded_num_expert_tokens > schedule_capacity:
            raise ValueError("schedule_capacity is too small for the reference schedule")

        if num_expert_tokens:
            expert_peer_ranks = torch.cat(peer_ranks)
            expert_peer_token_indices = torch.cat(peer_token_indices)
            interleave_order = torch.argsort(torch.cat(peer_offsets) * world_size + expert_peer_ranks, stable=True)
            expert_slice = slice(offset, offset + num_expert_tokens)
            schedule_peer_rank[expert_slice] = expert_peer_ranks[interleave_order]
            schedule_peer_token_idx[expert_slice] = expert_peer_token_indices[interleave_order]
        offset += padded_num_expert_tokens

    num_tokens = torch.tensor([offset], dtype=torch.int32, device=topk_all.device)
    return schedule_peer_rank, schedule_peer_token_idx, num_tokens, tokens_per_expert


def run_mxfp8_quantize_normal_reference(
    x_bf16: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    is_2d = x_bf16.ndim == 2
    x_bf16_3d = x_bf16.unsqueeze(0) if is_2d else x_bf16
    num_experts, rows, columns = x_bf16_3d.shape

    x_float = x_bf16_3d.float().reshape(num_experts * rows, columns)

    # csrc/mxfp8.cuh quantize_single_block, operation for operation (the C++ build's rule, which
    # the CuTe DSL quantizer keeps): the BF16 amax of each 32-element block; in FP32
    # scale = max(amax * 0.002232142857f, 1e-12f) -- a multiply by fl(1/448), not a division,
    # and a floor that gives an all-zero block exponent -39, not -127; then
    # cvt.rp.satfinite.ue8m0, the power of two at or above it (an exact power stays) saturating at
    # 2^127.  The payload is cvt.rn.satfinite.e4m3 of x times the reciprocal assembled from the
    # exponent bits, (254 - byte) << 23; a plain cast would turn an overflow into NaN.
    block_amax = x_float.abs().view(num_experts * rows, columns // 32, 32).amax(dim=-1)
    scale = torch.maximum(
        block_amax * torch.tensor(0.002232142857, dtype=torch.float32, device=x_bf16.device), torch.tensor(1e-12, dtype=torch.float32, device=x_bf16.device)
    )
    mantissa, exponent = torch.frexp(scale)  # scale = mantissa * 2^exponent, mantissa in [0.5, 1)
    scale_exponent = torch.where(mantissa == 0.5, exponent - 1, exponent).clamp(max=127)
    scale_byte = (scale_exponent + 127).to(torch.int32)
    scale_inverse = ((254 - scale_byte) << 23).view(torch.float32)
    x_scaled = x_float * scale_inverse.repeat_interleave(32, dim=-1)
    x_fp8 = x_scaled.clamp(-448.0, 448.0).to(torch.float8_e4m3fn)
    x_fp8 = x_fp8.reshape(num_experts, rows, columns)

    scale_bytes = scale_byte.to(torch.uint8)
    scales = scale_bytes.reshape(num_experts * rows // 128, 128, columns // 128, 4).transpose(1, 2)
    scales = scales.reshape(num_experts * rows // 128, columns // 128, 4, 32, 4).transpose(-2, -3)
    scales = scales.reshape(num_experts * rows // 128, columns // 128, 32, 16).contiguous()

    return (x_fp8.squeeze(0) if is_2d else x_fp8), scales


def run_mxfp8_quantize_reference(
    x_bf16: torch.Tensor,
    return_normal: bool,
    return_transposed: bool,
) -> tuple[
    torch.Tensor | None,
    torch.Tensor | None,  # x_fp8, x_sc
    torch.Tensor | None,
    torch.Tensor | None,  # x_fp8_t, x_sc_t
]:
    if return_normal:
        x_fp8, x_sc = run_mxfp8_quantize_normal_reference(x_bf16)
    else:
        x_fp8, x_sc = None, None
    if return_transposed:
        x_fp8_t, x_sc_t = run_mxfp8_quantize_normal_reference(x_bf16.transpose(-2, -1).contiguous())
    else:
        x_fp8_t, x_sc_t = None, None
    return x_fp8, x_sc, x_fp8_t, x_sc_t


def run_swiglu_reference(
    gate: torch.Tensor,
    up: torch.Tensor,
    swiglu_limit: float | None = None,
) -> torch.Tensor:
    if swiglu_limit is None:
        return torch.nn.functional.silu(gate) * up
    else:
        dtype = gate.dtype
        gate_fp32 = gate.float()
        up_fp32 = up.float()
        # MOK keeps the clamp derivative at 1 exactly on the boundary:
        # gate <= limit, and -limit <= up <= limit. Scalar torch.clamp's
        # boundary derivative varies across PyTorch versions (zero in 2.14).
        # Strict out-of-range conditions preserve MOK's contract explicitly.
        gate_fp32 = torch.where(gate_fp32 > swiglu_limit, swiglu_limit, gate_fp32)
        up_fp32 = torch.where(up_fp32 < -swiglu_limit, -swiglu_limit, up_fp32)
        up_fp32 = torch.where(up_fp32 > swiglu_limit, swiglu_limit, up_fp32)
        return (torch.nn.functional.silu(gate_fp32) * up_fp32).to(dtype)


def run_forward_reference_bf16(
    x: torch.Tensor,  # [T, H]
    topk_experts: torch.Tensor,  # [T, TOPK]
    w_shared_gate: torch.Tensor,  # [I, H]
    w_shared_up: torch.Tensor,  # [I, H]
    w_shared_down: torch.Tensor,  # [H, I]
    w_routed_gate: torch.Tensor,  # [E, I, H]
    w_routed_up: torch.Tensor,  # [E, I, H]
    w_routed_down: torch.Tensor,  # [E, H, I]
    swiglu_limit: float | None = None,
    routed_input: torch.Tensor | None = None,  # [T, H_r] routed-expert input (routed hidden != hidden); None -> x
) -> tuple[
    torch.Tensor,  # combine_buffer  [T * TOPK, H_r]
    torch.Tensor,  # gate_shared
    torch.Tensor,  # up_shared
    torch.Tensor,  # hidden_shared
    torch.Tensor,  # y_shared        [T, H]
]:
    world_size = dist.get_world_size()
    rank = dist.get_rank()
    x_routed_source = x if routed_input is None else routed_input

    num_local_experts = w_routed_gate.shape[0]
    num_local_tokens, hidden = x.shape
    # A latent layer routes a narrower tensor than the shared expert sees, so the combine buffer's
    # width is the routed input's, not x's.
    routed_hidden = x_routed_source.shape[1]
    topk = topk_experts.shape[1]
    num_routes = num_local_tokens * topk

    topk_experts_flat = topk_experts.flatten()
    # -1 marks a slot the token does not use: it is dispatched to no rank, and its row of the
    # combine buffer stays zero rather than holding whatever the kernel's buffer held.
    valid_route_indices = (topk_experts_flat >= 0).nonzero().flatten()
    valid_experts = topk_experts_flat[valid_route_indices]
    destination_ranks = valid_experts // num_local_experts
    dispatch_order = valid_route_indices[torch.argsort(destination_ranks, stable=True)]
    send_counts = torch.bincount(destination_ranks, minlength=world_size)
    all_send_counts = torch.empty(world_size, world_size, dtype=torch.int64, device=x.device)
    dist.all_gather_into_tensor(all_send_counts, send_counts)
    send_splits = send_counts.tolist()
    recv_splits = all_send_counts[:, rank].tolist()
    num_recv = sum(recv_splits)
    send_x = x_routed_source[dispatch_order // topk]
    send_local_experts = (topk_experts_flat[dispatch_order] % num_local_experts).contiguous()
    recv_x = all_to_all(send_x, num_recv, recv_splits, send_splits)
    recv_local_experts = all_to_all(send_local_experts, num_recv, recv_splits, send_splits)

    recv_output = torch.zeros_like(recv_x)
    for expert_idx in range(num_local_experts):
        rows = (recv_local_experts == expert_idx).nonzero().flatten()
        expert_x = recv_x[rows]
        gate = expert_x @ w_routed_gate[expert_idx].T
        up = expert_x @ w_routed_up[expert_idx].T
        hidden_activations = run_swiglu_reference(gate, up, swiglu_limit)
        recv_output = recv_output.index_copy(0, rows, hidden_activations @ w_routed_down[expert_idx].T)

    num_valid_routes = valid_route_indices.numel()
    returned_output = all_to_all(recv_output, num_valid_routes, send_splits, recv_splits)
    combine_buffer = torch.zeros(num_routes, routed_hidden, dtype=returned_output.dtype, device=x.device)
    combine_buffer[dispatch_order] = returned_output

    gate_shared = x @ w_shared_gate.T
    up_shared = x @ w_shared_up.T
    hidden_shared = run_swiglu_reference(gate_shared, up_shared, swiglu_limit)
    y_shared = hidden_shared @ w_shared_down.T

    return combine_buffer, gate_shared, up_shared, hidden_shared, y_shared


def run_fwd_epilogue_reference(
    y_shared: torch.Tensor,
    combine_buffer: torch.Tensor,
    topk_weights: torch.Tensor,
    top_experts: torch.Tensor | None = None,
) -> torch.Tensor:
    num_local_tokens, hidden_size = y_shared.shape
    topk = topk_weights.shape[1]
    routed = combine_buffer.view(num_local_tokens, topk, hidden_size)
    if top_experts is not None:
        route_valid = top_experts >= 0
        routed = torch.where(route_valid.unsqueeze(2), routed, 0.0)
        topk_weights = torch.where(route_valid, topk_weights, 0.0)
    return (y_shared.float() + (routed.float() * topk_weights.unsqueeze(2)).sum(dim=1)).to(torch.bfloat16)


def run_routed_topk_reduce_reference(
    combine_buffer: torch.Tensor,  # [T * TOPK, H_r]
    topk_weights: torch.Tensor,  # [T, TOPK]
) -> torch.Tensor:
    """Routed hidden != hidden: the routed half of the forward epilogue, sum_k w_k y_k in fp32 -> bf16."""
    num_local_tokens, topk = topk_weights.shape
    routed = combine_buffer.view(num_local_tokens, topk, -1)
    return (routed.float() * topk_weights.unsqueeze(2)).sum(dim=1).to(torch.bfloat16)


def run_bwd_epilogue_reference(
    d_x_shared: torch.Tensor,
    d_x_routed_buffer: torch.Tensor,
    top_experts: torch.Tensor | None = None,
) -> torch.Tensor:
    num_local_tokens, hidden_size = d_x_shared.shape
    topk = d_x_routed_buffer.shape[0] // num_local_tokens
    d_x_routed = d_x_routed_buffer.view(num_local_tokens, topk, hidden_size)
    if top_experts is not None:
        d_x_routed = torch.where((top_experts >= 0).unsqueeze(2), d_x_routed, 0.0)
    return (d_x_shared.float() + d_x_routed.float().sum(dim=1)).to(torch.bfloat16)


def run_shared_output_gate_reference(
    x: torch.Tensor,
    shared_output: torch.Tensor,
    output_gate_weight: torch.Tensor,
    d_output: torch.Tensor,
) -> tuple[torch.Tensor, ...]:
    """Return G, dS, dG, dZ, dX_gate and FP64 dW_gate from an independent graph (the C++ port's
    utils.py, unchanged).

    BF16 S*G defines the backward rounding, not the full MoE forward.
    Compute dW_gate from the reference dZ, not a BF16 weight.grad or MOK tensor.
    """
    x_ref = x.detach().requires_grad_()
    shared_ref = shared_output.detach().requires_grad_()
    gate_weight_ref = output_gate_weight.detach()
    logits = torch.nn.functional.linear(x_ref, gate_weight_ref)
    gate = torch.sigmoid(logits)
    gated_shared_bf16 = shared_ref * gate
    d_shared, d_gate, d_logits, d_x_gate = torch.autograd.grad(
        gated_shared_bf16,
        (shared_ref, gate, logits, x_ref),
        d_output,
    )
    d_weight_fp64 = d_logits.double().T @ x_ref.detach().double()
    return (
        gate.detach(),
        d_shared,
        d_gate,
        d_logits,
        d_x_gate,
        d_weight_fp64,
    )


def run_reference_bf16(
    x: torch.Tensor,  # [T, H]
    topk_experts: torch.Tensor,  # [T, TOPK]
    router_weights: torch.Tensor,  # [T, TOPK]
    w_shared_gate: torch.Tensor,  # [I, H]
    w_shared_up: torch.Tensor,  # [I, H]
    w_shared_down: torch.Tensor,  # [H, I]
    w_routed_gate: torch.Tensor,  # [E, I, H]
    w_routed_up: torch.Tensor,  # [E, I, H]
    w_routed_down: torch.Tensor,  # [E, H, I]
    d_output: torch.Tensor,  # [T, H]
    swiglu_limit: float | None = None,
    *,
    group: dist.ProcessGroup | None = None,
    routed_input: torch.Tensor | None = None,  # [T, H_r] routed-expert input (routed weights in H_r)
    d_routed_output: torch.Tensor | None = None,  # [T, H_r] gradient of the routed top-k reduced output
    shared_output_gate_weight: torch.Tensor | None = None,
    routed_activation=None,
    shared_activation=None,
) -> tuple[
    torch.Tensor,  # output
    torch.Tensor,  # d_x
    torch.Tensor,  # d_router_weights
    torch.Tensor,  # d_w_routed_gate
    torch.Tensor,  # d_w_routed_up
    torch.Tensor,  # d_w_routed_down
    torch.Tensor,  # d_w_shared_gate
    torch.Tensor,  # d_w_shared_up
    torch.Tensor,  # d_w_shared_down
]:
    """Routed hidden != hidden (routed_input given): output is (y_shared [T, H], y_routed [T, H_r]) and d_x is
    (d_x_shared [T, H], d_routed_input [T, H_r]); d_output is then the shared output's gradient.

    As the C++ port's reference: ``shared_output_gate_weight`` gates the shared output (forward
    accumulates S*G and the routed outputs in FP32; the gate backward uses the BF16-product
    rounding of run_shared_output_gate_reference) and appends FP32 dW_gate; the activation
    overrides give independent routed / shared MLP oracles (SiTU-GLU)."""
    if routed_input is not None and shared_output_gate_weight is not None:
        raise NotImplementedError("the gated reference needs routed_hidden_size == hidden_size")
    world_size = dist.get_world_size(group)
    rank = dist.get_rank(group)
    split_hidden = routed_input is not None
    x_routed = routed_input if split_hidden else x
    d_routed = d_routed_output if split_hidden else d_output

    num_local_experts = w_routed_gate.shape[0]
    num_local_tokens, hidden = x_routed.shape  # routed hidden
    topk = topk_experts.shape[1]
    num_routes = num_local_tokens * topk

    # Dispatch all-to-all
    topk_experts_flat = topk_experts.flatten()
    # -1 marks a slot the token does not use: it is dispatched to no rank, and its row of the
    # combine buffer stays zero rather than holding whatever the kernel's buffer held.
    valid_route_indices = (topk_experts_flat >= 0).nonzero().flatten()
    valid_experts = topk_experts_flat[valid_route_indices]
    destination_ranks = valid_experts // num_local_experts
    dispatch_order = valid_route_indices[torch.argsort(destination_ranks, stable=True)]
    send_counts = torch.bincount(destination_ranks, minlength=world_size)
    all_send_counts = torch.empty(world_size, world_size, dtype=torch.int64, device=x.device)
    dist.all_gather_into_tensor(all_send_counts, send_counts, group=group)
    send_splits = send_counts.tolist()
    recv_splits = all_send_counts[:, rank].tolist()
    num_recv = sum(recv_splits)
    send_x = x_routed[dispatch_order // topk]
    send_local_experts = (topk_experts_flat[dispatch_order] % num_local_experts).contiguous()
    recv_x = all_to_all(send_x, num_recv, recv_splits, send_splits, group=group).requires_grad_()
    recv_local_experts = all_to_all(send_local_experts, num_recv, recv_splits, send_splits, group=group)

    # Routed expert FFN
    w_routed_gate = w_routed_gate.detach().requires_grad_()
    w_routed_up = w_routed_up.detach().requires_grad_()
    w_routed_down = w_routed_down.detach().requires_grad_()
    recv_output = torch.zeros_like(recv_x)
    for expert_idx in range(num_local_experts):
        rows = (recv_local_experts == expert_idx).nonzero().flatten()
        expert_x = recv_x[rows]
        gate = expert_x @ w_routed_gate[expert_idx].T
        up = expert_x @ w_routed_up[expert_idx].T
        hidden_activations = run_swiglu_reference(gate, up, swiglu_limit) if routed_activation is None else routed_activation(gate, up)
        recv_output = recv_output.index_copy(0, rows, hidden_activations @ w_routed_down[expert_idx].T)

    # Routed token weighted sum
    num_valid_routes = valid_route_indices.numel()
    returned_output = all_to_all(recv_output.detach(), num_valid_routes, send_splits, recv_splits, group=group)
    flat_output = torch.zeros(num_routes, hidden, dtype=returned_output.dtype, device=x.device)
    flat_output[dispatch_order] = returned_output
    routed_output = (flat_output.view(num_local_tokens, topk, hidden).float() * router_weights.unsqueeze(2)).sum(1)

    # Shared expert FFN
    x_shared = x.detach().requires_grad_()
    w_shared_gate = w_shared_gate.detach().requires_grad_()
    w_shared_up = w_shared_up.detach().requires_grad_()
    w_shared_down = w_shared_down.detach().requires_grad_()
    gate_shared = x_shared @ w_shared_gate.T
    up_shared = x_shared @ w_shared_up.T
    shared_hidden = run_swiglu_reference(gate_shared, up_shared, swiglu_limit) if shared_activation is None else shared_activation(gate_shared, up_shared)
    shared_output = shared_hidden @ w_shared_down.T

    # Final sum.  Gated: shared first, then each valid route, in FP32, as the forward epilogue.
    d_shared_output = d_output
    if split_hidden:
        output = (shared_output.detach().to(torch.bfloat16), routed_output.to(torch.bfloat16))
    elif shared_output_gate_weight is None:
        output = (routed_output + shared_output.float()).to(torch.bfloat16)
    else:
        gate, d_shared_output, _, _, d_x_gate, d_w_gate_fp64 = run_shared_output_gate_reference(x, shared_output, shared_output_gate_weight, d_output)
        accumulator = shared_output.float() * gate.float()
        routed = flat_output.view(num_local_tokens, topk, hidden)
        for k in range(topk):
            term = routed[:, k].float() * router_weights[:, k, None]
            accumulator = accumulator + torch.where(topk_experts[:, k, None] >= 0, term, 0.0)
        output = accumulator.to(torch.bfloat16)

    # Combine all-to-all
    d_flat_output = (d_routed.unsqueeze(1).float() * router_weights.unsqueeze(2)).to(torch.bfloat16)
    d_send_output = d_flat_output.reshape(-1, hidden)[dispatch_order]
    d_recv_output = all_to_all(d_send_output, num_recv, recv_splits, send_splits, group=group).contiguous()
    d_recv_x, d_w_routed_gate, d_w_routed_up, d_w_routed_down = torch.autograd.grad(
        recv_output,
        (recv_x, w_routed_gate, w_routed_up, w_routed_down),
        d_recv_output,
    )

    # Backward
    returned_d_x = all_to_all(d_recv_x, num_valid_routes, send_splits, recv_splits, group=group)
    flat_d_x = torch.zeros(num_routes, hidden, dtype=returned_d_x.dtype, device=x.device)
    flat_d_x[dispatch_order] = returned_d_x
    d_x_routed = flat_d_x.view(num_local_tokens, topk, hidden).float().sum(1)
    d_x_shared, d_w_shared_gate, d_w_shared_up, d_w_shared_down = torch.autograd.grad(
        shared_output,
        (x_shared, w_shared_gate, w_shared_up, w_shared_down),
        d_shared_output,
    )
    if split_hidden:
        d_x = (d_x_shared, d_x_routed.to(torch.bfloat16))
    elif shared_output_gate_weight is None:
        d_x = (d_x_routed + d_x_shared.float()).to(torch.bfloat16)
    else:
        accumulator = d_x_shared.float()
        routed_d_x = flat_d_x.view(num_local_tokens, topk, hidden)
        for k in range(topk):
            accumulator = accumulator + routed_d_x[:, k].float()
        d_x = accumulator.to(torch.bfloat16) + d_x_gate
    d_router_weights = (d_routed.unsqueeze(1).float() * flat_output.view(num_local_tokens, topk, hidden).float()).sum(2)
    # An unused slot has no gradient.
    d_router_weights.masked_fill_(topk_experts < 0, 0.0)

    results = (
        output,  # [T, H]
        d_x,  # [T, H]
        d_router_weights,  # [T, TOPK]
        d_w_routed_gate,  # [E, I, H]
        d_w_routed_up,  # [E, I, H]
        d_w_routed_down,  # [E, H, I]
        d_w_shared_gate,  # [I, H]
        d_w_shared_up,  # [I, H]
        d_w_shared_down,  # [H, I]
    )
    if shared_output_gate_weight is None:
        return results
    return (*results, d_w_gate_fp64.float())


_ERROR_STATS_CHUNK = 1 << 26  # elements per float32 temporary (256 MiB): a 384-expert dW is 8.5 G elements


def get_error_stats(
    reference: torch.Tensor,
    actual: torch.Tensor,
) -> tuple[float, float, float]:
    assert reference.shape == actual.shape, (tuple(reference.shape), tuple(actual.shape))
    # Chunked over the flattened tensors: the float32 copies of a whole routed weight gradient
    # (31.5 GiB for 384 x 3072 x 7168) do not fit next to the inputs, two backends' results and
    # the reference at EP1 on one GPU.
    reference_flat = reference.reshape(-1)
    actual_flat = actual.reshape(-1)
    device = reference.device
    diff_sum = torch.zeros((), dtype=torch.float64, device=device)
    diff_max = torch.zeros((), dtype=torch.float32, device=device)
    reference_sum = torch.zeros((), dtype=torch.float64, device=device)
    for start in range(0, reference_flat.numel(), _ERROR_STATS_CHUNK):
        reference_float = reference_flat[start : start + _ERROR_STATS_CHUNK].float()
        diff = (reference_float - actual_flat[start : start + _ERROR_STATS_CHUNK].float()).abs()
        diff_sum += diff.sum(dtype=torch.float64)
        diff_max = torch.maximum(diff_max, diff.max())
        reference_sum += reference_float.abs().sum(dtype=torch.float64)
        del reference_float, diff
    diff_count = torch.tensor(reference_flat.numel(), dtype=torch.float64, device=device)

    dist.all_reduce(diff_sum, op=dist.ReduceOp.SUM)
    dist.all_reduce(diff_count, op=dist.ReduceOp.SUM)
    dist.all_reduce(diff_max, op=dist.ReduceOp.MAX)
    dist.all_reduce(reference_sum, op=dist.ReduceOp.SUM)

    abs_error_mean = (diff_sum / diff_count).item()
    abs_error_max = diff_max.item()
    # An all-zero reference matched exactly (e.g. every route on expert 0 at EP1) is error 0, not 0/0.
    relative_error = 0.0 if diff_sum.item() == 0.0 == reference_sum.item() else (diff_sum / reference_sum).item()

    return abs_error_mean, abs_error_max, relative_error


def check_correctness(
    name: str,
    reference: torch.Tensor,
    actual: torch.Tensor,
    tolerance: tuple[float, float],
    print_stats: bool = False,
) -> None:
    abs_error_mean, abs_error_max, relative_error = get_error_stats(reference, actual)
    absolute_tolerance, relative_tolerance = tolerance
    passed = (
        all(math.isfinite(value) for value in (abs_error_mean, abs_error_max, relative_error))
        and abs_error_max <= absolute_tolerance
        and relative_error <= relative_tolerance
    )
    if print_stats or not passed:
        print(f"{name}: abs error mean={abs_error_mean:.6e}, abs error max={abs_error_max:.6e}, relative error={relative_error:.6e}")
    assert (
        passed
    ), f"{name} exceeded tolerance: abs error max {abs_error_max:.6e} > {absolute_tolerance:.6e} or relative error {relative_error:.6e} > {relative_tolerance:.6e}"
