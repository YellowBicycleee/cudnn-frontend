# Copyright 2026 Cursor Research
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Derived from Mixture-of-Kittens (https://github.com/cursor/mixture-of-kittens, commit 22fc95ae); modified by NVIDIA.

import itertools

import torch
import torch.distributed as dist

from cudnn.moe_dispatch_mlp_combine import functional
from cudnn.moe_dispatch_mlp_combine.functional import get_workspace
from cudnn.moe_dispatch_mlp_combine.ops import mxfp8_quantize

from .utils import (
    ungated,
    capacity_multiplier,
    check_correctness,
    generate_inputs,
    mok_params,
    run_reference_bf16,
    shapes,
)

BF16_TOLERANCE = (0.5, 0.01)
MXFP8_TOLERANCE = (1.0, 0.1)
RESULT_NAMES = (
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


def test_e2e_bf16(context: tuple[int, int, torch.device]) -> None:
    rank, world_size, device = context
    for shape, params in itertools.product(shapes(world_size), mok_params()):
        shape_name, num_experts, hidden_dim, intermediate_dim, topk, num_local_tokens = shape
        params_name, fwd_num_comm_sms, bwd_num_comm_sms, minibatch_size, macrobatch_size = params
        assert num_experts % world_size == 0
        num_local_experts = num_experts // world_size
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

        reference_results = run_reference_bf16(*inputs)
        config = functional.CuteMoKConfig(
            fwd_num_comm_sms=fwd_num_comm_sms,
            bwd_num_comm_sms=bwd_num_comm_sms,
            minibatch_size=minibatch_size,
            macrobatch_size=macrobatch_size,
            schedule_capacity_multiplier=capacity_multiplier(world_size, num_local_tokens, topk, num_local_experts),  # must be lower for production
        )
        workspace = get_workspace(
            config,
            dist.group.WORLD,
            device=device,
            num_local_tokens=num_local_tokens,
            hidden_size=hidden_dim,
            topk=topk,
        )
        schedule = functional.build_schedule(
            workspace,
            config,
            topk_experts,
            num_local_experts=num_local_experts,
        )

        bf16_output, forward_context = functional.forward(
            config,
            workspace,
            schedule,
            x,
            router_weights,
            w_shared_gate,
            w_shared_up,
            w_shared_down,
            w_routed_gate,
            w_routed_up,
            w_routed_down,
        )
        bf16_gradients = ungated(
            functional.backward(
                config,
                workspace,
                schedule,
                forward_context,
                d_output,
                x,
                router_weights,
                w_shared_gate,
                w_shared_up,
                w_shared_down,
                w_routed_gate,
                w_routed_up,
                w_routed_down,
            )
        )
        bf16_results = (bf16_output, *bf16_gradients)

        for name, reference, actual in zip(RESULT_NAMES, reference_results, bf16_results, strict=True):
            check_correctness(
                f"{shape_name}/{params_name}/{name}",
                reference,
                actual,
                BF16_TOLERANCE,
                print_stats=rank == 0,
            )
        # Free this case before the next generate_inputs: at EP1 one case of a 384-expert shape holds
        # about 200 GiB (inputs, quantized weights, results and the reference) on the single GPU.
        del inputs, reference, actual, forward_context, reference_results, bf16_results, bf16_gradients
        del bf16_output, schedule, x, topk_experts, router_weights, w_shared_gate, w_shared_up
        del w_shared_down, w_routed_gate, w_routed_up, w_routed_down, d_output
        torch.cuda.empty_cache()


def test_e2e_mxfp8(context: tuple[int, int, torch.device]) -> None:
    rank, world_size, device = context
    for shape, params in itertools.product(shapes(world_size), mok_params()):
        shape_name, num_experts, hidden_dim, intermediate_dim, topk, num_local_tokens = shape
        params_name, fwd_num_comm_sms, bwd_num_comm_sms, minibatch_size, macrobatch_size = params
        assert num_experts % world_size == 0
        num_local_experts = num_experts // world_size
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

        reference_results = run_reference_bf16(*inputs)
        config = functional.CuteMoKConfig(
            fwd_num_comm_sms=fwd_num_comm_sms,
            bwd_num_comm_sms=bwd_num_comm_sms,
            minibatch_size=minibatch_size,
            macrobatch_size=macrobatch_size,
            schedule_capacity_multiplier=capacity_multiplier(world_size, num_local_tokens, topk, num_local_experts),  # must be lower for production
        )
        workspace = get_workspace(
            config,
            dist.group.WORLD,
            device=device,
            num_local_tokens=num_local_tokens,
            hidden_size=hidden_dim,
            topk=topk,
        )
        schedule = functional.build_schedule(
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

        mxfp8_output, forward_context = functional.forward(
            config,
            workspace,
            schedule,
            x,
            router_weights,
            w_shared_gate,
            w_shared_up,
            w_shared_down,
            (w_routed_gate_fp8, w_routed_gate_sc),
            (w_routed_up_fp8, w_routed_up_sc),
            (w_routed_down_fp8, w_routed_down_sc),
        )
        mxfp8_gradients = ungated(
            functional.backward(
                config,
                workspace,
                schedule,
                forward_context,
                d_output,
                x,
                router_weights,
                w_shared_gate,
                w_shared_up,
                w_shared_down,
                (
                    w_routed_gate_fp8,
                    w_routed_gate_sc,
                    w_routed_gate_t_fp8,
                    w_routed_gate_t_sc,
                ),
                (
                    w_routed_up_fp8,
                    w_routed_up_sc,
                    w_routed_up_t_fp8,
                    w_routed_up_t_sc,
                ),
                (
                    w_routed_down_t_fp8,
                    w_routed_down_t_sc,
                ),
            )
        )
        mxfp8_results = (mxfp8_output, *mxfp8_gradients)

        for name, reference, actual in zip(RESULT_NAMES, reference_results, mxfp8_results, strict=True):
            check_correctness(
                f"{shape_name}/{params_name}/{name}",
                reference,
                actual,
                MXFP8_TOLERANCE,
                print_stats=rank == 0,
            )
        # Free this case before the next generate_inputs: at EP1 one case of a 384-expert shape holds
        # about 200 GiB (inputs, quantized weights, results and the reference) on the single GPU.
        del inputs, reference, actual, forward_context, reference_results, mxfp8_results, mxfp8_gradients
        del mxfp8_output, schedule, w_routed_gate_fp8, w_routed_up_fp8, w_routed_down_fp8
        del w_routed_gate_t_fp8, w_routed_up_t_fp8, w_routed_down_t_fp8, w_routed_gate_sc, w_routed_up_sc
        del w_routed_down_sc, w_routed_gate_t_sc, w_routed_up_t_sc, w_routed_down_t_sc, x, topk_experts
        del router_weights, w_shared_gate, w_shared_up, w_shared_down, w_routed_gate, w_routed_up
        del w_routed_down, d_output
        torch.cuda.empty_cache()
