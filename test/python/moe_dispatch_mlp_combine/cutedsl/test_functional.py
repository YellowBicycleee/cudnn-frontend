# Copyright 2026 Cursor Research
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Derived from Mixture-of-Kittens (https://github.com/cursor/mixture-of-kittens, commit 22fc95ae); modified by NVIDIA.

import itertools

import pytest
import torch
import torch.distributed as dist

from cudnn.moe_dispatch_mlp_combine import functional
from cudnn.moe_dispatch_mlp_combine.functional import clear_workspace_cache, get_workspace
from cudnn.moe_dispatch_mlp_combine.ops import mxfp8_quantize

from .utils import (
    ungated,
    BF16_TOLERANCE,
    MXFP8_TOLERANCE,
    MXFP8_UPSTREAM_ABS_MAX,
    capacity_multiplier,
    check_correctness,
    generate_inputs,
    generate_topk_experts,
    mok_params,
    mxfp8_tolerance,
    swiglu_params,
    run_all_gather_top_experts_reference,
    run_forward_reference_bf16,
    run_fwd_epilogue_reference,
    run_reference_bf16,
    run_schedule_reference,
    shapes,
)

EXACT_TOLERANCE = (0.0, 0.0)
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

        output, forward_context = functional.forward(
            config,
            workspace,
            mok_schedule,
            x,
            router_weights,
            w_shared_gate,
            w_shared_up,
            w_shared_down,
            w_routed_gate,
            w_routed_up,
            w_routed_down,
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
                BF16_TOLERANCE,
                print_stats=rank == 0,
            )
        # Free this case before the next generate_inputs: at EP1 one case of a 384-expert shape holds
        # about 200 GiB (inputs, quantized weights, results and the reference) on the single GPU.
        del inputs, reference, actual, forward_context, mok_schedule, output, x, topk_experts
        del router_weights, w_shared_gate, w_shared_up, w_shared_down, w_routed_gate, w_routed_up
        del w_routed_down
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


def test_mxfp8_tolerance_is_widened_only_where_upstream_is_outside_it() -> None:
    """``mxfp8_tolerance`` returns upstream's abs max for exactly the recorded world size, case, SwiGLU mode and
    result, and ``MXFP8_TOLERANCE`` for everything else."""
    recorded = (8, "All minimums", "Clamped SwiGLU", "d_w_routed_gate")
    assert mxfp8_tolerance(*recorded) == (1.0859375, MXFP8_TOLERANCE[1])
    for position, other in enumerate((4, "Minimum top-k", "Unclamped SwiGLU", "d_w_routed_up")):
        neighbour = (*recorded[:position], other, *recorded[position + 1 :])
        assert mxfp8_tolerance(*neighbour) == MXFP8_TOLERANCE, neighbour
    assert all(bound > MXFP8_TOLERANCE[0] for bound in MXFP8_UPSTREAM_ABS_MAX.values())


def test_backward_mxfp8(context: tuple[int, int, torch.device]) -> None:
    """Against the BF16 reference within ``MXFP8_TOLERANCE``; where upstream Mixture-of-Kittens is itself outside
    the absolute bound, that result's bound is upstream's abs max (``mxfp8_tolerance``, validate/README.md)."""
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
        _, forward_context = functional.forward(
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

        actual = ungated(
            functional.backward(
                config,
                workspace,
                mok_schedule,
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
                swiglu_limit,
            )
        )
        reference = run_reference_bf16(*inputs, swiglu_limit)[1:]
        for name, expected, result in zip(BACKWARD_RESULT_NAMES, reference, actual, strict=True):
            check_correctness(
                f"{shape_name}/{mok_param_name}/{swiglu_param_name}/{name}",
                expected,
                result,
                mxfp8_tolerance(world_size, shape_name, swiglu_param_name, name),
                print_stats=rank == 0,
            )
        # Free this case before the next generate_inputs: at EP1 one case of a 384-expert shape holds
        # about 200 GiB (inputs, quantized weights, results and the reference) on the single GPU.
        del inputs, reference, actual, forward_context, mok_schedule, w_routed_gate_fp8, w_routed_up_fp8
        del w_routed_down_fp8, w_routed_gate_t_fp8, w_routed_up_t_fp8, w_routed_down_t_fp8
        del w_routed_gate_sc, w_routed_up_sc, w_routed_down_sc, w_routed_gate_t_sc, w_routed_up_t_sc
        del w_routed_down_t_sc, x, topk_experts, router_weights, w_shared_gate, w_shared_up, w_shared_down
        del w_routed_gate, w_routed_up, w_routed_down, d_output
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
    _, forward_context = functional.forward(
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
    )
    valid_kwargs = {
        "config": config,
        "workspace": workspace,
        "schedule": mok_schedule,
        "forward_context": forward_context,
        "grad_output": d_output,
        "x": x,
        "router_weights": router_weights,
        "shared_gate_weights": w_shared_gate,
        "shared_up_weights": w_shared_up,
        "shared_down_weights": w_shared_down,
        "routed_gate_weights": (
            w_routed_gate_fp8,
            w_routed_gate_sc,
            w_routed_gate_t_fp8,
            w_routed_gate_t_sc,
        ),
        "routed_up_weights": (
            w_routed_up_fp8,
            w_routed_up_sc,
            w_routed_up_t_fp8,
            w_routed_up_t_sc,
        ),
        "routed_down_weights": (
            w_routed_down_t_fp8,
            w_routed_down_t_sc,
        ),
    }

    actual = ungated(functional.backward(**valid_kwargs))
    reference = run_reference_bf16(*inputs)[1:]
    for name, expected, result in zip(BACKWARD_RESULT_NAMES, reference, actual, strict=True):
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
        (
            "gradient output dtype",
            {"grad_output": d_output.float()},
            ValueError,
        ),
        (
            "gradient output shape",
            {"grad_output": d_output[:-256]},
            ValueError,
        ),
        ("input dtype", {"x": x.float()}, ValueError),
        (
            "router weight dtype",
            {"router_weights": router_weights.to(torch.bfloat16)},
            ValueError,
        ),
        (
            "schedule capacity",
            {"schedule": invalid_schedule},
            ValueError,
        ),
    ):
        try:
            ungated(functional.backward(**(valid_kwargs | overrides)))
        except expected_exception:
            pass
        else:
            raise AssertionError(f"{failure_name} should have failed")


def test_backward_bf16_paired_replay_smem_lifetime(
    context: tuple[int, int, torch.device],
) -> None:
    """Exercise generic/paired input storage transitions at a short split segment."""
    rank, world_size, device = context
    if world_size != 8:
        pytest.skip("paired replay SMEM lifetime regression requires EP8")

    # The session fixture may retain workspaces from preceding tests.
    dist.barrier()
    clear_workspace_cache()
    inputs = generate_inputs(rank, device, 512, 64, 10, 102400, 4096, 1024)
    route_counts = torch.stack([(inputs[1] == expert).sum() for expert in (128, 129, 130)])
    dist.all_reduce(route_counts, op=dist.ReduceOp.SUM)
    padded = [((int(value) + 255) // 256) * 256 for value in route_counts.tolist()]
    expected_begin, expected_end = sum(padded[:2]), sum(padded)
    with torch.enable_grad():
        first_reference = run_reference_bf16(*inputs, None)[1:]

    def check_complete_call(label, current_inputs, reference):
        with torch.no_grad():
            _, forward_context = functional.forward(
                fwd_config,
                workspace,
                schedule,
                current_inputs[0],
                current_inputs[2],
                *current_inputs[3:9],
            )
            actual = ungated(
                functional.backward(
                    bwd_config,
                    workspace,
                    schedule,
                    forward_context,
                    current_inputs[9],
                    current_inputs[0],
                    current_inputs[2],
                    *current_inputs[3:9],
                )
            )
            actual = tuple(tensor.detach().clone() for tensor in actual)
            torch.cuda.synchronize(device)
        for name, expected, result in zip(BACKWARD_RESULT_NAMES, reference, actual, strict=True):
            check_correctness(
                f"{label}/{name}",
                expected,
                result,
                BF16_TOLERANCE,
                print_stats=rank == 0,
            )

    for macro, mini, comm in (
        (8192, 4096, 56),
        (16384, 8192, 40),
        (32768, 16384, 42),
        (32768, 16384, 56),
    ):
        label = f"paired replay/macro{macro}/mini{mini}/comm{comm}"
        representative = (macro, mini, comm) == (32768, 16384, 56)
        if representative:
            changed_inputs = list(inputs)
            changed_inputs[9] = inputs[9] * -0.5
            with torch.enable_grad():
                changed_reference = run_reference_bf16(*changed_inputs, None)[1:]
        fwd_config = functional.CuteMoKConfig(
            fwd_num_comm_sms=40,
            bwd_num_comm_sms=comm,
            minibatch_size=4096,
            macrobatch_size=macro,
        )
        bwd_config = functional.CuteMoKConfig(
            fwd_num_comm_sms=40,
            bwd_num_comm_sms=comm,
            minibatch_size=mini,
            macrobatch_size=macro,
        )
        # The references leave ~150 GiB in PyTorch's cache, and the workspace's symmetric memory is
        # allocated from the driver: on a 184 GiB GB200 the allocation fails unless the cache is released.
        torch.cuda.empty_cache()
        workspace = get_workspace(
            fwd_config,
            dist.group.WORLD,
            device=device,
            num_local_tokens=102400,
            hidden_size=4096,
            topk=10,
        )
        schedule = functional.build_schedule(
            workspace,
            fwd_config,
            inputs[1],
            num_local_experts=64,
        )
        # Derive the split from routing, rather than manufacturing a short tile.
        boundary_ok = True
        if rank == 2:
            actual_begin = int(schedule.tokens_per_expert[:2].sum().item())
            actual_end = int(schedule.tokens_per_expert[:3].sum().item())
            first_segment_rows = min(actual_end, (actual_begin // macro + 1) * macro) - actual_begin
            boundary_ok = (actual_begin, actual_end) == (expected_begin, expected_end) and first_segment_rows == 256
        flag = torch.tensor([int(boundary_ok)], dtype=torch.int32, device=device)
        dist.all_reduce(flag, op=dist.ReduceOp.MIN)
        assert bool(flag.item()), "routing must exercise the short first macro segment"

        check_complete_call(f"{label}/first", inputs, first_reference)
        if representative:
            check_complete_call(f"{label}/changed", changed_inputs, changed_reference)
            del changed_inputs, changed_reference
            # Check CUDA API reuse after a version-only schedule mutation.
            before_version = schedule.peer_token_idx._version
            schedule.peer_token_idx.add_(0)
            assert schedule.peer_token_idx._version > before_version
            check_complete_call(f"{label}/version_changed", inputs, first_reference)
        dist.barrier()
        clear_workspace_cache()
        del workspace, schedule


def test_backward_bf16(context: tuple[int, int, torch.device]) -> None:
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
            d_output,
        ) = inputs
        mok_schedule = functional.build_schedule(
            workspace,
            config,
            topk_experts,
            num_local_experts=num_local_experts,
        )
        _, forward_context = functional.forward(
            config,
            workspace,
            mok_schedule,
            x,
            router_weights,
            w_shared_gate,
            w_shared_up,
            w_shared_down,
            w_routed_gate,
            w_routed_up,
            w_routed_down,
            swiglu_limit,
        )

        actual = ungated(
            functional.backward(
                config,
                workspace,
                mok_schedule,
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
                swiglu_limit,
            )
        )
        reference = run_reference_bf16(*inputs, swiglu_limit)[1:]
        for name, expected, result in zip(BACKWARD_RESULT_NAMES, reference, actual, strict=True):
            check_correctness(
                f"{shape_name}/{mok_param_name}/{swiglu_param_name}/{name}",
                expected,
                result,
                BF16_TOLERANCE,
                print_stats=rank == 0,
            )
        # Free this case before the next generate_inputs: at EP1 one case of a 384-expert shape holds
        # about 200 GiB (inputs, quantized weights, results and the reference) on the single GPU.
        del inputs, reference, actual, forward_context, mok_schedule, x, topk_experts, router_weights
        del w_shared_gate, w_shared_up, w_shared_down, w_routed_gate, w_routed_up, w_routed_down, d_output
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
        d_output,
    ) = inputs
    mok_schedule = functional.build_schedule(
        workspace,
        config,
        topk_experts,
        num_local_experts=num_local_experts,
    )
    _, forward_context = functional.forward(
        config,
        workspace,
        mok_schedule,
        x,
        router_weights,
        w_shared_gate,
        w_shared_up,
        w_shared_down,
        w_routed_gate,
        w_routed_up,
        w_routed_down,
    )
    valid_kwargs = {
        "config": config,
        "workspace": workspace,
        "schedule": mok_schedule,
        "forward_context": forward_context,
        "grad_output": d_output,
        "x": x,
        "router_weights": router_weights,
        "shared_gate_weights": w_shared_gate,
        "shared_up_weights": w_shared_up,
        "shared_down_weights": w_shared_down,
        "routed_gate_weights": w_routed_gate,
        "routed_up_weights": w_routed_up,
        "routed_down_weights": w_routed_down,
    }

    actual = ungated(functional.backward(**valid_kwargs))
    reference = run_reference_bf16(*inputs)[1:]
    for name, expected, result in zip(BACKWARD_RESULT_NAMES, reference, actual, strict=True):
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
        (
            "gradient output dtype",
            {"grad_output": d_output.float()},
            ValueError,
        ),
        (
            "gradient output shape",
            {"grad_output": d_output[:-256]},
            ValueError,
        ),
        ("input dtype", {"x": x.float()}, ValueError),
        (
            "router weight dtype",
            {"router_weights": router_weights.to(torch.bfloat16)},
            ValueError,
        ),
        (
            "schedule capacity",
            {"schedule": invalid_schedule},
            ValueError,
        ),
    ):
        try:
            ungated(functional.backward(**(valid_kwargs | overrides)))
        except expected_exception:
            pass
        else:
            raise AssertionError(f"{failure_name} should have failed")
