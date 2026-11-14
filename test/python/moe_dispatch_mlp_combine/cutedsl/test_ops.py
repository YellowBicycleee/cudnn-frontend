# Copyright 2026 Cursor Research
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Derived from Mixture-of-Kittens (https://github.com/cursor/mixture-of-kittens, commit 22fc95ae); modified by NVIDIA.

"""Op-level tests: the C++ port's tests/test_ops.py (d59188a), on the CuTe DSL kernels.

Every test body is the C++ one.  The fused megakernels are reached through op_adapters.py,
which takes the C++ ops' arguments; the two recompute_forward_context op tests are not here: this
package has no such op (``functional.recompute_forward_context`` leaves the routed recompute to the backward
kernel, see test_recompute_forward_context.py).  test_schedule also checks the
scheduler's fifth output, which the C++ scheduler does not have.
The schedule tables are sized with ``utils.capacity_multiplier`` instead of the C++ tests' fixed 1.5: on one GPU a rank
owns every expert, their padding to 256 rows overflows a 1.5 table and the scheduler traps (as upstream).
"""

import itertools
import math

import torch
import pytest
from cudnn.moe_dispatch_mlp_combine import ops
import torch.distributed as dist

from cudnn.moe_dispatch_mlp_combine import functional
from cudnn.moe_dispatch_mlp_combine.functional import get_workspace
from cudnn.moe_dispatch_mlp_combine.ops import (
    all_gather_top_experts,
    barrier_all,
    bwd_epilogue,
    fwd_epilogue,
    mxfp8_quantize,
    schedule,
)

# This port has no fused megakernel ops; the adapters take the C++ ops' arguments and return their
# outputs in the C++ order (see op_adapters.py).
from .op_adapters import (
    _valid_tokens_per_expert,
    dispatch_mlp_swiglu_combine_bwd_mxfp8,
    dispatch_mlp_swiglu_combine_bwd_bf16,
    dispatch_mlp_swiglu_combine_fwd_mxfp8,
    dispatch_mlp_swiglu_combine_fwd_bf16,
)

from .utils import (
    capacity_multiplier,
    check_correctness,
    generate_inputs,
    generate_topk_experts,
    mok_params,
    mxfp8_tolerance,
    swiglu_params,
    run_all_gather_top_experts_reference,
    run_bwd_epilogue_reference,
    run_forward_reference_bf16,
    run_fwd_epilogue_reference,
    run_mxfp8_quantize_reference,
    run_reference_bf16,
    run_schedule_reference,
    shapes,
)

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
        # The port's fifth output, the unpadded rows per expert, is what op_adapters derives from the
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
        (
            w_routed_gate_fp8,
            w_routed_gate_sc,
            _,
            _,
        ) = mxfp8_quantize(w_routed_gate, True, False)
        (
            w_routed_up_fp8,
            w_routed_up_sc,
            _,
            _,
        ) = mxfp8_quantize(w_routed_up, True, False)
        (
            w_routed_down_fp8,
            w_routed_down_sc,
            _,
            _,
        ) = mxfp8_quantize(w_routed_down, True, False)

        workspace.x_buffer.copy_(x)
        barrier_all(workspace.barrier_buffer, workspace.barrier_buffer_ptrs, workspace.barrier_buffer_multicast_ptr, workspace.barrier_target)
        outputs = dispatch_mlp_swiglu_combine_fwd_mxfp8(
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
            swiglu_limit,
            fwd_num_comm_sms,
            macrobatch_size,
            minibatch_size,
        )
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
            swiglu_limit,
        )
        for name, expected, result in zip(FORWARD_RESULT_NAMES, reference, actual, strict=True):
            check_correctness(
                f"{shape_name}/{mok_param_name}/{swiglu_param_name}/{name}",
                expected,
                result,
                MXFP8_TOLERANCE,
                print_stats=rank == 0,
            )

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
        outputs = dispatch_mlp_swiglu_combine_fwd_bf16(
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
            swiglu_limit,
            fwd_num_comm_sms,
            macrobatch_size,
            minibatch_size,
        )
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
            swiglu_limit,
        )
        for name, expected, result in zip(FORWARD_RESULT_NAMES, reference, actual, strict=True):
            check_correctness(
                f"{shape_name}/{mok_param_name}/{swiglu_param_name}/{name}",
                expected,
                result,
                BF16_TOLERANCE,
                print_stats=rank == 0,
            )

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
            swiglu_limit,
            fwd_num_comm_sms,
            macrobatch_size,
            minibatch_size,
        )
        barrier_all(workspace.barrier_buffer, workspace.barrier_buffer_ptrs, workspace.barrier_buffer_multicast_ptr, workspace.barrier_target)

        workspace.d_y_buffer.copy_(d_output)
        workspace.x_buffer.copy_(x)
        workspace.router_weight_buffer.copy_(router_weights)
        barrier_all(workspace.barrier_buffer, workspace.barrier_buffer_ptrs, workspace.barrier_buffer_multicast_ptr, workspace.barrier_target)
        outputs = dispatch_mlp_swiglu_combine_bwd_mxfp8(
            workspace.d_y_buffer,
            workspace.d_y_buffer_ptrs,
            workspace.d_x_routed_buffer,
            workspace.d_x_routed_buffer_ptrs,
            workspace.router_weight_buffer,
            workspace.router_weight_buffer_ptrs,
            workspace.d_router_weight_buffer,
            workspace.d_router_weight_buffer_ptrs,
            w_shared_gate,
            w_routed_gate_t_fp8,
            w_routed_gate_t_sc,
            w_shared_up,
            w_routed_up_t_fp8,
            w_routed_up_t_sc,
            w_shared_down,
            w_routed_down_t_fp8,
            w_routed_down_t_sc,
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
            workspace.x_buffer,
            workspace.x_buffer_ptrs,
            w_routed_gate_fp8,
            w_routed_gate_sc,
            w_routed_up_fp8,
            w_routed_up_sc,
            mok_schedule.peer_rank,
            mok_schedule.peer_token_idx,
            mok_schedule.num_tokens,
            mok_schedule.tokens_per_expert,
            topk,
            swiglu_limit,
            bwd_num_comm_sms,
            macrobatch_size,
            minibatch_size,
        )
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
        reference = run_reference_bf16(*inputs, swiglu_limit)[1:]
        for name, expected, result in zip(BACKWARD_RESULT_NAMES, reference, actual, strict=True):
            check_correctness(
                f"{shape_name}/{mok_param_name}/{swiglu_param_name}/{name}",
                expected,
                result,
                mxfp8_tolerance(world_size, shape_name, swiglu_param_name, name),
                print_stats=rank == 0,
            )
        # Free this case before the next generate_inputs (as the main branch's test): at one rank a 384-expert case
        # holds about 200 GiB (weights, their MXFP8 copies, the dW outputs and the reference's).
        del inputs, reference, actual, outputs, mok_schedule, w_routed_gate_fp8, w_routed_up_fp8, w_routed_down_fp8
        del w_routed_gate_t_fp8, w_routed_up_t_fp8, w_routed_down_t_fp8, w_routed_gate_sc, w_routed_up_sc
        del w_routed_down_sc, w_routed_gate_t_sc, w_routed_up_t_sc, w_routed_down_t_sc, x, topk_experts
        del router_weights, w_shared_gate, w_shared_up, w_shared_down, w_routed_gate, w_routed_up
        del w_routed_down, d_output, x_fp8_t_routed, x_sc_t_routed, gate_shared, gate_fp8_routed, gate_sc_routed
        del up_shared, up_fp8_routed, up_sc_routed, hidden_shared, hidden_fp8_t_routed, hidden_sc_t_routed
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
            swiglu_limit,
            fwd_num_comm_sms,
            macrobatch_size,
            minibatch_size,
        )
        barrier_all(workspace.barrier_buffer, workspace.barrier_buffer_ptrs, workspace.barrier_buffer_multicast_ptr, workspace.barrier_target)

        workspace.d_y_buffer.copy_(d_output)
        workspace.x_buffer.copy_(x)
        workspace.router_weight_buffer.copy_(router_weights)
        barrier_all(workspace.barrier_buffer, workspace.barrier_buffer_ptrs, workspace.barrier_buffer_multicast_ptr, workspace.barrier_target)
        outputs = dispatch_mlp_swiglu_combine_bwd_bf16(
            workspace.d_y_buffer,
            workspace.d_y_buffer_ptrs,
            workspace.d_x_routed_buffer,
            workspace.d_x_routed_buffer_ptrs,
            workspace.router_weight_buffer,
            workspace.router_weight_buffer_ptrs,
            workspace.d_router_weight_buffer,
            workspace.d_router_weight_buffer_ptrs,
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
            workspace.x_buffer,
            workspace.x_buffer_ptrs,
            mok_schedule.peer_rank,
            mok_schedule.peer_token_idx,
            mok_schedule.num_tokens,
            mok_schedule.tokens_per_expert,
            topk,
            swiglu_limit,
            bwd_num_comm_sms,
            macrobatch_size,
            minibatch_size,
        )
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
        reference = run_reference_bf16(*inputs, swiglu_limit)[1:]
        for name, expected, result in zip(BACKWARD_RESULT_NAMES, reference, actual, strict=True):
            check_correctness(
                f"{shape_name}/{mok_param_name}/{swiglu_param_name}/{name}",
                expected,
                result,
                BF16_TOLERANCE,
                print_stats=rank == 0,
            )
        # Free this case before the next generate_inputs (as the main branch's test): at one rank a 384-expert case
        # holds about 150 GiB (weights, the dW outputs and the reference's).
        del inputs, reference, actual, outputs, mok_schedule, x, topk_experts, router_weights, w_shared_gate
        del w_shared_up, w_shared_down, w_routed_gate, w_routed_up, w_routed_down, d_output
        del x_routed, gate_shared, gate_routed, up_shared, up_routed, hidden_shared, hidden_routed
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
