# Copyright 2026 Cursor Research
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Derived from Mixture-of-Kittens (https://github.com/cursor/mixture-of-kittens, commit 22fc95ae); modified by NVIDIA.

"""The backward megakernels (each after its forward) against the PyTorch reference.

Run under torchrun on SM100 / SM103, e.g. ``torchrun --standalone --nproc-per-node=8 -m pytest -s
test_backward.py``.  The test matrix in both precisions, the BF16 options (epilogue combine, split
minibatches, combined FC1, main_grad accumulation, sparse routes, a routed hidden size and the latent layer), the
MXFP8 backward's own geometries (EP8), the task -> expert decode variants, the per-call scratch and the metadata
tables.
"""

import contextlib
import dataclasses
import itertools
import math
import random
import unittest.mock
from unittest import mock

import pytest
import torch
import torch.distributed as dist

from cudnn.moe_dispatch_mlp_combine import functional
from cudnn.moe_dispatch_mlp_combine import functional as f
from cudnn.moe_dispatch_mlp_combine.cutedsl import persistent_bf16_bwd
from cudnn.moe_dispatch_mlp_combine.cutedsl import persistent_mxfp8_bwd as mxfp8_bwd_host
from cudnn.moe_dispatch_mlp_combine.cutedsl import persistent_mxfp8_fwd as mxfp8_fwd_host
from cudnn.moe_dispatch_mlp_combine.cutedsl._arch import SUPPORTED_CAPABILITIES
from cudnn.moe_dispatch_mlp_combine.cutedsl.persistent_bf16_bwd_contract import BWD_EXPERT_TABLE_CAP, RAW13_NAMES, raw13_descriptors
from cudnn.moe_dispatch_mlp_combine.cutedsl.persistent_bf16_fwd_contract import EXPERT_TABLE_CAP, TaskDecode
from cudnn.moe_dispatch_mlp_combine.functional import clear_workspace_cache, get_workspace
from cudnn.moe_dispatch_mlp_combine.ops import (
    make_routed_d_weight_storage_table,
    make_routed_scale_storage_table,
    make_routed_weight_storage_table_mxfp8,
    mxfp8_quantize,
)

from .test_forward import CLAMP_LIMIT, GEOMETRIES, LATENT_GEOMETRIES, _bytes, _sync_ranks
from .utils import (
    BACKWARD_RESULT_NAMES,
    BF16_TOLERANCE,
    FORWARD_RESULT_NAMES,
    MXFP8_TOLERANCE,
    MXFP8_UPSTREAM_ABS_MAX,
    capacity_multiplier,
    check_correctness,
    generate_inputs,
    mok_params,
    mxfp8_tolerance,
    run_forward_reference_bf16,
    run_fwd_epilogue_reference,
    run_reference_bf16,
    run_routed_topk_reduce_reference,
    shapes,
    swiglu_params,
    ungated,
)

# --------------------------------------------------------------------------------------------------------------------
# Backward over the test matrix (MXFP8; the BF16 matrix is test_cutedsl_forward_backward_bf16 below)


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
    the absolute bound, that result's bound is upstream's abs max (``mxfp8_tolerance``)."""
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


# --------------------------------------------------------------------------------------------------------------------
# BF16 forward and backward: matrix and options
#
# The megakernels against the PyTorch reference, and against their own previous run.
#
# Run under torchrun with eight GPUs on SM100 / SM103, e.g.
# ``torchrun --standalone --nproc-per-node=8 -m pytest -s moe_dispatch_mlp_combine/cutedsl/test_backward.py``.
# The CuTe DSL ports are SM100 / SM103 only, so the tests skip elsewhere. Every
# (shape, MoK parameter, SwiGLU) combination of the test matrix is checked against
# the PyTorch reference with ``BF16_TOLERANCE``, and every backward is run twice and
# required to be bit-identical.


# The EP sizes a workspace accepts; the kernels take the peer count from it at runtime.
SUPPORTED_WORLD_SIZES = (1, 4, 8, 16, 32, 64)


def _skip_unless_supported(world_size: int, device: torch.device) -> None:
    if world_size not in SUPPORTED_WORLD_SIZES:
        pytest.skip(f"a workspace supports EP sizes {SUPPORTED_WORLD_SIZES}")
    if torch.cuda.get_device_capability(device) not in SUPPORTED_CAPABILITIES:
        pytest.skip("CuTe DSL backends are SM100/SM103 only")


def _cases(world_size: int):
    return itertools.product(shapes(world_size), mok_params(), swiglu_params())


def _config(mok_param, world_size, shape, **overrides) -> functional.CuteMoKConfig:
    """The config of one test-matrix point; its minibatch serves both directions."""
    _, fwd_num_comm_sms, bwd_num_comm_sms, minibatch_size, macrobatch_size = mok_param
    _, num_experts, _, _, topk, num_local_tokens = shape
    return functional.CuteMoKConfig(
        fwd_num_comm_sms=fwd_num_comm_sms,
        bwd_num_comm_sms=bwd_num_comm_sms,
        minibatch_size=minibatch_size,
        macrobatch_size=macrobatch_size,
        schedule_capacity_multiplier=capacity_multiplier(world_size, num_local_tokens, topk, num_experts // world_size),
        **overrides,
    )


def _run(config, workspace, schedule, inputs, swiglu_limit):
    x, _, router_weights, w_shared_gate, w_shared_up, w_shared_down, w_routed_gate, w_routed_up, w_routed_down, d_output = inputs
    weights = (w_shared_gate, w_shared_up, w_shared_down, w_routed_gate, w_routed_up, w_routed_down)
    output, forward_context = functional.forward(config, workspace, schedule, x, router_weights, *weights, swiglu_limit)
    grads = ungated(functional.backward(config, workspace, schedule, forward_context, d_output, x, router_weights, *weights, swiglu_limit))
    return (output, forward_context.gate_shared, forward_context.up_shared, forward_context.hidden_shared), grads


def _check_case(context, config_overrides: dict | None = None, only_first_shape: bool = False) -> None:
    rank, world_size, device = context
    _skip_unless_supported(world_size, device)
    first_shape = shapes(world_size)[0]
    for shape, mok_param, swiglu_param in _cases(world_size):
        if only_first_shape and shape != first_shape:
            continue
        shape_name, num_experts, hidden_dim, intermediate_dim, topk, num_local_tokens = shape
        mok_param_name = mok_param[0]
        swiglu_param_name, swiglu_limit = swiglu_param
        assert num_experts % world_size == 0
        num_local_experts = num_experts // world_size
        label = f"cutedsl/{shape_name}/{mok_param_name}/{swiglu_param_name}"

        dsl_config = _config(mok_param, world_size, shape, **(config_overrides or {}))
        workspace = get_workspace(dsl_config, dist.group.WORLD, device=device, num_local_tokens=num_local_tokens, hidden_size=hidden_dim, topk=topk)
        inputs = generate_inputs(rank, device, num_experts, num_local_experts, topk, num_local_tokens, hidden_dim, intermediate_dim)
        topk_experts = inputs[1]
        dsl_schedule = functional.build_schedule(workspace, dsl_config, topk_experts, num_local_experts=num_local_experts)

        dsl_forward, dsl_grads = _run(dsl_config, workspace, dsl_schedule, inputs, swiglu_limit)
        # The opt CuTe BWD writes every routed dW tile itself in a fixed macrobatch
        # order (no zero kernel, no arrival-order reduce-add): the same inputs must
        # give bit-identical gradients on a second run.  The first results are kept as
        # they are (every call allocates fresh outputs; the raw13 poisoning relies on
        # that) and dropped right after the comparison: at EP1 each copy of the routed
        # dW is 50 GiB (384 x 3072 x 7168), and the inputs, the two results and the
        # reference have to fit on one GPU.  This is also the bit-exact replay MCore's
        # determinism test requires of the whole MoE layer.
        first_grads = dsl_grads
        _, dsl_grads = _run(dsl_config, workspace, dsl_schedule, inputs, swiglu_limit)
        for name, first, second in zip(BACKWARD_RESULT_NAMES, first_grads, dsl_grads, strict=True):
            assert torch.equal(first, second), f"{label}/{name}: second run differs bitwise"
        del first_grads
        # Return the freed copies' cached segments before the reference allocates its own gradients
        # (EP1: the allocator otherwise holds tens of GiB of unusable fragments next to 200 GiB live).
        torch.cuda.empty_cache()

        reference_combine, reference_gate, reference_up, reference_hidden, reference_y_shared = run_forward_reference_bf16(
            inputs[0], topk_experts, *inputs[3:9], swiglu_limit
        )
        reference_forward = (run_fwd_epilogue_reference(reference_y_shared, reference_combine, inputs[2]), reference_gate, reference_up, reference_hidden)
        reference_grads = run_reference_bf16(*inputs, swiglu_limit)[1:]

        for name, expected, dsl_result in zip(
            FORWARD_RESULT_NAMES + BACKWARD_RESULT_NAMES, reference_forward + tuple(reference_grads), dsl_forward + tuple(dsl_grads), strict=True
        ):
            check_correctness(f"{label}/{name} vs reference", expected, dsl_result, BF16_TOLERANCE, print_stats=rank == 0)
        # Free this case before the next generate_inputs (EP1: about 200 GiB per 384-expert case).
        del inputs, dsl_schedule, dsl_forward, dsl_grads
        del reference_forward, reference_grads, topk_experts
        torch.cuda.empty_cache()


def test_cutedsl_forward_backward_bf16(context: tuple[int, int, torch.device]) -> None:
    """Persistent BF16 megakernel ports (CuteMoKConfig)."""
    # The opt CuTe BWD owns every routed dW tile (no zero kernel): NaN-poison the routed dW
    # outputs of each call (raw13 8, 10, 12, fresh allocations) so a tile the kernel failed to
    # write cannot pass.
    # The swap-AB FWD skips the padding subtiles of its routed tiles, but the BWD reads the
    # macrobatch-0 Gate/Up context of every padded row (times router weight 0): NaN-poison
    # the fresh FWD outputs too, so a padded row it failed to zero cannot pass.
    from cudnn.moe_dispatch_mlp_combine.cutedsl import persistent_bf16_bwd as bwd_host
    from cudnn.moe_dispatch_mlp_combine.cutedsl import persistent_bf16_fwd as fwd_host

    allocate_raw13 = bwd_host._allocate_raw13
    allocate_state = fwd_host._allocate_state
    poisoned_calls = []

    def poisoned_raw13(shapes, proto, main_grads=None, *args, **kwargs):
        # main_grads is the fused accumulation's destination; these cases do not use it, and
        # poisoning a caller's buffer would be wrong anyway.
        raw13 = allocate_raw13(shapes, proto, main_grads, *args, **kwargs)
        for tensor in raw13[8::2]:  # d_w_routed_{gate,up,down}
            tensor.fill_(float("nan"))
        poisoned_calls.append(len(raw13))
        return raw13

    def poisoned_state(layout, device):
        outputs, counters = allocate_state(layout, device)
        for output in outputs:
            output.fill_(float("nan"))
        return outputs, counters

    with unittest.mock.patch.object(bwd_host, "_allocate_raw13", poisoned_raw13), unittest.mock.patch.object(fwd_host, "_allocate_state", poisoned_state):
        _check_case(context)
    assert poisoned_calls, "the backward did not allocate its raw13 through the patched helper"


@contextlib.contextmanager
def _epilogue_combine_probe():
    """What the forwards inside the block did, for the ``epilogue_combine`` tests.  The fresh forward outputs are
    NaN-poisoned (as in test_cutedsl_forward_backward_bf16).  Yields the list of "the Y ring was left untouched" flags,
    one per BF16 forward call: the default placement stores the routed Down rows into the Y ring, the
    ``epilogue_combine`` variant never writes that output, so the flag tells which variant a call really ran."""
    from cudnn.moe_dispatch_mlp_combine.cutedsl import persistent_bf16_fwd as fwd_host

    allocate_state = fwd_host._allocate_state
    forward_bf16 = fwd_host.forward_bf16
    y_ring_untouched = []

    def poisoned_state(layout, device):
        outputs, counters = allocate_state(layout, device)
        for output in outputs:
            output.fill_(float("nan"))
        return outputs, counters

    def recording_forward(*args, **kwargs):
        outputs = forward_bf16(*args, **kwargs)
        y_ring_untouched.append(torch.isnan(outputs[8]).all())  # y_routed, the last of the nine outputs
        return outputs

    with unittest.mock.patch.object(fwd_host, "_allocate_state", poisoned_state), unittest.mock.patch.object(fwd_host, "forward_bf16", recording_forward):
        yield y_ring_untouched


def _poison_combine(workspace) -> None:
    """NaN in every row of this rank's combine buffer, so a row a forward fails to store cannot pass on the values an
    earlier forward of the same workspace left there.  Every rank calls it before the forward: the fill is ahead of
    this rank's entry barrier on its stream, and a peer stores combine rows only after that barrier."""
    workspace.combine_buffer.fill_(float("nan"))


def _assert_combine_written(workspace, label: str) -> None:
    # Every (token, top-k) route has one combine row, so no row may keep the poison.
    assert not torch.isnan(workspace.combine_buffer).any(), f"{label}: a combine row was not written"


def _assert_variants_ran(y_ring_untouched, label: str) -> None:
    """The default call wrote the Y ring, the ``epilogue_combine`` call did not, and its executor is a cache entry
    of its own."""
    from cudnn.moe_dispatch_mlp_combine.cutedsl import persistent_bf16_fwd as fwd_host

    default_untouched, pushed_untouched = (bool(flag) for flag in y_ring_untouched)
    assert not default_untouched, f"{label}: the default placement left the Y ring untouched"
    assert pushed_untouched, f"{label}: epilogue_combine=True wrote the Y ring (the default variant ran)"
    assert any(("epilogue_combine", True) in key for key in fwd_host._EXECUTOR_CACHE), f"{label}: no executor was compiled for epilogue_combine=True"


def test_cutedsl_epilogue_combine_bf16(context: tuple[int, int, torch.device]) -> None:
    """CuteMoKConfig(epilogue_combine=True): the routed Down epilogue pushes the combine rows and the communication
    CTAs only pull.  Every test-matrix case (one macrobatch and several): the forward output and context are bit for
    bit those of the default placement (the same GEMMs and rows, only the store path differs), and so are the
    gradients of the backward that follows it.  The combine buffer is NaN-poisoned before each run and must be fully
    rewritten (the two runs share one workspace), and each run is checked to be the variant it names
    (_epilogue_combine_probe)."""
    rank, world_size, device = context
    _skip_unless_supported(world_size, device)
    with _epilogue_combine_probe() as y_ring_untouched:
        for shape, mok_param, swiglu_param in _cases(world_size):
            shape_name, num_experts, hidden_dim, intermediate_dim, topk, num_local_tokens = shape
            swiglu_param_name, swiglu_limit = swiglu_param
            label = f"epilogue_combine/{shape_name}/{mok_param[0]}/{swiglu_param_name}"
            num_local_experts = num_experts // world_size
            base = _config(mok_param, world_size, shape)
            pushed = dataclasses.replace(base, epilogue_combine=True)
            workspace = get_workspace(
                _config(mok_param, world_size, shape), dist.group.WORLD, device=device, num_local_tokens=num_local_tokens, hidden_size=hidden_dim, topk=topk
            )
            inputs = generate_inputs(rank, device, num_experts, num_local_experts, topk, num_local_tokens, hidden_dim, intermediate_dim)
            schedule = functional.build_schedule(workspace, base, inputs[1], num_local_experts=num_local_experts)
            y_ring_untouched.clear()
            _poison_combine(workspace)
            base_forward, base_grads = _run(base, workspace, schedule, inputs, swiglu_limit)
            _assert_combine_written(workspace, f"{label}/default")
            _poison_combine(workspace)
            pushed_forward, pushed_grads = _run(pushed, workspace, schedule, inputs, swiglu_limit)
            _assert_combine_written(workspace, label)
            _assert_variants_ran(y_ring_untouched, label)
            for name, expected, actual in zip(
                FORWARD_RESULT_NAMES + BACKWARD_RESULT_NAMES, base_forward + tuple(base_grads), pushed_forward + tuple(pushed_grads), strict=True
            ):
                assert torch.equal(expected, actual), f"{label}/{name}: differs from the default placement"
            del inputs, schedule, base_forward, base_grads, pushed_forward, pushed_grads
            torch.cuda.empty_cache()


def test_cutedsl_epilogue_combine_small_minibatch_bf16(context: tuple[int, int, torch.device]) -> None:
    """``epilogue_combine`` on two macrobatches of 256-row minibatches with a small hidden (one 256-column tile): each
    rank owns four experts of about 640 rows, padded to 768, so its 3072 rows are a full macrobatch of 2048 and a
    short one, eight and four minibatches -- the pull-only stream crosses the macrobatch boundary once per
    minibatch.  The forward matches the torch reference and is bit for bit the default placement's; poisoned combine
    buffer and variant check as in test_cutedsl_epilogue_combine_bf16."""
    rank, world_size, device = context
    _skip_unless_supported(world_size, device)
    num_local_experts, hidden_dim, intermediate_dim, topk, num_local_tokens = 4, 256, 512, 2, 1280
    num_experts = num_local_experts * world_size
    label = "epilogue_combine/small-minibatch"
    base = functional.CuteMoKConfig(
        fwd_minibatch_size=256,
        bwd_minibatch_size=256,
        macrobatch_size=2048,
        schedule_capacity_multiplier=capacity_multiplier(world_size, num_local_tokens, topk, num_local_experts),
    )
    pushed = dataclasses.replace(base, epilogue_combine=True)
    workspace = get_workspace(base, dist.group.WORLD, device=device, num_local_tokens=num_local_tokens, hidden_size=hidden_dim, topk=topk)
    inputs = generate_inputs(rank, device, num_experts, num_local_experts, topk, num_local_tokens, hidden_dim, intermediate_dim)
    x, topk_experts, router_weights, *weights, _ = inputs
    schedule = functional.build_schedule(workspace, base, topk_experts, num_local_experts=num_local_experts)
    assert (
        base.macrobatch_size < int(schedule.num_tokens[0]) <= 2 * base.macrobatch_size
    ), f"{label}: {int(schedule.num_tokens[0])} rows are not two macrobatches"
    results = []
    with _epilogue_combine_probe() as y_ring_untouched:
        for config, name in ((base, f"{label}/default"), (pushed, label)):
            _poison_combine(workspace)
            output, forward_context = functional.forward(config, workspace, schedule, x, router_weights, *weights, None)
            _assert_combine_written(workspace, name)
            results.append((output, forward_context.gate_shared, forward_context.up_shared, forward_context.hidden_shared))
    _assert_variants_ran(y_ring_untouched, label)
    reference_combine, reference_gate, reference_up, reference_hidden, reference_y_shared = run_forward_reference_bf16(x, topk_experts, *weights, None)
    reference = (run_fwd_epilogue_reference(reference_y_shared, reference_combine, router_weights), reference_gate, reference_up, reference_hidden)
    for name, expected, default, actual in zip(FORWARD_RESULT_NAMES, reference, *results, strict=True):
        check_correctness(f"{label}/{name} vs reference", expected, actual, BF16_TOLERANCE, print_stats=rank == 0)
        assert torch.equal(default, actual), f"{label}/{name}: differs from the default placement"


def test_cutedsl_epilogue_combine_routed_hidden_bf16(context: tuple[int, int, torch.device]) -> None:
    """``epilogue_combine`` with the routed experts in their own hidden (latent MoE: routed_hidden_size !=
    hidden_size), where a combine row has the routed width: Qwen with a routed hidden of 1280 on one macrobatch and
    DeepSeek H/2 on several.  y_shared, the routed top-k reduction and the shared context are bit for bit the default
    placement's (checked against the reference in test_cutedsl_routed_hidden_bf16); poisoned combine buffer and
    variant check as in test_cutedsl_epilogue_combine_bf16."""
    rank, world_size, device = context
    _skip_unless_supported(world_size, device)
    minimum_macro = next(p for p in mok_params() if p[0] == "Minimum macrobatch")
    qwen, deepseek = shapes(world_size)[2], shapes(world_size)[3]
    for shape, mok_param, routed_hidden_dim, swiglu_limit in ((qwen, mok_params()[0], 1280, None), (deepseek, minimum_macro, deepseek[2] // 2, 4.0)):
        shape_name, num_experts, hidden_dim, intermediate_dim, topk, num_local_tokens = shape
        num_local_experts = num_experts // world_size
        label = f"epilogue_combine-routed-hidden/{shape_name}/{routed_hidden_dim}/{mok_param[0]}"
        base = _config(mok_param, world_size, shape)
        pushed = dataclasses.replace(base, epilogue_combine=True)
        workspace = functional.create_workspace(
            base, dist.group.WORLD, device=device, num_local_tokens=num_local_tokens, hidden_size=hidden_dim, topk=topk, routed_hidden_size=routed_hidden_dim
        )
        hidden_inputs = generate_inputs(rank, device, num_experts, num_local_experts, topk, num_local_tokens, hidden_dim, intermediate_dim)
        routed_inputs = generate_inputs(rank, device, num_experts, num_local_experts, topk, num_local_tokens, routed_hidden_dim, intermediate_dim)
        x, topk_experts, router_weights, w_shared_gate, w_shared_up, w_shared_down = hidden_inputs[:6]
        routed_input, routed_weights = routed_inputs[0], routed_inputs[6:9]
        schedule = functional.build_schedule(workspace, base, topk_experts, num_local_experts=num_local_experts)
        results = []
        with _epilogue_combine_probe() as y_ring_untouched:
            for config, name in ((base, f"{label}/default"), (pushed, label)):
                _poison_combine(workspace)
                (y_shared, y_routed), forward_context = functional.forward(
                    config,
                    workspace,
                    schedule,
                    x,
                    router_weights,
                    w_shared_gate,
                    w_shared_up,
                    w_shared_down,
                    *routed_weights,
                    swiglu_limit,
                    routed_input=routed_input,
                )
                _assert_combine_written(workspace, name)
                assert tuple(y_routed.shape) == (num_local_tokens, routed_hidden_dim)
                results.append((y_shared, y_routed, forward_context.gate_shared, forward_context.up_shared, forward_context.hidden_shared))
        _assert_variants_ran(y_ring_untouched, label)
        for name, default, actual in zip(("y_shared", "y_routed", *FORWARD_RESULT_NAMES[1:]), *results, strict=True):
            assert not torch.isnan(actual).any(), f"{label}/{name}: NaN"
            assert torch.equal(default, actual), f"{label}/{name}: differs from the default placement"
        del hidden_inputs, routed_inputs, schedule, results, workspace
        torch.cuda.empty_cache()


@pytest.mark.parametrize("hidden_dim", [1152, 4096])  # a 128-wide H tail (9 x 128), and a 256-aligned H
@pytest.mark.parametrize("macrobatch_size", [512, 4096])  # several macrobatches, and one
def test_cutedsl_epilogue_combine_hidden_tail_bf16(context: tuple[int, int, torch.device], hidden_dim: int, macrobatch_size: int) -> None:
    """``epilogue_combine`` with a 128-wide hidden tail (H % 256 == 128).  The tail Down tile has valid features in
    its first CTA only; the second CTA's 128 features lie past H and must not be stored (they would land in the first
    128 columns of the next combine row, racing with that row's own stores).  512 tokens per rank, two local experts,
    top-k 2, 256-row minibatches.  The forward output and context and every gradient match the PyTorch reference and
    are bit for bit the default placement's, and a second run of the variant is bit for bit the first.  Poisoned
    combine buffer and variant check as in test_cutedsl_epilogue_combine_bf16."""
    rank, world_size, device = context
    _skip_unless_supported(world_size, device)
    num_local_experts, intermediate_dim, topk, num_local_tokens = 2, 512, 2, 512
    num_experts = num_local_experts * world_size
    label = f"epilogue_combine/hidden-{hidden_dim}/macrobatch-{macrobatch_size}"
    base = functional.CuteMoKConfig(
        fwd_num_comm_sms=2,
        bwd_num_comm_sms=2,
        minibatch_size=256,
        macrobatch_size=macrobatch_size,
        schedule_capacity_multiplier=capacity_multiplier(world_size, num_local_tokens, topk, num_local_experts),
    )
    pushed = dataclasses.replace(base, epilogue_combine=True)
    workspace = functional.create_workspace(base, dist.group.WORLD, device=device, num_local_tokens=num_local_tokens, hidden_size=hidden_dim, topk=topk)
    inputs = generate_inputs(rank, device, num_experts, num_local_experts, topk, num_local_tokens, hidden_dim, intermediate_dim)
    schedule = functional.build_schedule(workspace, base, inputs[1], num_local_experts=num_local_experts)
    results = []
    with _epilogue_combine_probe() as y_ring_untouched:
        for config, name in ((base, f"{label}/default"), (pushed, label), (pushed, f"{label}/second run")):
            _poison_combine(workspace)
            forward, grads = _run(config, workspace, schedule, inputs, None)
            _assert_combine_written(workspace, name)
            results.append(forward + tuple(grads))
    _assert_variants_ran(y_ring_untouched[:2], label)
    assert bool(y_ring_untouched[2]), f"{label}: the second run wrote the Y ring (the default variant ran)"
    reference_combine, reference_gate, reference_up, reference_hidden, reference_y_shared = run_forward_reference_bf16(inputs[0], inputs[1], *inputs[3:9], None)
    reference = (run_fwd_epilogue_reference(reference_y_shared, reference_combine, inputs[2]), reference_gate, reference_up, reference_hidden) + tuple(
        run_reference_bf16(*inputs, None)[1:]
    )
    for name, expected, default, actual, repeated in zip(FORWARD_RESULT_NAMES + BACKWARD_RESULT_NAMES, reference, *results, strict=True):
        check_correctness(f"{label}/{name} vs reference", expected, actual, BF16_TOLERANCE, print_stats=rank == 0)
        assert torch.equal(default, actual), f"{label}/{name}: differs from the default placement"
        assert torch.equal(actual, repeated), f"{label}/{name}: second run differs bitwise"
    del inputs, schedule, results, reference, workspace
    torch.cuda.empty_cache()


def test_cutedsl_split_minibatch_bf16(context: tuple[int, int, torch.device]) -> None:
    """Different forward and backward minibatches (shared macrobatch): each direction sizes its
    own readiness counters by its own minibatch, and the results match the reference.  The fresh
    forward
    outputs and routed dW are NaN-poisoned (as in test_cutedsl_forward_backward_bf16), so a row or
    tile a direction fails to write cannot pass on memory left from an earlier case."""
    rank, world_size, device = context
    _skip_unless_supported(world_size, device)
    from cudnn.moe_dispatch_mlp_combine.cutedsl import persistent_bf16_bwd as bwd_host
    from cudnn.moe_dispatch_mlp_combine.cutedsl import persistent_bf16_fwd as fwd_host

    allocate_state = fwd_host._allocate_state
    allocate_raw13 = bwd_host._allocate_raw13
    counter_lengths = {}
    poisoned_calls = []

    def recording_state(layout, device):
        outputs, counters = allocate_state(layout, device)
        for output in outputs:
            output.fill_(float("nan"))
        counter_lengths["x_routed_ready"] = counters[2].numel()
        return outputs, counters

    def poisoned_raw13(shapes, proto, main_grads=None, *args, **kwargs):
        # main_grads is the fused accumulation's destination; these cases do not use it, and
        # poisoning a caller's buffer would be wrong anyway.
        raw13 = allocate_raw13(shapes, proto, main_grads, *args, **kwargs)
        for tensor in raw13[8::2]:  # d_w_routed_{gate,up,down}
            tensor.fill_(float("nan"))
        poisoned_calls.append(len(raw13))
        return raw13

    swiglu_limit = None
    # Qwen and DeepSeek; (forward minibatch, backward minibatch, macrobatch): smaller and larger
    # backward minibatch, on one macrobatch and on several (the backward replays the later ones).
    for shape, (fwd_minibatch, bwd_minibatch, macrobatch) in itertools.product(
        (s for s in shapes(world_size) if s[0].startswith(("Qwen", "DeepSeek"))),
        ((16384, 8192, 131072), (8192, 16384, 131072), (4096, 8192, 16384), (8192, 4096, 16384)),
    ):
        shape_name, num_experts, hidden_dim, intermediate_dim, topk, num_local_tokens = shape
        num_local_experts = num_experts // world_size
        label = f"split/{shape_name}/fwd{fwd_minibatch}/bwd{bwd_minibatch}/macro{macrobatch}"
        multiplier = capacity_multiplier(world_size, num_local_tokens, topk, num_local_experts)
        dsl_config = functional.CuteMoKConfig(
            fwd_minibatch_size=fwd_minibatch, bwd_minibatch_size=bwd_minibatch, macrobatch_size=macrobatch, schedule_capacity_multiplier=multiplier
        )
        workspace = get_workspace(dsl_config, dist.group.WORLD, device=device, num_local_tokens=num_local_tokens, hidden_size=hidden_dim, topk=topk)
        inputs = generate_inputs(rank, device, num_experts, num_local_experts, topk, num_local_tokens, hidden_dim, intermediate_dim)
        x, topk_experts, router_weights, *weights, d_output = inputs
        dsl_schedule = functional.build_schedule(workspace, dsl_config, topk_experts, num_local_experts=num_local_experts)

        counter_lengths.clear()
        poisoned_calls.clear()
        with unittest.mock.patch.object(fwd_host, "_allocate_state", recording_state), unittest.mock.patch.object(bwd_host, "_allocate_raw13", poisoned_raw13):
            dsl_forward, dsl_grads = _run(dsl_config, workspace, dsl_schedule, inputs, swiglu_limit)
        # The readiness counters of the buffer set this backward ran on: a workspace keeps one set per
        # backward configuration, so an earlier call with this configuration may have prepared it.
        counter_lengths["d_y_routed_ready"] = workspace._cutedsl_bwd_preparation.counters[0].numel()
        capacity = workspace.schedule_capacity
        assert poisoned_calls, f"{label}: the backward did not allocate its raw13 through the patched helper"
        assert counter_lengths == {
            "x_routed_ready": -(-capacity // fwd_minibatch),
            "d_y_routed_ready": -(-capacity // bwd_minibatch),
        }, f"{label}: counters {counter_lengths} for capacity {capacity}"

        reference_combine, reference_gate, reference_up, reference_hidden, reference_y_shared = run_forward_reference_bf16(
            x, topk_experts, *inputs[3:9], swiglu_limit
        )
        reference_forward = (run_fwd_epilogue_reference(reference_y_shared, reference_combine, router_weights), reference_gate, reference_up, reference_hidden)
        reference_grads = run_reference_bf16(*inputs, swiglu_limit)[1:]
        for name, expected, dsl_result in zip(
            FORWARD_RESULT_NAMES + BACKWARD_RESULT_NAMES, reference_forward + tuple(reference_grads), dsl_forward + tuple(dsl_grads), strict=True
        ):
            check_correctness(f"{label}/{name} vs reference", expected, dsl_result, BF16_TOLERANCE, print_stats=rank == 0)
        # Free this case before the next generate_inputs (EP1: about 200 GiB per 384-expert case).
        del inputs, x, topk_experts, router_weights, weights, d_output, dsl_schedule
        del dsl_forward, dsl_grads
        del reference_combine, reference_gate, reference_up, reference_hidden, reference_y_shared
        del reference_forward, reference_grads
        torch.cuda.empty_cache()


def test_cutedsl_combined_fc1_bf16(context: tuple[int, int, torch.device]) -> None:
    """One combined [E, 2I, H] FC1 parameter, passed as both Gate and Up.

    MCore owns the routed experts' FC1 as a single parameter and hands MOK the same object twice
    rather than two narrow views.  The result must be what two separate weights give, bit for bit:
    the halves are addressed as strided views, so nothing is copied and nothing is reordered.
    Where Up starts is a runtime argument of the kernels: the combined call runs the executors
    the separate weights compiled.
    """
    from cudnn.moe_dispatch_mlp_combine.cutedsl import persistent_bf16_bwd as bwd_host
    from cudnn.moe_dispatch_mlp_combine.cutedsl import persistent_bf16_fwd as fwd_host

    rank, world_size, device = context
    _skip_unless_supported(world_size, device)
    shape = shapes(world_size)[2]
    shape_name, num_experts, hidden_dim, intermediate_dim, topk, num_local_tokens = shape
    num_local_experts = num_experts // world_size
    swiglu_limit = swiglu_params()[1][1]
    config = _config(mok_params()[0], world_size, shape)
    workspace = get_workspace(config, dist.group.WORLD, device=device, num_local_tokens=num_local_tokens, hidden_size=hidden_dim, topk=topk)
    inputs = generate_inputs(rank, device, num_experts, num_local_experts, topk, num_local_tokens, hidden_dim, intermediate_dim)
    schedule = functional.build_schedule(workspace, config, inputs[1], num_local_experts=num_local_experts)

    separate_forward, separate_grads = _run(config, workspace, schedule, inputs, swiglu_limit)
    separate_forward = tuple(t.clone() for t in separate_forward)
    separate_grads = tuple(t.clone() for t in separate_grads)
    executors = (set(fwd_host._EXECUTOR_CACHE), set(bwd_host._EXECUTOR_CACHE))

    # The same weights as one parameter, Gate first, and the same object for both arguments.
    x, topk_experts, router_weights, w_shared_gate, w_shared_up, w_shared_down, w_routed_gate, w_routed_up, w_routed_down, d_output = inputs
    combined_fc1 = torch.cat((w_routed_gate, w_routed_up), dim=1).contiguous()
    assert tuple(combined_fc1.shape) == (num_local_experts, 2 * intermediate_dim, hidden_dim)
    weights = (w_shared_gate, w_shared_up, w_shared_down, combined_fc1, combined_fc1, w_routed_down)
    combined_output, combined_context = functional.forward(config, workspace, schedule, x, router_weights, *weights, swiglu_limit)
    combined_forward = (combined_output, combined_context.gate_shared, combined_context.up_shared, combined_context.hidden_shared)
    combined_grads = ungated(functional.backward(config, workspace, schedule, combined_context, d_output, x, router_weights, *weights, swiglu_limit))

    for name, expected, actual in zip(
        FORWARD_RESULT_NAMES + BACKWARD_RESULT_NAMES, separate_forward + separate_grads, combined_forward + tuple(combined_grads), strict=True
    ):
        assert torch.equal(expected, actual), f"combined-fc1/{shape_name}/{name} differs"
    assert (set(fwd_host._EXECUTOR_CACHE), set(bwd_host._EXECUTOR_CACHE)) == executors, "the combined FC1 compiled executors of its own"


def test_cutedsl_main_grad_accumulation_bf16(context: tuple[int, int, torch.device]) -> None:
    """Weight gradients accumulated into the caller's buffers, as MCore's optimizer needs them.

    MCore requires gradient_accumulation_fusion: a weight's gradient belongs in the distributed
    optimizer's main_grad buffer, and MCore reads the buffer rather than the return value.  Three
    properties are checked: a buffer that starts at zero ends at the fresh gradient; a second
    microbatch adds to it; and the returned entries alias the buffers.  The add is deterministic,
    which the whole-layer replay MCore asserts depends on.
    """
    rank, world_size, device = context
    _skip_unless_supported(world_size, device)
    shape = shapes(world_size)[2]
    shape_name, num_experts, hidden_dim, intermediate_dim, topk, num_local_tokens = shape
    num_local_experts = num_experts // world_size
    swiglu_limit = swiglu_params()[1][1]
    config = _config(mok_params()[0], world_size, shape)
    workspace = get_workspace(config, dist.group.WORLD, device=device, num_local_tokens=num_local_tokens, hidden_size=hidden_dim, topk=topk)
    inputs = generate_inputs(rank, device, num_experts, num_local_experts, topk, num_local_tokens, hidden_dim, intermediate_dim)
    schedule = functional.build_schedule(workspace, config, inputs[1], num_local_experts=num_local_experts)
    x, _, router_weights, *weights, d_output = inputs

    _, context_fresh = functional.forward(config, workspace, schedule, x, router_weights, *weights, swiglu_limit)
    fresh = ungated(functional.backward(config, workspace, schedule, context_fresh, d_output, x, router_weights, *weights, swiglu_limit))
    # backward returns routed gate/up/down first, then shared; main_grads interleaves them.
    _, _, d_routed_gate, d_routed_up, d_routed_down, d_shared_gate, d_shared_up, d_shared_down = tuple(t.clone() for t in fresh)
    fresh_in_main_grad_order = (d_shared_gate, d_routed_gate, d_shared_up, d_routed_up, d_shared_down, d_routed_down)

    for dtype in (torch.float32, torch.bfloat16):
        buffers = tuple(torch.zeros_like(t, dtype=dtype) for t in fresh_in_main_grad_order)
        for microbatch in (1, 2):
            _, forward_context = functional.forward(config, workspace, schedule, x, router_weights, *weights, swiglu_limit)
            returned = ungated(
                functional.backward(config, workspace, schedule, forward_context, d_output, x, router_weights, *weights, swiglu_limit, main_grads=buffers)
            )
            returned_in_main_grad_order = (returned[5], returned[2], returned[6], returned[3], returned[7], returned[4])
            for name, buffer, entry in zip(functional.MAIN_GRAD_NAMES, buffers, returned_in_main_grad_order, strict=True):
                assert entry is buffer, f"{name}: the returned entry does not alias the buffer"
            for name, buffer, once in zip(functional.MAIN_GRAD_NAMES, buffers, fresh_in_main_grad_order, strict=True):
                expected = once.to(dtype) * microbatch
                check_correctness(f"main-grad/{shape_name}/{dtype}/{name}/microbatch{microbatch}", expected, buffer, BF16_TOLERANCE, print_stats=rank == 0)


def test_cutedsl_sparse_routes_bf16(context: tuple[int, int, torch.device]) -> None:
    """Tokens that use fewer than topk experts, end to end.

    MCore's routing map is the authority on where a token goes, so a slot may be unused: its
    expert is -1, its weight 0, and neither the dispatch nor the backward writes its row of the
    symmetric buffers.  Those buffers are allocated rather than zeroed, so this also checks that
    nothing stale leaks into the result -- the case a zero weight alone would not cover.
    """
    rank, world_size, device = context
    _skip_unless_supported(world_size, device)
    shape = shapes(world_size)[2]
    shape_name, num_experts, hidden_dim, intermediate_dim, topk, num_local_tokens = shape
    num_local_experts = num_experts // world_size
    swiglu_limit = swiglu_params()[1][1]
    config = _config(mok_params()[0], world_size, shape)
    workspace = get_workspace(config, dist.group.WORLD, device=device, num_local_tokens=num_local_tokens, hidden_size=hidden_dim, topk=topk)
    for fraction in (0.25, 1.0):  # some slots unused, then every slot of every token
        label = f"sparse/{shape_name}/missing{fraction}"
        inputs = generate_inputs(
            rank, device, num_experts, num_local_experts, topk, num_local_tokens, hidden_dim, intermediate_dim, missing_route_fraction=fraction
        )
        topk_experts = inputs[1]
        unused = int((topk_experts < 0).sum())
        assert unused > 0, f"{label}: the case produced no unused slot"
        schedule = functional.build_schedule(workspace, config, topk_experts, num_local_experts=num_local_experts)
        assert schedule.top_experts is not None and schedule.top_experts.dtype == torch.int32

        dsl_forward, dsl_grads = _run(config, workspace, schedule, inputs, swiglu_limit)
        for name, tensor in zip(FORWARD_RESULT_NAMES + BACKWARD_RESULT_NAMES, dsl_forward + tuple(dsl_grads), strict=True):
            assert torch.isfinite(tensor.float()).all(), f"{label}/{name}: not finite"
        # An unused slot has no router-weight gradient.
        assert torch.equal(dsl_grads[1][topk_experts < 0], torch.zeros_like(dsl_grads[1][topk_experts < 0])), f"{label}: d_router_weights"

        reference_combine, reference_gate, reference_up, reference_hidden, reference_y_shared = run_forward_reference_bf16(
            inputs[0], topk_experts, *inputs[3:9], swiglu_limit
        )
        reference_forward = (
            run_fwd_epilogue_reference(reference_y_shared, reference_combine, inputs[2], topk_experts),
            reference_gate,
            reference_up,
            reference_hidden,
        )
        reference_grads = run_reference_bf16(*inputs, swiglu_limit)[1:]
        for name, expected, actual in zip(
            FORWARD_RESULT_NAMES + BACKWARD_RESULT_NAMES, reference_forward + tuple(reference_grads), dsl_forward + tuple(dsl_grads), strict=True
        ):
            check_correctness(f"{label}/{name} vs reference", expected, actual, BF16_TOLERANCE, print_stats=rank == 0)
        del inputs, schedule, dsl_forward, dsl_grads, reference_forward, reference_grads
        torch.cuda.empty_cache()


def test_cutedsl_routed_hidden_bf16(context: tuple[int, int, torch.device]) -> None:
    """Routed experts in routed_hidden_size, shared expert in hidden_size (FWD + BWD).

    Cases: Qwen H/2 (one macrobatch), DeepSeek H/2 on multiple macrobatches, Qwen 1280 (a short last 256-column
    communication tile) and Qwen 5120 (routed hidden larger than the shared hidden, multiple macrobatches).

    x / shared weights use the hidden dim, routed_input / routed weights the routed hidden dim; the
    routed source pulled by the peers is routed_input.  Checked against the torch reference for
    y_shared and the routed top-k reduction separately (no CUDA path exists for this shape).
    """
    rank, world_size, device = context
    _skip_unless_supported(world_size, device)
    # "Minimum macrobatch" 4096 gives multiple macrobatches so the routed ring reuse (y_routed_done /
    # y_routed_ready across macrobatches) runs in the routed width.
    minimum_macro = next(p for p in mok_params() if p[0] == "Minimum macrobatch")
    qwen, deepseek = shapes(world_size)[2], shapes(world_size)[3]
    clamped, unclamped = swiglu_params()[1], swiglu_params()[0]
    cases = (
        (qwen, mok_params()[0], qwen[2] // 2, clamped),
        (deepseek, minimum_macro, deepseek[2] // 2, clamped),
        (qwen, mok_params()[0], 1280, unclamped),
        (qwen, minimum_macro, 5120, unclamped),
    )
    for shape, mok_param, routed_hidden_dim, (swiglu_param_name, swiglu_limit) in cases:
        shape_name, num_experts, hidden_dim, intermediate_dim, topk, num_local_tokens = shape
        num_local_experts = num_experts // world_size
        label = f"cutedsl-routed-hidden/{shape_name}/{routed_hidden_dim}/{mok_param[0]}/{swiglu_param_name}"
        config = _config(mok_param, world_size, shape)
        workspace = functional.create_workspace(
            config, dist.group.WORLD, device=device, num_local_tokens=num_local_tokens, hidden_size=hidden_dim, topk=topk, routed_hidden_size=routed_hidden_dim
        )
        hidden_inputs = generate_inputs(rank, device, num_experts, num_local_experts, topk, num_local_tokens, hidden_dim, intermediate_dim)
        routed_inputs = generate_inputs(rank, device, num_experts, num_local_experts, topk, num_local_tokens, routed_hidden_dim, intermediate_dim)
        x, topk_experts, router_weights, w_shared_gate, w_shared_up, w_shared_down = hidden_inputs[:6]
        routed_input, _, _, _, _, _, w_routed_gate, w_routed_up, w_routed_down, _ = routed_inputs
        schedule = functional.build_schedule(workspace, config, topk_experts, num_local_experts=num_local_experts)
        (y_shared, y_routed), forward_context = functional.forward(
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
            swiglu_limit,
            routed_input=routed_input,
        )
        assert tuple(y_shared.shape) == (num_local_tokens, hidden_dim)
        assert tuple(y_routed.shape) == (num_local_tokens, routed_hidden_dim)
        assert tuple(forward_context.x_routed.shape)[1] == routed_hidden_dim
        reference_combine, reference_gate, reference_up, reference_hidden, reference_y_shared = run_forward_reference_bf16(
            x, topk_experts, w_shared_gate, w_shared_up, w_shared_down, w_routed_gate, w_routed_up, w_routed_down, swiglu_limit, routed_input=routed_input
        )
        check_correctness(f"{label}/y_shared vs reference", reference_y_shared, y_shared, BF16_TOLERANCE, print_stats=rank == 0)
        check_correctness(
            f"{label}/y_routed_topk vs reference",
            run_routed_topk_reduce_reference(reference_combine, router_weights),
            y_routed,
            BF16_TOLERANCE,
            print_stats=rank == 0,
        )
        # Negative control: the routed chain must have consumed routed_input, not x cut or zero-padded to
        # routed_hidden_dim columns (which is what a shared/routed hidden mix-up would produce).
        x_as_routed = (
            x[:, :routed_hidden_dim] if routed_hidden_dim <= hidden_dim else torch.nn.functional.pad(x, (0, routed_hidden_dim - hidden_dim))
        ).contiguous()
        wrong_combine = run_forward_reference_bf16(
            x, topk_experts, w_shared_gate, w_shared_up, w_shared_down, w_routed_gate, w_routed_up, w_routed_down, swiglu_limit, routed_input=x_as_routed
        )[0]
        wrong_routed = run_routed_topk_reduce_reference(wrong_combine, router_weights)
        assert not torch.allclose(
            wrong_routed.float(), y_routed.float(), atol=0.5, rtol=0.01
        ), f"{label}: routed output matches the x[:, :routed_hidden] control; routed_input was not the routed source"
        for name, expected, actual in (
            ("gate_shared", reference_gate, forward_context.gate_shared),
            ("up_shared", reference_up, forward_context.up_shared),
            ("hidden_shared", reference_hidden, forward_context.hidden_shared),
        ):
            check_correctness(f"{label}/{name} vs reference", expected, actual, BF16_TOLERANCE, print_stats=rank == 0)
        # Backward: shared gradient in hidden, routed (top-k reduced output) gradient in the routed hidden.
        grad_shared, routed_grad_output = hidden_inputs[9], routed_inputs[9]
        grads = ungated(
            functional.backward(
                config,
                workspace,
                schedule,
                forward_context,
                grad_shared,
                x,
                router_weights,
                w_shared_gate,
                w_shared_up,
                w_shared_down,
                w_routed_gate,
                w_routed_up,
                w_routed_down,
                swiglu_limit,
                routed_grad_output=routed_grad_output,
                routed_input=routed_input,
            )
        )
        (d_x_shared, d_routed_input), d_router_weights, *weight_grads = grads
        assert tuple(d_x_shared.shape) == (num_local_tokens, hidden_dim)
        assert tuple(d_routed_input.shape) == (num_local_tokens, routed_hidden_dim)
        reference = run_reference_bf16(
            x,
            topk_experts,
            router_weights,
            w_shared_gate,
            w_shared_up,
            w_shared_down,
            w_routed_gate,
            w_routed_up,
            w_routed_down,
            grad_shared,
            swiglu_limit,
            routed_input=routed_input,
            d_routed_output=routed_grad_output,
        )
        (reference_d_x_shared, reference_d_routed_input), *reference_rest = reference[1:]
        for name, expected, actual in (
            ("d_x_shared", reference_d_x_shared, d_x_shared),
            ("d_routed_input", reference_d_routed_input, d_routed_input),
            *zip(BACKWARD_RESULT_NAMES[1:], reference_rest, [d_router_weights, *weight_grads], strict=True),
        ):
            check_correctness(f"{label}/{name} vs reference", expected, actual, BF16_TOLERANCE, print_stats=rank == 0)
        # On this path d_x_shared is the backward's own raw13[0] (no epilogue copy): a second forward + backward
        # with doubled gradients must leave it bitwise unchanged and must not hand out its storage while it is
        # held -- the resident-scratch backward returned the same buffer every call.
        del grads, weight_grads, d_router_weights, d_routed_input, reference, reference_rest, forward_context
        d_x_shared_before = d_x_shared.clone()
        _, forward_context_2 = functional.forward(
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
            swiglu_limit,
            routed_input=routed_input,
        )
        (d_x_shared_2, _), *_ = ungated(
            functional.backward(
                config,
                workspace,
                schedule,
                forward_context_2,
                grad_shared * 2,
                x,
                router_weights,
                w_shared_gate,
                w_shared_up,
                w_shared_down,
                w_routed_gate,
                w_routed_up,
                w_routed_down,
                swiglu_limit,
                routed_grad_output=routed_grad_output * 2,
                routed_input=routed_input,
            )
        )
        torch.cuda.synchronize(device)
        assert torch.equal(d_x_shared, d_x_shared_before), f"{label}: the next backward overwrote d_x_shared"
        assert (
            d_x_shared.untyped_storage().data_ptr() != d_x_shared_2.untyped_storage().data_ptr()
        ), f"{label}: the next backward handed out d_x_shared's storage"
        assert not torch.equal(d_x_shared, d_x_shared_2), f"{label}: doubled gradients gave the same d_x_shared"
        del d_x_shared, d_x_shared_2, d_x_shared_before, forward_context_2


def test_cutedsl_forward_latent_layer_bf16(context: tuple[int, int, torch.device]) -> None:
    """Latent MoE layer (``forward_latent_layer``): x -> latent GEMM -> mega (shared in hidden,
    routed in latent) -> routed top-k reduction -> latent -> hidden GEMM + y_shared, against the
    torch reference of the same composition.  Qwen 4096 -> 1024 (one macrobatch) and DeepSeek
    7168 -> 3584 (multiple macrobatches)."""
    rank, world_size, device = context
    _skip_unless_supported(world_size, device)
    swiglu_param_name, swiglu_limit = swiglu_params()[1]
    minimum_macro = next(p for p in mok_params() if p[0] == "Minimum macrobatch")
    for shape, divisor, mok_param in ((shapes(world_size)[2], 4, mok_params()[0]), (shapes(world_size)[3], 2, minimum_macro)):
        shape_name, num_experts, hidden_dim, intermediate_dim, topk, num_local_tokens = shape
        routed_hidden_dim = hidden_dim // divisor
        num_local_experts = num_experts // world_size
        label = f"cutedsl-latent-layer/{shape_name}/{routed_hidden_dim}/{swiglu_param_name}"
        config = _config(mok_param, world_size, shape)
        workspace = functional.create_workspace(
            config, dist.group.WORLD, device=device, num_local_tokens=num_local_tokens, hidden_size=hidden_dim, topk=topk, routed_hidden_size=routed_hidden_dim
        )
        hidden_inputs = generate_inputs(rank, device, num_experts, num_local_experts, topk, num_local_tokens, hidden_dim, intermediate_dim)
        routed_inputs = generate_inputs(rank, device, num_experts, num_local_experts, topk, num_local_tokens, routed_hidden_dim, intermediate_dim)
        x, topk_experts, router_weights, w_shared_gate, w_shared_up, w_shared_down = hidden_inputs[:6]
        w_routed_gate, w_routed_up, w_routed_down = routed_inputs[6:9]
        generator = torch.Generator(device=device).manual_seed(1234)
        w_latent_down = (torch.randn(routed_hidden_dim, hidden_dim, device=device, generator=generator) / hidden_dim**0.5).to(torch.bfloat16)
        w_latent_up = (torch.randn(hidden_dim, routed_hidden_dim, device=device, generator=generator) / routed_hidden_dim**0.5).to(torch.bfloat16)
        schedule = functional.build_schedule(workspace, config, topk_experts, num_local_experts=num_local_experts)
        output, layer_context = functional.forward_latent_layer(
            config,
            workspace,
            schedule,
            x,
            router_weights,
            w_latent_down,
            w_latent_up,
            w_shared_gate,
            w_shared_up,
            w_shared_down,
            w_routed_gate,
            w_routed_up,
            w_routed_down,
            swiglu_limit,
        )
        assert tuple(output.shape) == (num_local_tokens, hidden_dim)
        routed_input = torch.matmul(x, w_latent_down.t())
        reference_combine, *_, reference_y_shared = run_forward_reference_bf16(
            x, topk_experts, w_shared_gate, w_shared_up, w_shared_down, w_routed_gate, w_routed_up, w_routed_down, swiglu_limit, routed_input=routed_input
        )
        reference_y_routed = run_routed_topk_reduce_reference(reference_combine, router_weights)
        reference = torch.addmm(reference_y_shared, reference_y_routed, w_latent_up.t())
        check_correctness(f"{label}/output vs reference", reference, output, BF16_TOLERANCE, print_stats=rank == 0)
        # Negative control: dropping the routed half must not pass.
        assert not torch.allclose(
            reference_y_shared.float(), output.float(), atol=0.5, rtol=0.01
        ), f"{label}: output equals y_shared alone; the routed half is missing"
        # Backward of the layer: the reference composes the same projection GEMMs around the torch MoE backward.
        grad_output = hidden_inputs[9]
        layer_grads = functional.backward_latent_layer(
            config,
            workspace,
            schedule,
            layer_context,
            grad_output,
            x,
            router_weights,
            w_latent_down,
            w_latent_up,
            w_shared_gate,
            w_shared_up,
            w_shared_down,
            w_routed_gate,
            w_routed_up,
            w_routed_down,
            swiglu_limit,
        )
        routed_grad_output = torch.matmul(grad_output, w_latent_up)
        moe = run_reference_bf16(
            x,
            topk_experts,
            router_weights,
            w_shared_gate,
            w_shared_up,
            w_shared_down,
            w_routed_gate,
            w_routed_up,
            w_routed_down,
            grad_output,
            swiglu_limit,
            routed_input=routed_input,
            d_routed_output=routed_grad_output,
        )
        (reference_d_x_shared, reference_d_routed_input), *moe_rest = moe[1:]
        reference_grads = (
            torch.addmm(reference_d_x_shared, reference_d_routed_input, w_latent_down),
            moe_rest[0],
            torch.matmul(reference_d_routed_input.t(), x),
            torch.matmul(grad_output.t(), reference_y_routed),
            *moe_rest[1:],
        )
        names = ("d_x", "d_router_weights", "d_w_latent_down", "d_w_latent_up", *BACKWARD_RESULT_NAMES[2:])
        for name, expected, actual in zip(names, reference_grads, layer_grads, strict=True):
            check_correctness(f"{label}/{name} vs reference", expected, actual, BF16_TOLERANCE, print_stats=rank == 0)


def test_cutedsl_routed_hidden_rejects_unsupported(context: tuple[int, int, torch.device]) -> None:
    """Differing hiddens: the CuTe DSL forward / backward (BF16 opt and MXFP8) accept them and the CUDA
    ``CuteMoKConfig`` rejects them, the routed tensors are required where accepted, and an ordinary workspace
    rejects routed tensors other than x / grad_output."""
    rank, world_size, device = context
    _skip_unless_supported(world_size, device)
    shape = shapes(world_size)[2]
    shape_name, num_experts, hidden_dim, intermediate_dim, topk, num_local_tokens = shape
    routed_hidden_dim = hidden_dim // 2
    num_local_experts = num_experts // world_size
    config = _config(mok_params()[0], world_size, shape)
    workspace = functional.create_workspace(
        config, dist.group.WORLD, device=device, num_local_tokens=num_local_tokens, hidden_size=hidden_dim, topk=topk, routed_hidden_size=routed_hidden_dim
    )
    inputs = generate_inputs(rank, device, num_experts, num_local_experts, topk, num_local_tokens, hidden_dim, intermediate_dim)
    x, topk_experts, router_weights, *weights, grad_output = inputs
    routed_input = torch.zeros(num_local_tokens, routed_hidden_dim, dtype=torch.bfloat16, device=device)
    schedule = functional.build_schedule(workspace, config, topk_experts, num_local_experts=num_local_experts)
    with pytest.raises(ValueError, match="routed_input is required"):
        functional.forward(config, workspace, schedule, x, router_weights, *weights)
    with pytest.raises(ValueError, match="routed_grad_output must be"):
        ungated(functional.backward(config, workspace, schedule, None, grad_output, x, router_weights, *weights, routed_input=routed_input))
    ordinary = functional.create_workspace(config, dist.group.WORLD, device=device, num_local_tokens=num_local_tokens, hidden_size=hidden_dim, topk=topk)
    ordinary_schedule = functional.build_schedule(ordinary, config, topk_experts, num_local_experts=num_local_experts)
    with pytest.raises(ValueError, match="only accepted when"):
        functional.forward(config, ordinary, ordinary_schedule, x, router_weights, *weights, routed_input=x.clone())
    with pytest.raises(ValueError, match="only accepted when"):
        ungated(
            functional.backward(config, ordinary, ordinary_schedule, None, grad_output, x, router_weights, *weights, routed_grad_output=grad_output.clone())
        )
    with pytest.raises(ValueError, match="latent layer requires"):
        functional.forward_latent_layer(config, ordinary, ordinary_schedule, x, router_weights, weights[0], weights[0], *weights)
    latent_down = torch.zeros(routed_hidden_dim, hidden_dim, dtype=torch.bfloat16, device=device)
    latent_up = torch.zeros(hidden_dim, routed_hidden_dim, dtype=torch.bfloat16, device=device)
    with pytest.raises(TypeError, match="config must be a CuteMoKConfig"):
        functional.forward_latent_layer(object(), workspace, schedule, x, router_weights, latent_down, latent_up, *weights)
    with pytest.raises(TypeError, match="config must be a CuteMoKConfig"):
        functional.backward_latent_layer(object(), workspace, schedule, None, grad_output, x, router_weights, latent_down, latent_up, *weights)
    # an ordinary workspace accepts the routed tensors when they are x / grad_output themselves
    output, forward_context = functional.forward(config, ordinary, ordinary_schedule, x, router_weights, *weights, routed_input=x)
    assert isinstance(output, torch.Tensor) and tuple(output.shape) == (num_local_tokens, hidden_dim)
    grads = ungated(
        functional.backward(
            config, ordinary, ordinary_schedule, forward_context, grad_output, x, router_weights, *weights, routed_grad_output=grad_output, routed_input=x
        )
    )
    assert isinstance(grads[0], torch.Tensor) and tuple(grads[0].shape) == (num_local_tokens, hidden_dim)


def test_config_contract(context: tuple[int, int, torch.device]) -> None:
    """CuteMoKConfig is the one public config (MoKConfig is its alias), and MCore's assumptions about it hold.

    MCore builds it from a user JSON blob (`MoKConfig(**options)`), reads none of its attributes,
    and turns a TypeError into a configuration error — so an unknown keyword must raise, and every
    field name here is part of that contract.
    """
    rank, world_size, device = context
    _skip_unless_supported(world_size, device)
    assert functional.MoKConfig is functional.CuteMoKConfig
    for unknown in (dict(fwd_backend="cutedsl"), dict(bwd_backend="cuda"), dict(bwd_schedule="minibatch"), dict(bogus=1)):
        with pytest.raises(TypeError):
            functional.CuteMoKConfig(**unknown)
        with pytest.raises(TypeError):
            dataclasses.replace(functional.CuteMoKConfig(), **unknown)
    # minibatch_size serves both directions unless a direction overrides it.
    shared = functional.CuteMoKConfig(minibatch_size=4096)
    assert (functional._fwd_minibatch_size(shared), functional._bwd_minibatch_size(shared)) == (4096, 4096)
    split = functional.CuteMoKConfig(minibatch_size=4096, bwd_minibatch_size=8192)
    assert (functional._fwd_minibatch_size(split), functional._bwd_minibatch_size(split)) == (4096, 8192)
    # bwd_num_comm_sms serves every backward unless the one that replays the first macrobatch too has its own.
    assert [functional._bwd_num_comm_sms(shared, recompute) for recompute in (False, True)] == [28, 28]
    own = functional.CuteMoKConfig(bwd_recompute_num_comm_sms=36)
    assert [functional._bwd_num_comm_sms(own, recompute) for recompute in (False, True)] == [28, 36]
    shape = next(s for s in shapes(world_size) if s[0] == "Local tokens 16384")
    _, num_experts, hidden_dim, intermediate_dim, topk, num_local_tokens = shape
    num_local_experts = num_experts // world_size
    config = _config(mok_params()[0], world_size, shape)
    workspace = get_workspace(config, dist.group.WORLD, device=device, num_local_tokens=num_local_tokens, hidden_size=hidden_dim, topk=topk)
    inputs = generate_inputs(rank, device, num_experts, num_local_experts, topk, num_local_tokens, hidden_dim, intermediate_dim)
    x, topk_experts, router_weights, *weights, d_output = inputs
    schedule = functional.build_schedule(workspace, config, topk_experts, num_local_experts=num_local_experts)
    with pytest.raises(TypeError, match="config must be"):
        functional.build_schedule(workspace, object(), topk_experts, num_local_experts=num_local_experts)
    with pytest.raises(TypeError, match="epilogue_combine must be a bool"):
        functional.build_schedule(workspace, dataclasses.replace(config, epilogue_combine=1), topk_experts, num_local_experts=num_local_experts)
    for invalid in (3, 0, 4096):  # odd, not positive, no compute SM left
        with pytest.raises(ValueError, match="bwd_recompute_num_comm_sms|communication-SM counts"):
            functional.build_schedule(
                workspace, dataclasses.replace(config, bwd_recompute_num_comm_sms=invalid), topk_experts, num_local_experts=num_local_experts
            )
    with pytest.raises(ValueError, match="comm_tile_shape must be"):
        functional.build_schedule(workspace, dataclasses.replace(config, comm_tile_shape=(64, 256)), topk_experts, num_local_experts=num_local_experts)
    # The option is a variant of the BF16 forward: the MXFP8 forward (routed weights as e4m3 / scale tuples) refuses it.
    routed_mxfp8 = [tuple(mxfp8_quantize(weight, True, False))[:2] for weight in weights[3:]]
    with pytest.raises(NotImplementedError, match="epilogue_combine is implemented for the BF16 forward only"):
        functional.forward(dataclasses.replace(config, epilogue_combine=True), workspace, schedule, x, router_weights, *weights[:3], *routed_mxfp8)
    # So is the communication tile shape (BF16 megakernels only): MXFP8 refuses anything but the default.
    with pytest.raises(ValueError, match="comm_tile_shape is an option of the BF16 megakernels"):
        functional.forward(dataclasses.replace(config, comm_tile_shape=(16, 1024)), workspace, schedule, x, router_weights, *weights[:3], *routed_mxfp8)


# --------------------------------------------------------------------------------------------------------------------
# MXFP8 backward
#
# GPU-only EP8 MXFP8 backward through its low-level entry and the public one.
#
# Run under torchrun on SM100 / SM103 (8 GPUs), for both Gate / Up context precisions
# (``gate_up_context="mxfp8"``: e4m3 / scale pairs; ``"bf16"``: one BF16 tensor each).  Each geometry
# runs the MXFP8 forward for its context, then ``backward_mxfp8`` on a fresh copy of that context
# (the replay overwrites its routed part) and the public ``functional.backward``.  Every raw output
# and the two symmetric buffers the peers write must be written and finite, the communication
# counters must hold the arrivals the kernel's waits expect, and the public gradients must be as
# close to the BF16 PyTorch reference as upstream Mixture-of-Kittens' (``MIN_COSINE``, ``NOT_WORSE``,
# ``UPSTREAM``).  The latent layer is compared byte for byte
# with the ordinary backward of each hidden, a repeated call with itself, and two layers launched
# back to back each with its own reference.  The forward's geometries run through one compiled
# executor per clamp mode, with a clamp limit the inputs reach.  ``recompute_forward_context`` (macrobatch 0 replayed
# instead of read from the forward's routed context) is compared byte for byte with the backward on the
# saved context.


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
# and nowhere else: those entries are the only ones that lower a bar.  The values were measured by running
# upstream on the same inputs (seed 1234).
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
    cached_before = set(mxfp8_bwd_host._EXECUTOR_CACHE)  # other tests of this module may have compiled this variant already
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
    # Sizes are runtime arguments: at most one new executor per clamp mode and Gate / Up context, whatever the geometry.
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
        assert (
            sum(
                ("is_clamped", is_clamped) in key and all(axis in key for axis in this_variant)
                for key in mxfp8_bwd_host._EXECUTOR_CACHE
                if key not in cached_before
            )
            <= 1
        )


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
    # Other tests of this module leave free blocks of the tensormaps buffer's size in the caching allocator; release
    # them so that the second backward is handed the block the first one freed.
    torch.cuda.synchronize(device)
    torch.cuda.empty_cache()
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


# --------------------------------------------------------------------------------------------------------------------
# BF16 backward task decode variants
#
# The three BWD task->expert decode variants (TaskDecode) give the same backward.
#
# Run under torchrun on SM100 / SM103 (any world size; EP1 covers 512 local experts), e.g.
# ``torchrun --standalone --nproc-per-node=1 -m pytest -s moe_dispatch_mlp_combine/cutedsl/test_backward.py``.
# For every shape the auto-selected variant is checked against the PyTorch BF16 reference
# (``run_reference_bf16``) with the BF16 tolerance -- the CUDA backend it was once compared with is
# gone, and comparing the port with a second run of itself checked nothing -- and every valid forced
# variant (SMEM_BINARY only when the local experts fit the
# shared-memory table) must reproduce the auto variant's gradients bit for bit.  Every backward runs
# on a fresh forward context of the same CuTe DSL forward (the routed Down dgrad epilogue overwrites
# the context's hidden_routed with w * Hidden, so a context serves exactly one backward); the forward's
# own decode is not forced here.
#
# ``test_bwd_routed_dgrad_partial_blocks`` routes a fixed number of rows to every local expert so that
# the experts' last 256-row blocks hold 1, 15, 17, 59, 64, 100, 127, 129, 255 and 1 (257 rows) valid rows: the
# swap-AB routed Gate/Up dgrad's N (align16 of the valid rows, at least 16, below and above the 128-row
# cta_group::2 split, exactly 256) and the Wgrad K skip at a non-multiple of 64 are all exercised, and a
# wrong N, a wrong non-leader B offset or an unstored valid dX row shows up against the reference.


GRADIENT_NAMES = ("d_x", "d_router_weights", "d_w_routed_gate", "d_w_routed_up", "d_w_routed_down", "d_w_shared_gate", "d_w_shared_up", "d_w_shared_down")


# (name, local experts, experts that receive tokens (per rank), hidden, intermediate, top-k, local tokens,
#  macrobatch).  Minibatch 2048 = 8 row blocks.  Coverage of the BWD decode:
#   stride 1 (E <= 512 = BWD_EXPERT_TABLE_CAP, SMEM_BINARY auto): 512 (one entry per expert at the capacity),
#   383 and 257 (above the forward's 256), 256, 128 and 96, 64 (~2 row blocks per expert) and 16 (~8 row blocks
#   per expert, so experts span minibatch boundaries).  All of the larger shapes use H = 4096 (16 column blocks for
#   the Gate/Up dgrad) or I = 2048 (8 column blocks for Down dgrad; 16 for the 6144/2048 shape): the tile swizzle
#   (supergroups of 8 columns) only differs from row-major order with more than 8 column blocks, and only then does
#   a wrong row-block origin or row count inside a minibatch move tiles.  16 experts with 1024 tokens fit one
#   macrobatch of 16384 rows, which takes the kernel's macro fallback (dense per-expert Wgrad grid, generation
#   decode without the owner buckets); every other case has several macrobatches (minibatch release, replay).
#   stride 2 with tokens for the first 100 experts only and empty groups (600), and a last group of one (769);
#   stride 3 with empty groups (1100, tokens for the first 200 only).  Empty experts at stride 1: 128 with tokens
#   for the first 100 only and 383 with tokens for the first 300 only (EP1 only, local expert indices).  The decode
#   variant is the only compile-time input: after every case each variant has compiled at most one executor (the
#   expert count and the sizes are runtime values).
CASES = (
    ("512 local experts (Qwen3.5 shape, stride 1 at the BWD table capacity)", 512, 512, 4096, 1024, 10, 4096, 8192),
    ("256 local experts (hidden 6144, stride 1)", 256, 256, 6144, 2048, 8, 2048, 8192),
    ("128 local experts, 16 column blocks (stride 1, above the old BWD cap)", 128, 128, 4096, 2048, 8, 4096, 8192),
    ("96 local experts, 16 column blocks (stride 1, above the old BWD cap)", 96, 96, 4096, 2048, 8, 4096, 8192),
    ("64 local experts, ~2 row blocks each, 16 column blocks (stride 1)", 64, 64, 4096, 2048, 8, 4096, 8192),
    ("128 local experts, experts 100..127 empty, 16 column blocks (stride 1)", 128, 100, 4096, 2048, 8, 4096, 8192),
    ("16 local experts, ~8 row blocks each, spanning minibatches, 16 column blocks (stride 1)", 16, 16, 4096, 2048, 8, 4096, 8192),
    ("16 local experts, one macrobatch (macro fallback: dense Wgrad grid), 16 column blocks", 16, 16, 4096, 2048, 8, 1024, 16384),
    ("257 local experts (stride 1, above the forward's table)", 257, 257, 256, 256, 8, 4096, 8192),
    ("383 local experts, experts 300..382 empty (stride 1)", 383, 300, 256, 256, 8, 4096, 8192),
    ("600 local experts, tokens for the first 100 only (stride 2, multi-block experts, empty groups)", 600, 100, 256, 256, 8, 4096, 8192),
    ("769 local experts (stride 2, last group of one)", 769, 769, 256, 256, 8, 4096, 8192),
    ("1100 local experts, tokens for the first 200 only (stride 3, empty groups)", 1100, 200, 256, 256, 8, 4096, 8192),
)


def _backward_with_task_decode(config, workspace, schedule, forward_context, inputs, task_decode):
    """functional.backward with the BWD task->expert decode forced to ``task_decode`` (None = the
    host's choice) by patching the BWD module's selector, the only selection point of backward_bf16."""
    x, _, router_weights, w_shared_gate, w_shared_up, w_shared_down, w_routed_gate, w_routed_up, w_routed_down, d_output = inputs
    select = persistent_bf16_bwd.select_task_decode
    forced = select if task_decode is None else (lambda num_local_experts, requested=None: select(num_local_experts, task_decode))
    with mock.patch.object(persistent_bf16_bwd, "select_task_decode", forced):
        grads = ungated(
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
                None,
            )
        )
    torch.cuda.synchronize(x.device)  # a device fault is reported against this call, not a later one
    return tuple(grads)


def test_shared_storage_layout(context: tuple[int, int, torch.device]) -> None:
    """The expert table lies in the arena tail, after every compute-role staging area (GEMM A/B/D)
    and inside the arena; the storage itself is barriers + arena (the module asserts both at
    import).  Print the layout so the placement is visible."""
    rank, _, device = context
    if torch.cuda.get_device_capability(device) not in SUPPORTED_CAPABILITIES:
        pytest.skip("CuTe DSL backends are SM100/SM103 only")
    from cudnn.moe_dispatch_mlp_combine.cutedsl import _persistent_bf16_bwd_mega as mega

    layout = mega.shared_storage_layout()
    by_name = {name: (offset, size) for name, offset, size in layout}
    arena_offset = by_name["arena"][0]
    table_begin = mega.EXPERT_TABLE_ARENA_OFFSET
    table_end = table_begin + BWD_EXPERT_TABLE_CAP * 4
    assert "expert_valid_end" not in by_name
    assert table_begin >= mega.GEMM_ARENA_BYTES
    assert table_end <= mega.REUSABLE_ARENA_BYTES and arena_offset % 1024 == 0
    assert mega.DYNAMIC_SHARED_MEMORY_BYTES == arena_offset + mega.REUSABLE_ARENA_BYTES < 232448
    # The decoded-task record ring (CLC warp -> other warps): every field has one Int32 word, the
    # stages are 16-byte aligned and lie with the barriers before the arena; the ten geometry words
    # are the _gemm_geometry tuple in its order.
    record_offset, record_bytes = by_name["task_info"]
    assert len(set(mega.TASK_INFO_FIELDS)) == len(mega.TASK_INFO_FIELDS) <= mega.TASK_INFO_WORDS
    assert record_bytes == mega.TASK_INFO_STAGES * mega.TASK_INFO_WORDS * 4 and record_offset % 16 == 0
    assert record_offset + record_bytes <= arena_offset
    assert mega.TASK_INFO_GEOMETRY_FIELDS == (
        "valid",
        "expert",
        "row",
        "column",
        "k_begin",
        "k_tiles",
        "wgrad_first",
        "wgrad_has_next",
        "wgrad_empty",
        "tile_valid",
    )
    if rank == 0:
        for name, offset, size in layout:
            print(f"SMEM {name}: offset {offset} size {size} end {offset + size}")
        print(
            f"SMEM expert table: arena [{table_begin}, {table_end}) of {mega.REUSABLE_ARENA_BYTES}; "
            f"dynamic shared memory {mega.DYNAMIC_SHARED_MEMORY_BYTES} B"
        )


def test_bwd_task_decode_variants_agree(context: tuple[int, int, torch.device]) -> None:
    rank, world_size, device = context
    if torch.cuda.get_device_capability(device) not in SUPPORTED_CAPABILITIES:
        pytest.skip("CuTe DSL backends are SM100/SM103 only")
    # The count below is over the whole executor cache, so this test has to own it: another test
    # file in the same session leaves backward executors of its own behind.
    persistent_bf16_bwd.clear_persistent_bf16_bwd_executor_cache()
    failures = []  # every case runs; the mismatches are reported together at the end
    for name, num_local_experts, routed_local_experts, hidden_dim, intermediate_dim, topk, num_local_tokens, macrobatch_size in CASES:
        if routed_local_experts != num_local_experts and world_size != 1:
            continue  # restricted routing is expressed in local expert indices
        num_experts = num_local_experts * world_size
        # Schedule capacity: every local expert is padded to a 256-row block, so the padded row count
        # can reach E_local * 256 + routed rows (the CUDA scheduler traps above the capacity).  The
        # capacity factor is max(2, ceil(world * multiplier)) times T * top-k.
        routed_rows = num_local_tokens * topk
        capacity_factor = -(-(num_local_experts * 256 + routed_rows * world_size) // routed_rows) + 1
        geometry = dict(fwd_num_comm_sms=30, bwd_num_comm_sms=14, macrobatch_size=macrobatch_size, schedule_capacity_multiplier=capacity_factor / world_size)
        config = functional.CuteMoKConfig(minibatch_size=2048, **geometry)
        workspace = get_workspace(config, dist.group.WORLD, device=device, num_local_tokens=num_local_tokens, hidden_size=hidden_dim, topk=topk)
        # Routing over the first routed_local_experts * world_size experts only; the weights
        # cover all num_local_experts, so the remaining experts receive no rows.
        inputs = generate_inputs(rank, device, routed_local_experts * world_size, num_local_experts, topk, num_local_tokens, hidden_dim, intermediate_dim)
        x, topk_experts, router_weights, w_shared_gate, w_shared_up, w_shared_down, w_routed_gate, w_routed_up, w_routed_down, _ = inputs
        dsl_schedule = functional.build_schedule(workspace, config, topk_experts, num_local_experts=num_local_experts)

        auto = persistent_bf16_bwd.select_task_decode(num_local_experts)
        expected_auto = TaskDecode.SMEM_BINARY if num_local_experts <= BWD_EXPERT_TABLE_CAP else TaskDecode.SMEM_GROUP
        assert auto is expected_auto, (name, num_local_experts, auto)
        stride = -(-num_local_experts // BWD_EXPERT_TABLE_CAP)
        forward_args = (x, router_weights, w_shared_gate, w_shared_up, w_shared_down, w_routed_gate, w_routed_up, w_routed_down, None)
        baseline_grads = run_reference_bf16(*inputs)[1:]

        def dsl_backward(task_decode):
            _, dsl_context = functional.forward(config, workspace, dsl_schedule, *forward_args)
            torch.cuda.synchronize(device)
            return [t.clone() for t in _backward_with_task_decode(config, workspace, dsl_schedule, dsl_context, inputs, task_decode)]

        reference = dsl_backward(None)
        for grad_name, want, got in zip(GRADIENT_NAMES, baseline_grads, reference, strict=True):
            check_correctness(f"{name}/auto={auto.name} {grad_name} vs reference", want, got, BF16_TOLERANCE, print_stats=rank == 0)
        for variant in TaskDecode:
            if variant is TaskDecode.SMEM_BINARY and num_local_experts > BWD_EXPERT_TABLE_CAP:
                with pytest.raises(ValueError):
                    persistent_bf16_bwd.select_task_decode(num_local_experts, variant)
                continue
            result = dsl_backward(variant)
            for grad_name, want, got, baseline_grad in zip(GRADIENT_NAMES, reference, result, baseline_grads, strict=True):
                if not torch.equal(want, got):
                    # Which side is off: the variant's distance to the reference next to the auto variant's.
                    differing = int((want != got).sum())
                    err = lambda t: float((t.float() - baseline_grad.float()).abs().max())
                    failures.append(
                        f"{name}/{variant.name}: {grad_name} differs from {auto.name} in {differing} of "
                        f"{got.numel()} elements (max |diff| {float((want.float() - got.float()).abs().max()):.3e}); "
                        f"max |err| vs reference: {auto.name} {err(want):.3e}, {variant.name} {err(got):.3e}"
                    )
            if rank == 0:
                print(f"{name}: BWD {variant.name} vs {auto.name} checked (E_local={num_local_experts}, stride {stride})")
        del workspace, inputs, reference, result, baseline_grads
        torch.cuda.empty_cache()
    # Grouped executors only: the routed-weight layout is another compile-time axis, so a split
    # test earlier in the session would otherwise make every variant look compiled twice.
    compiled = [
        part[1]
        for key in persistent_bf16_bwd._EXECUTOR_CACHE
        if ("routed_weights", "grouped") in key
        for part in key
        if isinstance(part, tuple) and len(part) == 2 and part[0] == "task_decode"
    ]
    assert compiled and all(compiled.count(variant) == 1 for variant in set(compiled)), compiled
    assert not failures, "\n".join(failures)


# Routed rows per local expert of the partial-block case (sum = 2048 tokens x top-k 8; each count is at most
# the token count so a token never routes to the same expert twice).  Valid rows of each expert's last
# 256-row block: 1, 15, 17, 100, 127, 129, 255, 1 (257), 59 (1595), 64 (1600) and 256 (2048, no padding).
PARTIAL_BLOCK_COUNTS = (1, 15, 17, 100, 127, 129, 255, 257, 1595, 1600, 2048, 2048, 2048, 2048, 2048, 2048)


# (name, macrobatch): three macrobatches (18176 padded rows = 71 row blocks, 9 minibatches of 2048: replay,
# ring slot reuse) and one macrobatch (CUDA's macro fallback).
PARTIAL_BLOCK_CASES = (("partial last blocks, 3 macrobatches", 8192), ("partial last blocks, 1 macrobatch", 32768))


def _routing_with_counts(rank: int, device: torch.device, num_local_experts: int, counts, topk: int, num_local_tokens: int) -> torch.Tensor:
    """top-k expert indices (num_local_tokens, topk) that route exactly ``counts[e]`` rows of this rank's
    tokens to this rank's local expert ``e`` (global index rank * E_local + e; no row goes to another rank, so
    every rank's local counts are known).  The sorted expert list is laid out so that token t takes
    positions t, t + T, ...: a run of one expert is at most T long, so a token never repeats an expert."""
    counts = torch.tensor(counts, dtype=torch.int64)
    assert len(counts) == num_local_experts and int(counts.sum()) == topk * num_local_tokens
    assert int(counts.max()) <= num_local_tokens
    flat = torch.repeat_interleave(torch.arange(num_local_experts, dtype=torch.int64), counts)
    topk_experts = flat.view(topk, num_local_tokens).t().contiguous() + rank * num_local_experts
    return topk_experts.to(device)


def test_bwd_routed_dgrad_partial_blocks(context: tuple[int, int, torch.device]) -> None:
    rank, world_size, device = context
    if torch.cuda.get_device_capability(device) not in SUPPORTED_CAPABILITIES:
        pytest.skip("CuTe DSL backends are SM100/SM103 only")
    num_local_experts, topk, num_local_tokens, hidden_dim, intermediate_dim = len(PARTIAL_BLOCK_COUNTS), 8, 2048, 512, 512
    num_experts = num_local_experts * world_size
    for name, macrobatch_size in PARTIAL_BLOCK_CASES:
        routed_rows = num_local_tokens * topk
        capacity_factor = -(-(num_local_experts * 256 + routed_rows * world_size) // routed_rows) + 1
        geometry = dict(fwd_num_comm_sms=30, bwd_num_comm_sms=14, macrobatch_size=macrobatch_size, schedule_capacity_multiplier=capacity_factor / world_size)
        config = functional.CuteMoKConfig(minibatch_size=2048, **geometry)
        workspace = get_workspace(config, dist.group.WORLD, device=device, num_local_tokens=num_local_tokens, hidden_size=hidden_dim, topk=topk)
        inputs = list(generate_inputs(rank, device, num_experts, num_local_experts, topk, num_local_tokens, hidden_dim, intermediate_dim))
        inputs[1] = _routing_with_counts(rank, device, num_local_experts, PARTIAL_BLOCK_COUNTS, topk, num_local_tokens)
        inputs = tuple(inputs)
        x, topk_experts, router_weights, w_shared_gate, w_shared_up, w_shared_down, w_routed_gate, w_routed_up, w_routed_down, _ = inputs
        dsl_schedule = functional.build_schedule(workspace, config, topk_experts, num_local_experts=num_local_experts)
        # The routing landed as intended: the padded and the unpadded counts.
        assert tuple(int(v) for v in dsl_schedule.tokens_per_expert.tolist()) == tuple(-(-c // 256) * 256 for c in PARTIAL_BLOCK_COUNTS), name
        assert tuple(int(v) for v in dsl_schedule.valid_tokens_per_expert.tolist()) == PARTIAL_BLOCK_COUNTS, name
        forward_args = (x, router_weights, w_shared_gate, w_shared_up, w_shared_down, w_routed_gate, w_routed_up, w_routed_down, None)
        baseline_grads = run_reference_bf16(*inputs)[1:]
        _, dsl_context = functional.forward(config, workspace, dsl_schedule, *forward_args)
        torch.cuda.synchronize(device)
        dsl_grads = _backward_with_task_decode(config, workspace, dsl_schedule, dsl_context, inputs, None)
        for grad_name, want, got in zip(GRADIENT_NAMES, baseline_grads, dsl_grads, strict=True):
            assert torch.isfinite(got.float()).all(), f"{name}: {grad_name} has non-finite values"
            check_correctness(f"{name} {grad_name} vs reference", want, got, BF16_TOLERANCE, print_stats=rank == 0)
        del workspace, inputs, baseline_grads, dsl_grads, dsl_context
        torch.cuda.empty_cache()


# --------------------------------------------------------------------------------------------------------------------
# MXFP8 backward task decode variants
#
# The MXFP8 backward's task -> expert decode variants (TaskDecode) give the same bytes.
#
# Run under torchrun on SM100 / SM103 (8 GPUs), like ``test_backward.py`` whose entries it
# reuses.  The MXFP8 backward compiles one kernel per decode variant (the BF16 forward's
# ``TaskDecode``: the per-CTA expert table searched per expert for E <= EXPERT_TABLE_CAP local
# experts, per expert group above it, or the linear scan) for the routed dgrad / replay owner decode and
# the routed Wgrad K ranges.  For every geometry the host's choice is checked (``_assert_backward``) and
# every forced variant must reproduce it byte for byte (the eighteen raw outputs, the peer-written buffers
# and the eight public gradients).  The cases reach 257 and 384 local experts (group stride 2, a short and a
# full last group, the group walk of the Wgrad K range).  A variant is forced by patching the host's selector,
# the only selection point of ``persistent_mxfp8_bwd``.  The 9-column-block geometry has about two row blocks
# per expert, so the tile swizzle differs from row-major order and a wrong expert start row or row count
# would move tiles.  Each geometry's executors -- the backward's and the forward's -- are cleared afterwards,
# so the other MXFP8 tests' executor counts do not depend on test order.  Both Gate / Up context precisions
# run (``gate_up_context``, as in ``test_backward.py``).


# (name, replayed macrobatches expected, (experts, hidden, intermediate, top-k, local tokens, comm SMs,
# minibatch, macrobatch)); the local expert count is experts / world size (8).  The schedule's
# capacity is 1.5 x 8 x tokens x top-k rows per rank; every local expert is padded to 256 rows.
MXFP8_CASES = (
    ("2 local experts, one macrobatch", False, (16, 512, 512, 2, 512, 8, 512, 4096)),
    ("64 local experts (Qwen-sized), replay", True, (512, 1024, 512, 4, 2048, 16, 1024, 2048)),
    ("16 local experts, 9 column blocks, about two row blocks per expert, replay", True, (128, 2304, 2304, 4, 2048, 16, 1024, 4096)),
    ("256 local experts (the table's capacity, stride 1), replay", True, (2048, 512, 512, 2, 4096, 16, 1024, 8192)),
    ("257 local experts (stride 2, last group of one), replay", True, (2056, 512, 512, 2, 4096, 16, 1024, 8192)),
    ("384 local experts (stride 2, full last group), replay", True, (3072, 512, 512, 4, 4096, 16, 1024, 8192)),
)


def _backward_with_decode(rank, world_size, device, geometry, task_decode, gate_up_context="bf16"):
    """``_backward`` with the MXFP8 backward's decode forced to ``task_decode`` (None = the host's choice);
    returns (raw, gradients, reference, rows, macrobatches)."""
    select = mxfp8_bwd_host.select_task_decode
    if task_decode is None:
        forced = select
    else:

        def forced(num_local_experts, requested=None):
            return select(num_local_experts, task_decode)

    with mock.patch.object(mxfp8_bwd_host, "select_task_decode", forced):
        return _backward(rank, world_size, device, geometry, None, gate_up_context)


def _assert_same_dsl(name, want, got, rows):
    raw_want, gradients_want = want
    raw_got, gradients_got = got
    for output, expected, actual in zip(RAW_OUTPUTS, raw_want[0], raw_got[0], strict=True):
        assert torch.equal(_bytes(_written(output, expected, rows)), _bytes(_written(output, actual, rows))), f"{name}: {output}"
    for expected, actual in zip(raw_want[1], raw_got[1], strict=True):
        assert torch.equal(_bytes(expected), _bytes(actual)), f"{name}: peer buffer"
    for index, (expected, actual) in enumerate(zip(gradients_want, gradients_got, strict=True)):
        assert torch.equal(_bytes(expected), _bytes(actual)), f"{name}: gradient {index}"


@pytest.mark.parametrize("gate_up_context", ("bf16", "mxfp8"))
def test_cutedsl_mxfp8_backward_task_decode_variants_agree(context, gate_up_context):
    rank, world_size, device = context
    if world_size != 8:
        pytest.skip("the MXFP8 backward geometries here are EP8")
    if torch.cuda.get_device_capability(device) not in SUPPORTED_CAPABILITIES:
        pytest.skip("the CuTe DSL MXFP8 backward is SM100/SM103 only")
    mxfp8_bwd_host.clear_persistent_mxfp8_bwd_executor_cache()
    mxfp8_fwd_host.clear_persistent_mxfp8_executor_cache()
    for name, replay, geometry in MXFP8_CASES:
        local_experts = geometry[0] // world_size
        fits_table = local_experts <= EXPERT_TABLE_CAP
        auto = mxfp8_bwd_host.select_task_decode(local_experts)
        expected = TaskDecode.SMEM_BINARY if fits_table else TaskDecode.SMEM_GROUP
        assert auto is expected, (name, local_experts, auto)
        raw, gradients, reference, rows, macrobatches = _backward_with_decode(rank, world_size, device, geometry, None, gate_up_context)
        assert (macrobatches > 1) == replay, (name, macrobatches)
        # The backward test's check; upstream's bar where UPSTREAM measured the geometry (only the first case).
        _assert_backward(raw, gradients, reference, rows, geometry, None, gate_up_context, upstream=(geometry, None) in UPSTREAM)
        for variant in TaskDecode:
            if variant is auto:
                continue
            if variant is TaskDecode.SMEM_BINARY and not fits_table:
                with pytest.raises(ValueError):
                    mxfp8_bwd_host.select_task_decode(local_experts, variant)
                continue
            result = _backward_with_decode(rank, world_size, device, geometry, variant, gate_up_context)
            _assert_same_dsl(f"{name}/{variant.name} vs {auto.name}", (raw, gradients), result[:2], rows)
            if rank == 0:
                print(f"{name}: {variant.name} == {auto.name} (E_local={local_experts}, macrobatches={macrobatches})")
        mxfp8_bwd_host.clear_persistent_mxfp8_bwd_executor_cache()
        mxfp8_fwd_host.clear_persistent_mxfp8_executor_cache()
        torch.cuda.empty_cache()


# --------------------------------------------------------------------------------------------------------------------
# BF16 backward scratch
#
# The CuTe DSL BF16 backward owns its macrobatch-sized scratch per call.
#
# Run under torchrun on SM100 / SM103 at any world size, e.g.
# ``torchrun --standalone --nproc-per-node=4 -m pytest -s moe_dispatch_mlp_combine/cutedsl/test_backward.py``.
# The CUDA backward allocates its temporaries (d_x_*, d_gate_*, d_up_*, d_hidden_*, d_y_routed) with
# ``at::empty`` on every call and frees them on return, so a forward that follows can reuse that memory.
# The CuTe DSL backward must behave the same: after ``backward`` returns, the workspace keeps only its
# small buffers (readiness counters and decoder metadata, schedule-capacity sized; router weights and
# partials, macrobatch-sized), no activation-sized scratch; its cached launch plans hold no scratch
# address; repeated calls, on the default stream and on a second stream, give bit-identical gradients;
# and a later call with other gradients neither overwrites nor reuses the storage of an earlier call's
# outputs.  Resident scratch itself is caught by the retained-memory check and, on the split-hidden path
# where d_x is the backward's own scratch, by test_cutedsl_routed_hidden_bf16.


# A generic shape with several macrobatches (T * top-k = 4 macrobatches of 8192 rows): the scratch of
# one call (raw13 0-6) is ~134 MB, two orders of magnitude above the buffers the workspace may keep.
HIDDEN, INTERMEDIATE, TOPK, TOKENS, LOCAL_EXPERTS = 2048, 1024, 8, 4096, 32


MACROBATCH, MINIBATCH = 8192, 2048


def _step(config, workspace, schedule, inputs):
    """One forward + backward on a fresh forward context (a context serves exactly one backward)."""
    x, _, router_weights, w_shared_gate, w_shared_up, w_shared_down, w_routed_gate, w_routed_up, w_routed_down, d_output = inputs
    weights = (w_shared_gate, w_shared_up, w_shared_down, w_routed_gate, w_routed_up, w_routed_down)
    output, forward_context = functional.forward(config, workspace, schedule, x, router_weights, *weights, None)
    grads = ungated(functional.backward(config, workspace, schedule, forward_context, d_output, x, router_weights, *weights, None))
    return output, forward_context, tuple(grads)


def _tensors(value):
    if isinstance(value, torch.Tensor):
        yield value
    elif isinstance(value, (tuple, list)):
        for item in value:
            yield from _tensors(item)


def test_cutedsl_backward_scratch_is_per_call(context: tuple[int, int, torch.device]) -> None:
    rank, world_size, device = context
    if torch.cuda.get_device_capability(device) not in SUPPORTED_CAPABILITIES:
        pytest.skip("CuTe DSL backends are SM100/SM103 only")
    num_experts = LOCAL_EXPERTS * world_size
    config = functional.CuteMoKConfig(
        macrobatch_size=MACROBATCH, fwd_minibatch_size=MINIBATCH, bwd_minibatch_size=MINIBATCH, fwd_num_comm_sms=30, bwd_num_comm_sms=14
    )
    inputs = generate_inputs(rank, device, num_experts, LOCAL_EXPERTS, TOPK, TOKENS, HIDDEN, INTERMEDIATE)
    # A workspace of its own (not the cached one): its first backward is the one measured below.
    workspace = functional.create_workspace(config, dist.group.WORLD, device=device, num_local_tokens=TOKENS, hidden_size=HIDDEN, topk=TOPK)
    schedule = functional.build_schedule(workspace, config, inputs[1], num_local_experts=LOCAL_EXPERTS)

    # 1. Memory: what the workspace keeps after a backward is smaller than the smallest scratch tensor
    #    (d_gate_shared / d_up_shared, TOKENS x INTERMEDIATE bf16), i.e. no scratch stays resident;
    #    and the call's peak over the forward's level stays below the thirteen contract tensors plus the public
    #    outputs: the SwiGLU backward keeps dH in registers since the fused epilogues, so no d_hidden_shared
    #    (TOKENS x INTERMEDIATE) / d_hidden_routed (MACROBATCH x INTERMEDIATE) buffer may be allocated per call.
    x, _, router_weights, *weights, d_output = inputs
    _, forward_context = functional.forward(config, workspace, schedule, x, router_weights, *weights, None)
    torch.cuda.synchronize(device)
    before = torch.cuda.memory_allocated(device)
    torch.cuda.reset_peak_memory_stats(device)
    grads = ungated(functional.backward(config, workspace, schedule, forward_context, d_output, x, router_weights, *weights, None))
    torch.cuda.synchronize(device)
    held = sum(t.numel() * t.element_size() for t in _tensors(grads))
    descriptors = raw13_descriptors_for(workspace, config)
    assert len(descriptors) == len(RAW13_NAMES) == 13 and not any("hidden" in d.name for d in descriptors)
    contract_bytes = sum(2 * math.prod(d.shape) for d in descriptors)
    public_bytes = held - sum(2 * math.prod(d.shape) for d in descriptors[7:])  # d_x, d_router_weights (the dW are raw 7-12)
    hidden_bytes = 2 * (TOKENS + MACROBATCH) * INTERMEDIATE
    peak = torch.cuda.max_memory_allocated(device) - before
    assert peak <= contract_bytes + public_bytes + hidden_bytes // 2, (
        f"backward peaked {peak} B above the forward's level; the thirteen contract tensors are {contract_bytes} B and the "
        f"public outputs {public_bytes} B: something the size of the dropped d_hidden buffers ({hidden_bytes} B) is still allocated"
    )
    del grads
    torch.cuda.synchronize(device)
    retained = torch.cuda.memory_allocated(device) - before
    scratch_bytes = [2 * d.shape[0] * d.shape[1] for d in raw13_descriptors_for(workspace, config)[:7]]
    assert 0 <= retained < min(scratch_bytes), (
        f"the workspace retains {retained} B after backward (gradients held {held} B); the smallest scratch "
        f"tensor is {min(scratch_bytes)} B: scratch stays resident"
    )

    # 2. The cached launch plan's template holds no macrobatch-sized scratch (a per-call tensor there
    #    would be a stale address on the next call).
    prepared = workspace._cutedsl_bwd_preparation
    assert prepared is not None
    for _, _, plan in prepared.launch_plans._entries.values():
        for tensor in _tensors(plan.template):
            assert tensor.numel() * tensor.element_size() < min(
                scratch_bytes
            ), f"launch template holds a {tuple(tensor.shape)} {tensor.dtype} tensor the size of a scratch buffer"

    # 3. Repeated calls on the default stream and one on a second stream are bit-identical.
    del forward_context
    first = tuple(t.clone() for t in _tensors(_step(config, workspace, schedule, inputs)[2]))
    second = _step(config, workspace, schedule, inputs)[2]
    side = torch.cuda.Stream(device)
    side.wait_stream(torch.cuda.current_stream(device))
    with torch.cuda.stream(side):
        third = _step(config, workspace, schedule, inputs)[2]
    torch.cuda.current_stream(device).wait_stream(side)
    for tensor in _tensors(third):
        tensor.record_stream(torch.cuda.current_stream(device))
    torch.cuda.synchronize(device)
    for name, a, b, c in zip(GRADIENT_NAMES, first, _tensors(second), _tensors(third), strict=True):
        assert torch.equal(a, b), f"{name}: second call differs bitwise"
        assert torch.equal(a, c), f"{name}: call on a second stream differs bitwise"

    # 4. The outputs of call N are the caller's.  Call N + 1 runs with a doubled d_output (so every gradient
    #    changes and an overwrite is visible) while call N's outputs are still held: it must leave them bitwise
    #    unchanged and must not hand out their storages.  A backward that hands out buffers the caller still holds
    #    fails both.  Resident scratch as such (raw13 0-6 kept, fresh dW and a fresh epilogue d_x) does not reach
    #    this test's outputs; check 1 catches it, and so does the same unchanged / distinct-storage check on the
    #    split-hidden path, where d_x is the backward's raw13[0] (test_cutedsl_routed_hidden_bf16).
    kept = tuple(_tensors(second))
    snapshots = tuple(t.clone() for t in kept)
    doubled = (*inputs[:-1], inputs[-1] * 2)
    later = tuple(_tensors(_step(config, workspace, schedule, doubled)[2]))
    torch.cuda.synchronize(device)
    for name, tensor, snapshot, new in zip(GRADIENT_NAMES, kept, snapshots, later, strict=True):
        assert torch.equal(tensor, snapshot), f"{name}: the next backward overwrote the previous call's output"
        assert tensor.untyped_storage().data_ptr() != new.untyped_storage().data_ptr(), f"{name}: the next backward handed out the previous call's storage"
        assert not torch.equal(tensor, new), f"{name}: a doubled d_output gave the same gradient (check is void)"
    dist.barrier()


def test_cutedsl_backward_scratch_layout(context: tuple[int, int, torch.device]) -> None:
    """One call's raw13 as backward_bf16 allocates it: the thirteen row-major tensors of the contract,
    128-byte aligned, each its own allocation (as the CUDA backward's at::empty temporaries), so that
    dropping the six kernel-internal scratch tensors frees their memory while the caller still holds
    d_x_shared and the dW."""
    rank, world_size, device = context
    from cudnn.moe_dispatch_mlp_combine.cutedsl.persistent_bf16_bwd import _allocate_raw13
    from cudnn.moe_dispatch_mlp_combine.cutedsl.persistent_bf16_bwd_contract import BwdGeometry

    geometry = BwdGeometry(
        schedule_capacity=TOKENS * TOPK * world_size,
        num_local_tokens=TOKENS,
        macrobatch_size=MACROBATCH,
        minibatch_size=MINIBATCH,
        num_comm_sms=14,
        num_local_experts=LOCAL_EXPERTS,
        hidden_size=HIDDEN,
        routed_hidden_size=None,
        intermediate_size=INTERMEDIATE,
        topk=TOPK,
        swiglu_limit=None,
        ep_size=world_size,
    )
    descriptors = raw13_descriptors(geometry)
    torch.cuda.synchronize(device)
    before = torch.cuda.memory_allocated(device)
    raw13 = _allocate_raw13(tuple(d.shape for d in descriptors), torch.empty(0, dtype=torch.bfloat16, device=device))
    assert len(raw13) == len(descriptors) == 13
    for descriptor, tensor in zip(descriptors, raw13, strict=True):
        assert tuple(tensor.shape) == tuple(descriptor.shape), descriptor.name
        assert tensor.dtype == torch.bfloat16 and tensor.device == torch.device(device), descriptor.name
        assert tensor.is_contiguous(), f"{descriptor.name} is not row-major"
        assert tensor.data_ptr() % 128 == 0, f"{descriptor.name} is not 128-byte aligned"
    del descriptor, tensor  # loop variables must not keep raw13[12] alive below
    assert len({tensor.untyped_storage().data_ptr() for tensor in raw13}) == len(raw13), "raw13 must be separate allocations"
    internal_bytes = sum(tensor.numel() * 2 for tensor in raw13[1:7])
    held = torch.cuda.memory_allocated(device)
    raw13 = (raw13[0], *raw13[7:])  # the caller keeps d_x_shared and the dW only
    assert held - torch.cuda.memory_allocated(device) >= internal_bytes, "internal scratch stayed allocated"
    del raw13
    assert torch.cuda.memory_allocated(device) == before


def raw13_descriptors_for(workspace, config):
    """The backward's raw13 descriptors for this workspace and config (shapes only)."""
    from cudnn.moe_dispatch_mlp_combine.cutedsl.persistent_bf16_bwd_contract import BwdGeometry

    geometry = BwdGeometry(
        schedule_capacity=workspace.schedule_capacity,
        num_local_tokens=workspace.num_local_tokens,
        macrobatch_size=config.macrobatch_size,
        minibatch_size=config.bwd_minibatch_size,
        num_comm_sms=config.bwd_num_comm_sms,
        num_local_experts=LOCAL_EXPERTS,
        hidden_size=workspace.hidden_size,
        routed_hidden_size=None,
        intermediate_size=INTERMEDIATE,
        topk=workspace.topk,
        swiglu_limit=None,
        ep_size=workspace.ep_size,
    )
    return raw13_descriptors(geometry)


# --------------------------------------------------------------------------------------------------------------------
# BF16 backward metadata tables
#
# The CuTe DSL BWD metadata builder writes exactly the tables of a plain full-scan reference.
#
# One GPU (SM100 / SM103): ``python -m pytest -s moe_dispatch_mlp_combine/cutedsl/test_backward.py``.  The builder visits, for
# minibatch m >= 1, only the experts overlapping that minibatch's rows; the reference is a per-minibatch
# scan of every expert (CUDA's ``build_minibatch_expert_offsets_kernel``
# order, plus the CuTe DSL rule that an expert without rows is registered once, in minibatch 0, and the
# task counts without SwiGLU tasks).  Every table slot is compared, including the capacity tails, on random
# expert row counts with empty experts (first, last, runs of them), experts spanning several minibatches and
# macrobatches, a partial last minibatch, minibatches holding more than 32 experts (several warp chunks),
# one macrobatch (no expert / owner tables) and several (the O(1)-decode tables).  ``recompute_forward_context`` (the
# megakernel replays the first macrobatch too): the expert / owner tables at any capacity, every minibatch with replay
# tasks.


SENTINEL = -7


def _reference(counts, capacity, macro, mini, hidden, intermediate, k, segment_capacity, bucket_capacity, recompute_forward_context=False):
    nrows = sum(counts)
    n = -(-nrows // mini)
    o1 = recompute_forward_context or capacity > macro
    pairs, completed = [0] * (k + 1), [0] * (k + 1)
    experts = [SENTINEL] * segment_capacity
    buckets = [SENTINEL] * bucket_capacity
    pair_count = segment_count = 0
    for m in range(n):
        begin, end = m * mini, min(m * mini + mini, nrows)
        macro_begin = begin // macro * macro
        macro_end = min(macro_begin + macro, nrows)
        expert_begin = 0
        for e, rows in enumerate(counts):
            expert_end = expert_begin + rows
            pair_count += max(expert_begin, begin) < min(expert_end, end)
            segment_begin, segment_end = max(expert_begin, macro_begin), min(expert_end, macro_end)
            if (segment_begin < segment_end and begin < segment_end <= end) or (segment_begin >= segment_end and rows == 0 and m == 0):
                if o1:
                    experts[segment_count] = e
                segment_count += 1
            expert_begin = expert_end
        pairs[m + 1], completed[m + 1] = pair_count, segment_count
    for m in range(n + 1, k + 1):
        pairs[m], completed[m] = pair_count, segment_count
    if o1:
        q, i, h = mini // 256, intermediate // 256, hidden // 256
        down, replay, wgrad = q * i + q * h, q * i, 3 * i * h
        width = 1
        while width * 2 <= down:
            width *= 2
        saved = 0 if recompute_forward_context else min(n, macro // mini)
        prefix = lambda owner: owner * down + max(0, owner - saved) * replay + completed[owner] * wgrad
        count = -(-prefix(n) // width)
        owner = 0
        for bucket in range(count):
            while owner + 1 < n and prefix(owner + 1) <= bucket * width:
                owner += 1
            buckets[bucket] = owner
        for bucket in range(count, bucket_capacity):
            buckets[bucket] = max(0, n - 1)
        for segment in range(segment_count, segment_capacity):
            experts[segment] = -1
    return pairs, completed, experts, buckets


def _counts(rng, experts, rows):
    """Random multiples of 256 summing to rows, with empty experts at the ends and in runs."""
    blocks = rows // 256
    weights = [0 if rng.random() < 0.3 else rng.random() for _ in range(experts)]
    weights[0] = weights[-1] = 0.0
    if not any(weights):
        weights[experts // 2] = 1.0
    total = sum(weights)
    counts = [int(blocks * w / total) for w in weights]
    counts[max(range(experts), key=lambda e: weights[e])] += blocks - sum(counts)
    return [c * 256 for c in counts]


# (local experts, rows, capacity, macrobatch, minibatch, hidden, intermediate)
METADATA_CASES = (
    (8, 16384, 16384, 16384, 4096, 2048, 1024),  # one macrobatch, few experts spanning minibatches
    (64, 65536, 65536, 65536, 4096, 4096, 1024),  # one macrobatch (the bench default)
    (512, 131072, 131072, 131072, 2048, 2048, 1280),  # one macrobatch, 512 experts
    (64, 61440, 65536, 16384, 4096, 4096, 1024),  # several macrobatches, partial last macrobatch
    (64, 62464, 65536, 16384, 4096, 4096, 1024),  # several macrobatches, partial last minibatch (1024 rows)
    (128, 32768, 65536, 32768, 16384, 2048, 1024),  # ~64 experts per minibatch: several warp chunks per minibatch
    (384, 98304, 131072, 32768, 4096, 7168, 3072),  # several macrobatches, 384 experts
    (16, 49152, 65536, 8192, 2048, 2048, 1024),  # experts span several macrobatches
)


@pytest.mark.skipif(not torch.cuda.is_available() or torch.cuda.get_device_capability() not in SUPPORTED_CAPABILITIES, reason="SM100/SM103 GPU required")
@pytest.mark.parametrize("recompute_forward_context", (False, True), ids=("saved", "recompute_forward_context"))
@pytest.mark.parametrize("seed", range(3))
@pytest.mark.parametrize("case", METADATA_CASES, ids=lambda c: f"E{c[0]}-rows{c[1]}-macro{c[3]}-mini{c[4]}")
def test_metadata_tables_match_full_scan(case, seed, recompute_forward_context):
    from types import SimpleNamespace

    from cudnn.moe_dispatch_mlp_combine.cutedsl._host_common import current_stream_handle
    from cudnn.moe_dispatch_mlp_combine.cutedsl._persistent_bf16_bwd_metadata import prepare_bwd_metadata_launch

    experts, rows, capacity, macro, mini, hidden, intermediate = case
    counts = _counts(random.Random(seed * 1000 + experts), experts, rows)
    k = capacity // mini
    segment_capacity = experts * (capacity // macro + 1) + 1
    bucket_capacity = 4096
    i32 = dict(dtype=torch.int32, device=torch.device("cuda"))
    inputs = SimpleNamespace(
        schedule_peer_rank=torch.zeros(capacity, **i32),
        schedule_peer_token_idx=torch.zeros(capacity, **i32),
        num_tokens=torch.tensor([sum(counts)], **i32),
        tokens_per_expert=torch.tensor(counts, **i32),
    )
    tables = tuple(torch.full((size,), SENTINEL, **i32) for size in (k + 1, k + 1, segment_capacity, bucket_capacity))
    config = SimpleNamespace(
        macrobatch_size=macro,
        minibatch_size=mini,
        num_local_tokens=1,
        topk=1,
        routed_hidden_size=hidden,
        intermediate_size=intermediate,
        recompute_forward_context=recompute_forward_context,
    )
    stream = current_stream_handle(0)
    executor, args = prepare_bwd_metadata_launch(inputs, SimpleNamespace(metadata=tables), config=config, stream=stream)
    executor(*args, stream)
    torch.cuda.synchronize()
    expected = _reference(counts, capacity, macro, mini, hidden, intermediate, k, segment_capacity, bucket_capacity, recompute_forward_context)
    for name, got, want in zip(("pairs", "completed", "experts", "buckets"), tables, expected):
        assert got.tolist() == want, f"{name} differs"
