# Copyright 2026 Cursor Research
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Derived from Mixture-of-Kittens (https://github.com/cursor/mixture-of-kittens, commit 22fc95ae); modified by NVIDIA.

"""Expert-parallel groups and workspaces: EP1 on every rank, subgroups, the workspace cache, the barrier under
torch.compile, and end-to-end layers over routings, local expert counts and top-k values.

Run under torchrun, e.g. ``torchrun --standalone --nproc-per-node=8 -m pytest -s moe_dispatch_mlp_combine/cutedsl/test_distributed.py``.
"""

import dataclasses
import math

import pytest
import torch
import torch.distributed as dist

from cudnn.moe_dispatch_mlp_combine import functional, ops

from .utils import capacity_multiplier, check_correctness, run_reference_bf16, ungated

# --------------------------------------------------------------------------------------------------------------------
# Groups, workspaces and end-to-end layers
#
# Miscellaneous tests.


BF16_TOLERANCE = (0.5, 0.01)


MXFP8_TOLERANCE = (1.0, 0.1)


BF16_GRADIENT_MIN_COSINE = 0.9999


MXFP8_GRADIENT_MIN_COSINE = 0.996


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


def _run_e2e_case(
    context: tuple[int, int, torch.device],
    *,
    name: str,
    hidden_size: int,
    intermediate_size: int,
    num_local_experts: int,
    topk: int,
    config: functional.CuteMoKConfig,
    precisions: tuple[str, ...],
    seed: int = 1234,
    grad_seed: int | None = None,
    top_experts: torch.Tensor | None = None,
    group: dist.ProcessGroup | None = None,
) -> None:
    rank, _, device = context
    ep_group = dist.group.WORLD if group is None else group
    num_local_tokens = 512
    ep_size = dist.get_world_size(ep_group)
    # Top-k routing needs at least topk experts in the group.  The cases below are written for
    # EP8 (4 local experts x 8 ranks >= top-8); at a smaller EP size the per-rank expert count is
    # raised to keep the same request valid (torch.topk raised "selected index k out of range"
    # at EP1).  EP8 is unchanged.
    num_local_experts = max(num_local_experts, -(-topk // ep_size))
    num_experts = num_local_experts * ep_size
    # The schedule table must hold the padded routing (see utils.capacity_multiplier); the
    # cases' multipliers hold it at EP8, at EP1 a rank with 32 local experts needs a larger table.
    config = dataclasses.replace(
        config, schedule_capacity_multiplier=capacity_multiplier(ep_size, num_local_tokens, topk, num_local_experts, base=config.schedule_capacity_multiplier)
    )
    generator = torch.Generator(device=device).manual_seed(seed + rank)
    router_logits = torch.randn(
        num_local_tokens,
        num_experts,
        generator=generator,
        device=device,
    )
    topk_values, generated_top_experts = torch.topk(router_logits, topk, dim=1)
    router_weights = torch.softmax(topk_values.float(), dim=-1)
    x = torch.randn(
        num_local_tokens,
        hidden_size,
        generator=generator,
        device=device,
        dtype=torch.bfloat16,
    )
    w_shared_gate = (
        torch.randn(
            intermediate_size,
            hidden_size,
            generator=generator,
            device=device,
            dtype=torch.bfloat16,
        )
        * hidden_size**-0.5
    )
    w_shared_up = (
        torch.randn(
            intermediate_size,
            hidden_size,
            generator=generator,
            device=device,
            dtype=torch.bfloat16,
        )
        * hidden_size**-0.5
    )
    w_shared_down = (
        torch.randn(
            hidden_size,
            intermediate_size,
            generator=generator,
            device=device,
            dtype=torch.bfloat16,
        )
        * intermediate_size**-0.5
    )
    w_routed_gate = (
        torch.randn(
            num_local_experts,
            intermediate_size,
            hidden_size,
            generator=generator,
            device=device,
            dtype=torch.bfloat16,
        )
        * hidden_size**-0.5
    )
    w_routed_up = (
        torch.randn(
            num_local_experts,
            intermediate_size,
            hidden_size,
            generator=generator,
            device=device,
            dtype=torch.bfloat16,
        )
        * hidden_size**-0.5
    )
    w_routed_down = (
        torch.randn(
            num_local_experts,
            hidden_size,
            intermediate_size,
            generator=generator,
            device=device,
            dtype=torch.bfloat16,
        )
        * intermediate_size**-0.5
    )
    if grad_seed is None:
        grad_generator = generator
    else:
        grad_generator = torch.Generator(device=device).manual_seed(grad_seed + rank)
    d_output = (
        torch.randn(
            num_local_tokens,
            hidden_size,
            generator=grad_generator,
            device=device,
            dtype=torch.bfloat16,
        )
        * hidden_size**-0.5
    )
    input_tuple = (
        x,
        generated_top_experts if top_experts is None else top_experts,
        router_weights,
        w_shared_gate,
        w_shared_up,
        w_shared_down,
        w_routed_gate,
        w_routed_up,
        w_routed_down,
        d_output,
    )
    reference = run_reference_bf16(*input_tuple, group=ep_group)
    workspace = functional.get_workspace(
        config,
        ep_group,
        device=device,
        num_local_tokens=num_local_tokens,
        hidden_size=hidden_size,
        topk=topk,
    )
    schedule = functional.build_schedule(
        workspace,
        config,
        input_tuple[1],
        num_local_experts=num_local_experts,
    )

    for precision in precisions:
        if precision == "bf16":
            output, forward_context = functional.forward(
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
            gradients = ungated(
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
            actual = (output, *gradients)
            tolerance = BF16_TOLERANCE
            gradient_min_cosine = BF16_GRADIENT_MIN_COSINE
        elif precision == "mxfp8":
            (
                w_routed_gate_fp8,
                w_routed_gate_sc,
                w_routed_gate_t_fp8,
                w_routed_gate_t_sc,
            ) = ops.mxfp8_quantize(w_routed_gate, True, True)
            (
                w_routed_up_fp8,
                w_routed_up_sc,
                w_routed_up_t_fp8,
                w_routed_up_t_sc,
            ) = ops.mxfp8_quantize(w_routed_up, True, True)
            (
                w_routed_down_fp8,
                w_routed_down_sc,
                w_routed_down_t_fp8,
                w_routed_down_t_sc,
            ) = ops.mxfp8_quantize(w_routed_down, True, True)
            assert all(
                tensor is not None
                for tensor in (
                    w_routed_gate_fp8,
                    w_routed_gate_sc,
                    w_routed_gate_t_fp8,
                    w_routed_gate_t_sc,
                    w_routed_up_fp8,
                    w_routed_up_sc,
                    w_routed_up_t_fp8,
                    w_routed_up_t_sc,
                    w_routed_down_fp8,
                    w_routed_down_sc,
                    w_routed_down_t_fp8,
                    w_routed_down_t_sc,
                )
            )
            output, forward_context = functional.forward(
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
            gradients = ungated(
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
            actual = (output, *gradients)
            tolerance = MXFP8_TOLERANCE
            gradient_min_cosine = MXFP8_GRADIENT_MIN_COSINE
        else:
            raise AssertionError(f"unsupported precision {precision!r}")
        for result_name, expected, result in zip(
            RESULT_NAMES,
            reference,
            actual,
            strict=True,
        ):
            check_correctness(
                f"{name}/{precision}/{result_name}",
                expected,
                result,
                tolerance,
                print_stats=rank == 0,
            )
        for result_name, expected, result in zip(
            RESULT_NAMES[1:],
            reference[1:],
            actual[1:],
            strict=True,
        ):
            expected_flat = expected.float().reshape(-1)
            result_flat = result.float().reshape(-1)
            denominator = expected_flat.norm() * result_flat.norm()
            if float(denominator.item()) <= 1e-12:
                local_cosine = 1.0 if torch.equal(expected_flat, result_flat) else 0.0
            else:
                local_cosine = float(((expected_flat @ result_flat) / denominator).item())
            finite = bool(torch.isfinite(expected_flat).all().item()) and bool(torch.isfinite(result_flat).all().item())
            stats = torch.tensor(
                [local_cosine, int(finite)],
                device=device,
                dtype=torch.float32,
            )
            dist.all_reduce(stats[0:1], op=dist.ReduceOp.MIN)
            dist.all_reduce(stats[1:2], op=dist.ReduceOp.MIN)
            minimum_cosine = float(stats[0].item())
            if rank == 0:
                print(f"{name}/{precision}/{result_name}: " f"minimum cosine={minimum_cosine:.8f}")
            assert bool(stats[1].item())
            assert minimum_cosine >= gradient_min_cosine


def _assert_metadata(
    tensors: tuple[torch.Tensor, ...],
    expected: tuple[tuple[tuple[int, ...], torch.dtype], ...],
) -> None:
    assert len(tensors) == len(expected)
    for tensor, (shape, dtype) in zip(tensors, expected, strict=True):
        assert isinstance(tensor, torch.Tensor)
        assert tuple(tensor.shape) == shape
        assert tensor.dtype == dtype
        assert tensor.device.type == "cuda"


def _make_fake_workspace(
    device: torch.device,
    *,
    num_local_tokens: int = 512,
    hidden_size: int = 1024,
    topk: int = 2,
    ep_size: int = 4,
    schedule_capacity: int = 4096,
) -> functional.CuteMoKWorkspace:
    pointers = list(range(1, ep_size + 1))

    def tensor(shape: tuple[int, ...], dtype: torch.dtype) -> torch.Tensor:
        return torch.empty(shape, device=device, dtype=dtype)

    return functional.CuteMoKWorkspace(
        group_name="fake-ep-group",
        ep_rank=0,
        ep_size=ep_size,
        device=device,
        num_local_tokens=num_local_tokens,
        hidden_size=hidden_size,
        topk=topk,
        schedule_capacity=schedule_capacity,
        x_buffer=tensor((num_local_tokens, hidden_size), torch.bfloat16),
        x_buffer_handle=None,
        x_buffer_ptrs=pointers,
        combine_buffer=tensor(
            (num_local_tokens * topk, hidden_size),
            torch.bfloat16,
        ),
        combine_buffer_handle=None,
        combine_buffer_ptrs=pointers,
        d_y_buffer=tensor((num_local_tokens, hidden_size), torch.bfloat16),
        d_y_buffer_handle=None,
        d_y_buffer_ptrs=pointers,
        d_x_routed_buffer=tensor(
            (num_local_tokens * topk, hidden_size),
            torch.bfloat16,
        ),
        d_x_routed_buffer_handle=None,
        d_x_routed_buffer_ptrs=pointers,
        router_weight_buffer=tensor(
            (num_local_tokens, topk),
            torch.float32,
        ),
        router_weight_buffer_handle=None,
        router_weight_buffer_ptrs=pointers,
        d_router_weight_buffer=tensor(
            (num_local_tokens, topk),
            torch.float32,
        ),
        d_router_weight_buffer_handle=None,
        d_router_weight_buffer_ptrs=pointers,
        all_gather_top_experts_buffer=tensor(
            (ep_size, num_local_tokens, topk),
            torch.int32,
        ),
        all_gather_top_experts_buffer_handle=None,
        all_gather_top_experts_buffer_multicast_ptr=1,
        barrier_buffer=tensor((1,), torch.int32),
        barrier_buffer_handle=None,
        barrier_buffer_ptrs=pointers,
        barrier_buffer_multicast_ptr=1,
        barrier_target=tensor((1,), torch.int32),
    )


def test_ep1_on_each_rank(
    context: tuple[int, int, torch.device],
) -> None:
    rank, world_size, device = context
    singleton_groups = [dist.new_group(ranks=[group_rank]) for group_rank in range(world_size)]
    ep_group = singleton_groups[rank]
    assert isinstance(ep_group, dist.ProcessGroup)

    config = functional.CuteMoKConfig(
        fwd_num_comm_sms=2,
        bwd_num_comm_sms=2,
        minibatch_size=256,
        macrobatch_size=512,
        all_gather_top_experts_chunk_bytes=16,
    )
    functional.clear_workspace_cache()
    try:
        workspace = functional.get_workspace(
            config,
            ep_group,
            device=device,
            num_local_tokens=512,
            hidden_size=256,
            topk=1,
        )

        assert workspace.ep_rank == 0
        assert workspace.ep_size == 1
        for buffer, handle, pointers in (
            (
                workspace.x_buffer,
                workspace.x_buffer_handle,
                workspace.x_buffer_ptrs,
            ),
            (
                workspace.combine_buffer,
                workspace.combine_buffer_handle,
                workspace.combine_buffer_ptrs,
            ),
            (
                workspace.d_y_buffer,
                workspace.d_y_buffer_handle,
                workspace.d_y_buffer_ptrs,
            ),
            (
                workspace.d_x_routed_buffer,
                workspace.d_x_routed_buffer_handle,
                workspace.d_x_routed_buffer_ptrs,
            ),
            (
                workspace.router_weight_buffer,
                workspace.router_weight_buffer_handle,
                workspace.router_weight_buffer_ptrs,
            ),
            (
                workspace.d_router_weight_buffer,
                workspace.d_router_weight_buffer_handle,
                workspace.d_router_weight_buffer_ptrs,
            ),
        ):
            assert handle is None
            assert pointers == [buffer.data_ptr()]

        assert workspace.all_gather_top_experts_buffer_handle is None
        assert workspace.all_gather_top_experts_buffer_multicast_ptr == workspace.all_gather_top_experts_buffer.data_ptr()
        assert workspace.barrier_buffer_handle is None
        assert workspace.barrier_buffer_ptrs == [workspace.barrier_buffer.data_ptr()]
        assert workspace.barrier_buffer_multicast_ptr == workspace.barrier_buffer.data_ptr()

        top_experts = torch.zeros(
            512,
            1,
            dtype=torch.int32,
            device=device,
        )
        ops.all_gather_top_experts(
            top_experts,
            workspace.all_gather_top_experts_buffer,
            workspace.all_gather_top_experts_buffer_multicast_ptr,
            0,
            16,
        )
        assert torch.equal(
            workspace.all_gather_top_experts_buffer[0],
            top_experts,
        )

        ops.barrier_all(
            workspace.barrier_buffer,
            workspace.barrier_buffer_ptrs,
            workspace.barrier_buffer_multicast_ptr,
            workspace.barrier_target,
        )
        torch.cuda.synchronize(device)
        assert int(workspace.barrier_buffer.item()) == 0
        assert int(workspace.barrier_target.item()) == 0

        _run_e2e_case(
            context,
            name="ep1-minimum",
            hidden_size=256,
            intermediate_size=256,
            num_local_experts=1,
            topk=1,
            config=config,
            precisions=("bf16", "mxfp8"),
            group=ep_group,
        )
    finally:
        functional.clear_workspace_cache()
        dist.barrier()
        for group in singleton_groups:
            if isinstance(group, dist.ProcessGroup):
                dist.destroy_process_group(group)


def test_workspace_cache_reuse_and_isolation(
    context: tuple[int, int, torch.device],
) -> None:
    _, world_size, device = context
    functional.clear_workspace_cache()
    duplicate_world = dist.new_group(ranks=list(range(world_size)))
    kwargs = {
        "device": device,
        "num_local_tokens": 512,
        "hidden_size": 256,
        "topk": 1,
    }
    default_config = functional.CuteMoKConfig(
        minibatch_size=256,
        macrobatch_size=512,
    )
    larger_config = functional.CuteMoKConfig(
        minibatch_size=256,
        macrobatch_size=512,
        # A larger capacity factor max(2, ceil(ep_size * multiplier)) at every EP size: at EP1,
        # 1.5 gives the same factor 2 as the default 0.5 and so the same (correctly cached) workspace.
        schedule_capacity_multiplier=2.5,
    )
    try:
        created_workspace = functional.create_workspace(
            default_config,
            dist.group.WORLD,
            **kwargs,
        )
        default_workspace = functional.get_workspace(
            default_config,
            dist.group.WORLD,
            **kwargs,
        )
        reused_workspace = functional.get_workspace(
            default_config,
            dist.group.WORLD,
            **kwargs,
        )
        larger_workspace = functional.get_workspace(
            larger_config,
            dist.group.WORLD,
            **kwargs,
        )
        duplicate_group_workspace = functional.get_workspace(
            default_config,
            duplicate_world,
            **kwargs,
        )

        expected_default_capacity = 512 * max(
            2,
            math.ceil(world_size * default_config.schedule_capacity_multiplier),
        )
        expected_larger_capacity = 512 * max(
            2,
            math.ceil(world_size * larger_config.schedule_capacity_multiplier),
        )
        assert created_workspace is not default_workspace
        assert reused_workspace is default_workspace
        assert larger_workspace is not default_workspace
        assert duplicate_group_workspace is not default_workspace
        assert default_workspace.schedule_capacity == expected_default_capacity
        assert larger_workspace.schedule_capacity == expected_larger_capacity
        assert duplicate_group_workspace.group_name == duplicate_world.group_name
        assert duplicate_group_workspace.group_name != default_workspace.group_name
    finally:
        functional.clear_workspace_cache()
        dist.barrier()
        if isinstance(duplicate_world, dist.ProcessGroup):
            dist.destroy_process_group(duplicate_world)


def test_ep_subgroups_do_not_use_world(
    context: tuple[int, int, torch.device],
) -> None:
    rank, world_size, device = context
    if world_size < 8 or world_size % 4 != 0:
        pytest.skip("requires a world size divisible by four and at least eight")

    functional.clear_workspace_cache()
    subgroup_ranks = [list(range(start, start + 4)) for start in range(0, world_size, 4)]
    subgroups = [dist.new_group(ranks=ranks) for ranks in subgroup_ranks]
    subgroup_index = rank // 4
    subgroup = subgroups[subgroup_index]
    subgroup_rank = rank % 4
    topk = 2 + subgroup_index % 2
    num_local_experts = 4
    config = functional.CuteMoKConfig(
        minibatch_size=256,
        macrobatch_size=512,
    )
    try:
        workspace = functional.get_workspace(
            config,
            subgroup,
            device=device,
            num_local_tokens=512,
            hidden_size=1024,
            topk=topk,
        )
        routes = (torch.arange(512 * topk, device=device, dtype=torch.int64) + subgroup_rank).remainder(4 * num_local_experts).view(512, topk)
        schedule = functional.build_schedule(
            workspace,
            config,
            routes,
            num_local_experts=num_local_experts,
        )
        valid_peer_ranks = (schedule.peer_rank == -1) | ((schedule.peer_rank >= 0) & (schedule.peer_rank < 4))

        assert workspace.group_name == subgroup.group_name
        assert workspace.group_name != dist.group.WORLD.group_name
        assert workspace.ep_rank == subgroup_rank
        assert workspace.ep_size == 4
        assert workspace.topk == topk
        assert len(workspace.x_buffer_ptrs) == 4
        assert len(workspace.combine_buffer_ptrs) == 4
        assert len(workspace.barrier_buffer_ptrs) == 4
        assert bool(valid_peer_ranks.all().item())
    finally:
        functional.clear_workspace_cache()
        dist.barrier()
        for process_group in subgroups:
            if isinstance(process_group, dist.ProcessGroup):
                dist.destroy_process_group(process_group)


def test_compiled_barrier_updates_state(
    context: tuple[int, int, torch.device],
) -> None:
    _, _, device = context
    functional.clear_workspace_cache()
    workspace = functional.get_workspace(
        functional.CuteMoKConfig(
            minibatch_size=256,
            macrobatch_size=512,
        ),
        dist.group.WORLD,
        device=device,
        num_local_tokens=512,
        hidden_size=256,
        topk=1,
    )

    def run_barrier() -> None:
        ops.barrier_all(
            workspace.barrier_buffer,
            workspace.barrier_buffer_ptrs,
            workspace.barrier_buffer_multicast_ptr,
            workspace.barrier_target,
        )

    try:
        assert int(workspace.barrier_buffer.item()) == 0
        assert int(workspace.barrier_target.item()) == 0
        torch._dynamo.reset()
        # fullgraph=False, and that is a real narrowing of what this used to check.  The CUDA
        # build registered barrier_all through torch.library, so Dynamo treated it as one opaque
        # op and a full graph could contain it.  Nothing registers mok::* ops now (see
        # cudnn/moe_dispatch_mlp_combine/ops.py): barrier_all is a plain function that reaches a CuTe DSL kernel, so Dynamo
        # traces into it and stops at tensor.data_ptr().  A graph break there is correct rather
        # than merely tolerable -- MCore calls MoK from a torch.autograd.Function, which breaks
        # the graph anyway, so no caller needs barrier_all inside a full graph.  What still has to
        # hold, and is what this tests, is that the barrier runs and advances its counters when it
        # is called from compiled code.  Re-registering the ops is the fix if a caller ever does
        # need fullgraph.
        compiled = torch.compile(
            run_barrier,
            backend="inductor",
            fullgraph=False,
        )
        compiled()
        torch.cuda.synchronize(device)
        compiled()
        torch.cuda.synchronize(device)
        dist.barrier()
        # ops.barrier_all is a no-op at EP1 (no peers, no multicast counter), so the counters stay 0
        # there (test_ep1_on_each_rank); otherwise every rank adds ep_size per barrier.
        expected = 2 * workspace.ep_size if workspace.ep_size > 1 else 0
        assert int(workspace.barrier_buffer.item()) == expected
        assert int(workspace.barrier_target.item()) == expected
    finally:
        functional.clear_workspace_cache()
        torch._dynamo.reset()


def test_e2e_fixed_first_topk_experts_default_capacity(
    context: tuple[int, int, torch.device],
) -> None:
    """Every token routed to the same first top-k experts, at the default schedule capacity."""
    _, _, device = context
    functional.clear_workspace_cache()
    topk = 8
    try:
        _run_e2e_case(
            context,
            name="fixed-first-topk/default-capacity",
            hidden_size=7168,
            intermediate_size=2048,
            num_local_experts=4,
            topk=topk,
            config=functional.CuteMoKConfig(
                minibatch_size=256,
                macrobatch_size=512,
            ),
            precisions=("bf16", "mxfp8"),
            top_experts=torch.arange(
                topk,
                device=device,
                dtype=torch.int64,
            )
            .expand(512, topk)
            .contiguous(),
        )
    finally:
        functional.clear_workspace_cache()


def test_e2e_requested_shapes_and_local_expert_counts(
    context: tuple[int, int, torch.device],
) -> None:
    functional.clear_workspace_cache()
    try:
        for hidden_size, intermediate_size in (
            (7168, 2048),
            (2048, 768),
            (2560, 2560),
        ):
            for num_local_experts in (4, 6, 8, 12, 16, 24, 32):
                _run_e2e_case(
                    context,
                    name=(f"shape-h{hidden_size}-i{intermediate_size}" f"/local-experts-{num_local_experts}"),
                    hidden_size=hidden_size,
                    intermediate_size=intermediate_size,
                    num_local_experts=num_local_experts,
                    topk=8,
                    config=functional.CuteMoKConfig(
                        minibatch_size=256,
                        macrobatch_size=512,
                    ),
                    precisions=("mxfp8",),
                )
    finally:
        functional.clear_workspace_cache()


def test_e2e_different_topk(
    context: tuple[int, int, torch.device],
) -> None:
    functional.clear_workspace_cache()
    try:
        for topk in (1, 2, 4, 6, 8):
            _run_e2e_case(
                context,
                name=f"topk-{topk}",
                hidden_size=7168,
                intermediate_size=2048,
                num_local_experts=4,
                topk=topk,
                config=functional.CuteMoKConfig(
                    minibatch_size=256,
                    macrobatch_size=512,
                ),
                precisions=("bf16", "mxfp8"),
            )
    finally:
        functional.clear_workspace_cache()


def test_e2e_finiteness_different_seeds(
    context: tuple[int, int, torch.device],
) -> None:
    functional.clear_workspace_cache()
    try:
        for seed in (42, 123, 456, 789):
            _run_e2e_case(
                context,
                name=f"finiteness/seed-{seed}",
                hidden_size=7168,
                intermediate_size=2048,
                num_local_experts=4,
                topk=8,
                config=functional.CuteMoKConfig(
                    minibatch_size=256,
                    macrobatch_size=512,
                ),
                precisions=("mxfp8",),
                seed=seed,
                grad_seed=10_000 + seed,
            )
    finally:
        functional.clear_workspace_cache()
