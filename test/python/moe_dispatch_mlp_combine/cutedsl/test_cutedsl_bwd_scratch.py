# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The CuTe DSL BF16 backward owns its macrobatch-sized scratch per call.

Run under torchrun on SM100 / SM103 at any world size, e.g.
``torchrun --standalone --nproc-per-node=4 -m pytest -s moe_dispatch_mlp_combine/cutedsl/test_cutedsl_bwd_scratch.py``.
The CUDA backward allocates its temporaries (d_x_*, d_gate_*, d_up_*, d_hidden_*, d_y_routed) with
``at::empty`` on every call and frees them on return, so a forward that follows can reuse that memory.
The CuTe DSL backward must behave the same: after ``backward`` returns, the workspace keeps only its
small buffers (readiness counters and decoder metadata, schedule-capacity sized; router weights and
partials, macrobatch-sized), no activation-sized scratch; its cached launch plans hold no scratch
address; repeated calls, on the default stream and on a second stream, give bit-identical gradients;
and a later call with other gradients neither overwrites nor reuses the storage of an earlier call's
outputs.  Resident scratch itself is caught by the retained-memory check and, on the split-hidden path
where d_x is the backward's own scratch, by test_cutedsl_functional.test_cutedsl_routed_hidden_bf16.
"""

import math

import pytest
import torch
import torch.distributed as dist

from cudnn.moe_dispatch_mlp_combine import functional
from cudnn.moe_dispatch_mlp_combine.cutedsl.persistent_bf16_bwd_contract import RAW13_NAMES, raw13_descriptors
from cudnn.moe_dispatch_mlp_combine.cutedsl._arch import SUPPORTED_CAPABILITIES

from .utils import ungated, generate_inputs

GRADIENT_NAMES = ("d_x", "d_router_weights", "d_w_routed_gate", "d_w_routed_up", "d_w_routed_down", "d_w_shared_gate", "d_w_shared_up", "d_w_shared_down")
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
    #    split-hidden path, where d_x is the backward's raw13[0] (test_cutedsl_functional.test_cutedsl_routed_hidden_bf16).
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
