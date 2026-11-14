# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The plan / execute form of a layer: ``schedule_execute`` / ``forward_execute`` / ``backward_execute`` on a
``make_plan`` match ``build_schedule`` / ``forward`` / ``backward`` bit for bit.

The plan compiles every kernel of its configuration and owns the counters and the scratch; the execute steps take
the caller's output buffers, compile nothing, allocate nothing and never wait for the device (no device-to-host read,
no synchronization), so the first execute after a plan can be captured in a CUDA graph.  One plan serves every token
count up to its maximum.  BF16 and MXFP8 routed weights (the MXFP8 Gate / Up context in both precisions).
"""

import contextlib

import pytest
import torch
import torch.distributed as dist

from cudnn.moe_dispatch_mlp_combine import functional
from cudnn.moe_dispatch_mlp_combine.cutedsl._arch import TARGET_CAPABILITY
from .call_helpers import call_arguments, generate_inputs, shared_output_gate_weight

LOCAL_EXPERTS, HIDDEN, INTERMEDIATE, MAX_TOKENS, TOPK = 4, 1024, 512, 2048, 4


def _empty(shape, device, dtype=torch.bfloat16):
    """Buffers of ``plan.*_shapes``: a shape, a (shape, dtype) pair (MXFP8 context), None, or a tuple of them."""
    if shape is None or isinstance(shape[0], int):
        return None if shape is None else torch.empty(shape, dtype=dtype, device=device)
    if isinstance(shape[1], torch.dtype):
        return torch.empty(shape[0], dtype=shape[1], device=device)
    return tuple(_empty(item, device, dtype) for item in shape)


def _forward_results(output, context):
    return [output, context.gate_shared, context.up_shared, context.hidden_shared] + (
        [] if context.shared_output_gate is None else [context.shared_output, context.shared_output_gate]
    )


def _same(expected, actual, label):
    for index, (a, b) in enumerate(zip(expected, actual, strict=True)):
        assert torch.equal(a, b), f"{label}: result {index} differs"


@contextlib.contextmanager
def _no_host_sync():
    """Inside, a CUDA call that synchronizes the host with the device (a device-to-host read, a stream or device
    synchronize, a pageable host-to-device copy) raises."""
    mode = torch.cuda.get_sync_debug_mode()
    torch.cuda.set_sync_debug_mode("error")
    try:
        yield
    finally:
        torch.cuda.set_sync_debug_mode(mode)


def _executor_counts():
    from cudnn.moe_dispatch_mlp_combine.cutedsl import persistent_bf16_bwd, persistent_bf16_fwd, persistent_mxfp8_bwd, persistent_mxfp8_fwd

    return tuple(len(module._EXECUTOR_CACHE) for module in (persistent_bf16_fwd, persistent_bf16_bwd, persistent_mxfp8_fwd, persistent_mxfp8_bwd))


# MCore's arguments: split routed weights, one combined FC1, FP32 main_grads (MXFP8: TransformerEngine's native
# columnwise payloads); "gated" adds the shared output gate, whose backward the execute step does not run.
# gate_up_context: the MXFP8 forward's routed Gate / Up context, e4m3 / scale pairs or BF16.
@pytest.mark.parametrize(
    "precision,layout,gate_up_context,gated",
    [
        ("bf16", "plain", "mxfp8", False),
        ("bf16", "mcore", "mxfp8", False),
        ("bf16", "mcore", "mxfp8", True),
        ("mxfp8", "plain", "mxfp8", False),
        ("mxfp8", "mcore", "mxfp8", False),
        ("mxfp8", "plain", "bf16", False),
    ],
)
def test_execute_matches_public_calls(context, precision, layout, gate_up_context, gated):
    rank, world_size, device = context
    if torch.cuda.get_device_capability(device) != TARGET_CAPABILITY:
        pytest.skip("CuTe DSL kernels")

    config = functional.CuteMoKConfig(
        fwd_num_comm_sms=8, bwd_num_comm_sms=8, minibatch_size=1024, macrobatch_size=4096, schedule_capacity_multiplier=1.5, gate_up_context=gate_up_context
    )
    mcore, mxfp8 = layout == "mcore", precision == "mxfp8"
    plan = functional.make_plan(
        config,
        dist.group.WORLD,
        device=device,
        max_tokens_per_rank=MAX_TOKENS,
        hidden_size=HIDDEN,
        intermediate_size=INTERMEDIATE,
        num_local_experts=LOCAL_EXPERTS,
        topk=TOPK,
        split_routed_weights=mcore,
        shared_output_gate=gated,
        backward=not gated,
        main_grad_dtype=torch.float32 if mcore else None,
        routed_precision=precision,
        native_columnwise=mxfp8 and mcore,
    )
    executors = _executor_counts()

    for seed, tokens in ((20, MAX_TOKENS), (21, MAX_TOKENS // 2), (22, MAX_TOKENS)):
        label = f"{precision}{' bf16 context' if mxfp8 and gate_up_context == 'bf16' else ''} {layout}" f"{' gated' if gated else ''} tokens {tokens}"
        inputs = generate_inputs(
            rank,
            device,
            seed,
            num_experts=LOCAL_EXPERTS * world_size,
            local_experts=LOCAL_EXPERTS,
            topk=TOPK,
            tokens=tokens,
            hidden=HIDDEN,
            intermediate=INTERMEDIATE,
        )
        x, top, router_weights, d_output = inputs[0], inputs[1].to(torch.int32), inputs[2], inputs[-1]
        gate_weight = shared_output_gate_weight(rank, device, seed, HIDDEN) if gated else None
        fw, bw, fk, bk, keepalive = call_arguments(layout, precision, inputs[3:6], inputs[6:9], output_gate=gate_weight)
        main_grads = bk.get("main_grads")
        # The whole accumulation buffers behind the main_grad views: shared FC1, shared FC2, every routed expert.
        accumulated = [] if main_grads is None else [main_grads[0]._base, main_grads[4], keepalive[-1]]

        def gradients(grads):
            for buffer in accumulated:
                buffer.zero_()
            results = list(grads()[: 2 if mcore else 8])
            return [t.clone() for t in results + accumulated]

        workspace = functional.get_workspace(config, dist.group.WORLD, device=device, num_local_tokens=tokens, hidden_size=HIDDEN, topk=TOPK)
        schedule = functional.build_schedule(workspace, config, top, num_local_experts=LOCAL_EXPERTS)
        output, forward_context = functional.forward(config, workspace, schedule, x, router_weights, *fw, None, **fk)
        expected = [schedule.num_tokens.clone(), schedule.tokens_per_expert.clone(), schedule.valid_tokens_per_expert.clone()]
        expected += [t.clone() for t in _forward_results(output, forward_context)]
        if not gated:
            expected += gradients(lambda: functional.backward(config, workspace, schedule, forward_context, d_output, x, router_weights, *bw, None, **bk))
        del schedule, output, forward_context

        schedule_out = _empty(plan.schedule_shapes(tokens), device, torch.int32)
        context_shapes, out_shape = plan.forward_shapes(config, tokens)
        context_out, out = _empty(context_shapes, device), _empty(out_shape, device)
        gate = None if not gated else torch.sigmoid(torch.nn.functional.linear(x, gate_weight))
        if not gated:
            d_x_shape, d_router_shape, d_weight_shapes = plan.backward_shapes(tokens)
            d_x_out, d_router_out = _empty(d_x_shape, device), _empty(d_router_shape, device, torch.float32)
            d_weights_out = None if mcore else _empty(d_weight_shapes, device)

        def execute():
            schedule = functional.schedule_execute(plan, config, top, out=schedule_out)
            output, forward_context = functional.forward_execute(
                plan, config, schedule, x, router_weights, *fw, None, context_out=context_out, out=out, shared_output_gate=gate
            )
            assert output is out and forward_context.gate_shared is context_out[2 if mxfp8 else 1]
            results = [schedule.num_tokens, schedule.tokens_per_expert, schedule.valid_tokens_per_expert]
            results += _forward_results(output, forward_context)
            if not gated:
                results += gradients(
                    lambda: functional.backward_execute(
                        plan,
                        config,
                        schedule,
                        forward_context,
                        d_output,
                        x,
                        router_weights,
                        *bw,
                        None,
                        main_grads,
                        bk.get("main_grad_storage_tables"),
                        out=d_x_out,
                        d_router_weights_out=d_router_out,
                        d_weights_out=d_weights_out,
                    )
                )
            return results, schedule

        actual, schedule = execute()
        _same(expected, actual, label)
        # A repeated call allocates no device memory (the clones of the comparison aside) and never waits for the
        # device, and no token count compiled a kernel.
        for buffer in accumulated:
            buffer.zero_()
        allocations = torch.cuda.memory_stats(device)["allocation.all.allocated"]
        with _no_host_sync():
            schedule = functional.schedule_execute(plan, config, top, out=schedule_out)
            _, forward_context = functional.forward_execute(
                plan, config, schedule, x, router_weights, *fw, None, context_out=context_out, out=out, shared_output_gate=gate
            )
            if not gated:
                functional.backward_execute(
                    plan,
                    config,
                    schedule,
                    forward_context,
                    d_output,
                    x,
                    router_weights,
                    *bw,
                    None,
                    main_grads,
                    bk.get("main_grad_storage_tables"),
                    out=d_x_out,
                    d_router_weights_out=d_router_out,
                    d_weights_out=d_weights_out,
                )
        assert torch.cuda.memory_stats(device)["allocation.all.allocated"] == allocations, f"{label}: the execute steps allocated device memory"
        with _no_host_sync(), pytest.raises(RuntimeError, match="synchroniz"):  # the detector is on
            top.sum().item()
        _same(expected, execute()[0], label + " repeated")
        assert _executor_counts() == executors, "an execute step compiled a kernel"

    # A variant the plan did not compile (the clamped SwiGLU) is refused, not compiled in the execute step.
    with pytest.raises(RuntimeError, match="not compiled by the plan"):
        functional.forward_execute(plan, config, schedule, x, router_weights, *fw, 1.0, context_out=context_out, out=out, shared_output_gate=gate)
    dist.barrier()


@pytest.mark.parametrize("precision", ["bf16", "mxfp8"])
def test_first_execute_captured(context, precision):
    """The first schedule + forward + backward execute after a plan, with no eager call before it, captured in a
    CUDA graph: nothing is compiled during the capture, the replays match eager execute calls bit for bit, and a
    variant the plan did not compile raises inside the capture instead of compiling there."""
    rank, world_size, device = context
    if torch.cuda.get_device_capability(device) != TARGET_CAPABILITY:
        pytest.skip("CuTe DSL kernels")
    config = functional.CuteMoKConfig(fwd_num_comm_sms=8, bwd_num_comm_sms=8, minibatch_size=1024, macrobatch_size=4096, schedule_capacity_multiplier=1.5)
    plan = functional.make_plan(
        config,
        dist.group.WORLD,
        device=device,
        max_tokens_per_rank=MAX_TOKENS,
        hidden_size=HIDDEN,
        intermediate_size=INTERMEDIATE,
        num_local_experts=LOCAL_EXPERTS,
        topk=TOPK,
        routed_precision=precision,
    )
    tokens = MAX_TOKENS // 2
    inputs = generate_inputs(
        rank,
        device,
        30,
        num_experts=LOCAL_EXPERTS * world_size,
        local_experts=LOCAL_EXPERTS,
        topk=TOPK,
        tokens=tokens,
        hidden=HIDDEN,
        intermediate=INTERMEDIATE,
    )
    x, top, router_weights, d_output = inputs[0], inputs[1].to(torch.int32), inputs[2], inputs[-1]
    fw, bw, _, _, _ = call_arguments("plain", precision, inputs[3:6], inputs[6:9])
    schedule_out = _empty(plan.schedule_shapes(tokens), device, torch.int32)
    context_shapes, out_shape = plan.forward_shapes(config, tokens)
    context_out, out = _empty(context_shapes, device), _empty(out_shape, device)
    d_x_shape, d_router_shape, d_weight_shapes = plan.backward_shapes(tokens)
    d_x_out, d_router_out = _empty(d_x_shape, device), _empty(d_router_shape, device, torch.float32)
    d_weights_out = _empty(d_weight_shapes, device)
    executors = _executor_counts()

    def execute(swiglu_limit=None):
        schedule = functional.schedule_execute(plan, config, top, out=schedule_out)
        output, forward_context = functional.forward_execute(plan, config, schedule, x, router_weights, *fw, swiglu_limit, context_out=context_out, out=out)
        grads = functional.backward_execute(
            plan,
            config,
            schedule,
            forward_context,
            d_output,
            x,
            router_weights,
            *bw,
            swiglu_limit,
            out=d_x_out,
            d_router_weights_out=d_router_out,
            d_weights_out=d_weights_out,
        )
        return [schedule.num_tokens, schedule.tokens_per_expert, *_forward_results(output, forward_context), *grads[:8]]

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        results = execute()
    assert _executor_counts() == executors, "the captured execute compiled a kernel"
    graph.replay()
    replayed = [t.clone() for t in results]
    _same(replayed, execute(), f"{precision}: eager execute after the replay")
    graph.replay()
    _same(replayed, results, f"{precision}: second replay")
    # The clamped SwiGLU, which the plan did not compile, raises inside the capture.
    with pytest.raises(RuntimeError, match="not compiled by the plan"):
        with torch.cuda.graph(torch.cuda.CUDAGraph()):
            execute(swiglu_limit=1.0)
    assert _executor_counts() == executors
    # int64 routes would need a conversion, which allocates: the execute step takes int32 only.
    with pytest.raises(TypeError, match="int32"):
        functional.schedule_execute(plan, config, top.to(torch.int64), out=schedule_out)
    del graph
    dist.barrier()
