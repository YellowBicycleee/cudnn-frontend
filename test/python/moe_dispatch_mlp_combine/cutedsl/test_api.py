# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The operator API (``cudnn.moe_dispatch_mlp_combine.api``): ``moe_dispatch_mlp_combine_{prepare, forward, backward}``.

CPU: the names follow cuDNN frontend's op naming rules, the signatures (data tensors positional, everything else
keyword-only; ``prepare`` takes exactly ``make_plan``'s arguments), the lazy export from the package and from
``cudnn.ops`` in every import order, the results' names, and the decline of an unsupported layout or dtype before anything is launched.
GPU: the operator functions return what ``schedule_execute`` + ``forward_execute`` and ``backward_execute`` return,
bit for bit, for BF16 and MXFP8 routed weights, compiling nothing and allocating no device memory.
"""

import dataclasses
import inspect
import os
import re
import subprocess
import sys
import textwrap

import pytest
import torch
import torch.distributed as dist

from cudnn._tupledict import TupleDict
from cudnn.moe_dispatch_mlp_combine import api, functional
from cudnn.moe_dispatch_mlp_combine.cutedsl._arch import TARGET_CAPABILITY
from .call_helpers import call_arguments, generate_inputs
from .utils import BACKWARD_RESULT_NAMES
from .test_plan_execute import HIDDEN, INTERMEDIATE, LOCAL_EXPERTS, MAX_TOKENS, TOPK, _empty, _executor_counts, _forward_results, _same

OPERATORS = ("moe_dispatch_mlp_combine_prepare", "moe_dispatch_mlp_combine_forward", "moe_dispatch_mlp_combine_backward")
TYPES = ("MoeDispatchMlpCombineConfig", "MoeDispatchMlpCombinePlan")
WEIGHTS = ("shared_gate_weights", "shared_up_weights", "shared_down_weights", "routed_gate_weights", "routed_up_weights", "routed_down_weights")


def test_names():
    """cuDNN frontend's rules: ``<op_name>_<direction>`` spelled out, functions only, no architecture or
    implementation technique in a name."""
    assert set(api.__all__) == set(OPERATORS + TYPES)
    for name in OPERATORS:
        assert inspect.isfunction(getattr(api, name)) and getattr(api, name).__name__ == name
        assert not re.search(r"fwd|bwd|wrapper|sm\d+|cute|dsl|cutlass|swiglu", name)
    assert api.MoeDispatchMlpCombineConfig is functional.CuteMoKConfig
    assert api.MoeDispatchMlpCombinePlan is functional.CuteMoKPlan


def _parameters(function):
    signature = inspect.signature(function)
    positional = [p.name for p in signature.parameters.values() if p.kind is p.POSITIONAL_OR_KEYWORD]
    keyword = [p.name for p in signature.parameters.values() if p.kind is p.KEYWORD_ONLY]
    assert len(positional) + len(keyword) == len(signature.parameters), "no *args / **kwargs"
    return positional, keyword, signature


def test_signatures():
    positional, keyword, _ = _parameters(api.moe_dispatch_mlp_combine_forward)
    assert positional == ["x", "top_experts", "router_weights", *WEIGHTS]
    assert keyword[:5] == ["plan", "config", "out", "context_out", "schedule_out"]
    assert set(keyword[5:]) == {"swiglu_limit", "routed_input", "shared_output_gate", "situ_beta"}

    positional, keyword, _ = _parameters(api.moe_dispatch_mlp_combine_backward)
    assert positional == ["grad_output", "x", "router_weights", *WEIGHTS, "schedule", "forward_context"]
    assert keyword[:5] == ["plan", "config", "out", "d_router_weights_out", "d_weights_out"]
    assert set(keyword[5:]) == {"main_grads", "main_grad_storage_tables", "swiglu_limit", "routed_grad_output", "routed_input", "zero_unrouted_router_grads"}

    # prepare is make_plan: the same parameters, kinds and defaults.
    prepare = inspect.signature(api.moe_dispatch_mlp_combine_prepare).parameters
    make_plan = inspect.signature(functional.make_plan).parameters
    assert [(p.name, p.kind, p.default) for p in prepare.values()] == [(p.name, p.kind, p.default) for p in make_plan.values()]


def _run(code: str) -> str:
    result = subprocess.run([sys.executable, "-c", textwrap.dedent(code)], env=dict(os.environ), capture_output=True, text=True, timeout=300)
    assert result.returncode == 0, result.stdout + result.stderr
    return result.stdout


@pytest.mark.parametrize("first", ["package", "module", "from-import", "ops"])
def test_lazy_export(first):
    """Importing the package does not import its ``api`` module; the package attribute is the module's function
    whichever is imported first, and ``cudnn.ops`` exports the same objects."""
    out = _run(f"""
        import sys
        first = {first!r}
        if first == "ops":
            import cudnn.ops
            assert "cudnn.moe_dispatch_mlp_combine" not in sys.modules, "import cudnn.ops imported the operator"
            forward = cudnn.ops.moe_dispatch_mlp_combine_forward
        import cudnn.moe_dispatch_mlp_combine as package
        if first == "package":
            assert "cudnn.moe_dispatch_mlp_combine.api" not in sys.modules, "the package imported the operator API"
            assert "moe_dispatch_mlp_combine_forward" in dir(package)
            forward = package.moe_dispatch_mlp_combine_forward
        elif first == "module":
            import cudnn.moe_dispatch_mlp_combine.api
            forward = package.moe_dispatch_mlp_combine_forward
        elif first == "from-import":
            from cudnn.moe_dispatch_mlp_combine import moe_dispatch_mlp_combine_forward as forward
        from cudnn.moe_dispatch_mlp_combine import api
        import cudnn.ops
        assert forward is api.moe_dispatch_mlp_combine_forward
        for name in api.__all__:
            assert getattr(package, name) is getattr(api, name), name
            assert getattr(cudnn.ops, name) is getattr(api, name), name
            assert name in cudnn.ops.__all__, name
        print("OK")
    """)
    assert out.startswith("OK")


def _host_plan(precision: str):
    """A plan object for the declines, which are raised before the plan's buffers are touched."""
    fields = {field.name: None for field in dataclasses.fields(functional.CuteMoKPlan)}
    return functional.CuteMoKPlan(**fields | {"routed_precision": precision, "_views": {}})


def test_declines():
    """An input the execute steps do not read natively is declined with NotImplementedError naming it."""
    tokens, plan = 512, _host_plan("bf16")
    x = torch.zeros(tokens, 256, dtype=torch.bfloat16)
    top = torch.zeros(tokens, 2, dtype=torch.int32)
    router_weights = torch.zeros(tokens, 2, dtype=torch.float32)
    routed = [torch.zeros(1, 256, 256, dtype=torch.bfloat16)] * 3
    buffers = dict(plan=plan, config=functional.CuteMoKConfig(), out=None, context_out=(), schedule_out=())

    def forward(**changes):
        arguments = dict(x=x, top_experts=top, router_weights=router_weights) | changes
        api.moe_dispatch_mlp_combine_forward(
            arguments["x"], arguments["top_experts"], arguments["router_weights"], None, None, None, *arguments.get("routed", routed), **buffers
        )

    for changes, name in (
        (dict(x=x.float()), "x"),
        (dict(x=x.t().contiguous().t()), "x"),
        (dict(top_experts=top.long()), "top_experts"),
        (dict(router_weights=router_weights.bfloat16()), "router_weights"),
        (dict(routed=[(w, w) for w in routed]), "routed_gate_weights"),
        (dict(routed=[w.float() for w in routed]), "routed_gate_weights"),
    ):
        with pytest.raises(NotImplementedError, match=name):
            forward(**changes)
    with pytest.raises(NotImplementedError, match="routed_gate_weights: bf16 routed weights on a plan prepared"):
        api.moe_dispatch_mlp_combine_forward(x, top, router_weights, None, None, None, *routed, **buffers | dict(plan=_host_plan("mxfp8")))
    with pytest.raises(NotImplementedError, match="grad_output"):
        api.moe_dispatch_mlp_combine_backward(
            x.float(), x, router_weights, None, None, None, *routed, None, None, plan=plan, config=buffers["config"], out=None, d_router_weights_out=None
        )


@pytest.mark.parametrize("precision", ["bf16", "mxfp8"])
def test_matches_execute_entries(context, precision):
    rank, world_size, device = context
    if torch.cuda.get_device_capability(device) != TARGET_CAPABILITY:
        pytest.skip("CuTe DSL kernels")
    config = api.MoeDispatchMlpCombineConfig(
        fwd_num_comm_sms=8, bwd_num_comm_sms=8, minibatch_size=1024, macrobatch_size=4096, schedule_capacity_multiplier=1.5
    )
    plan = api.moe_dispatch_mlp_combine_prepare(
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
    assert isinstance(plan, functional.CuteMoKPlan)
    tokens = MAX_TOKENS // 2
    inputs = generate_inputs(
        rank,
        device,
        40,
        num_experts=LOCAL_EXPERTS * world_size,
        local_experts=LOCAL_EXPERTS,
        topk=TOPK,
        tokens=tokens,
        hidden=HIDDEN,
        intermediate=INTERMEDIATE,
    )
    x, top, router_weights, d_output = inputs[0], inputs[1].to(torch.int32), inputs[2], inputs[-1]
    fw, bw, _, _, _ = call_arguments("plain", precision, inputs[3:6], inputs[6:9])

    context_shapes, out_shape = plan.forward_shapes(config, tokens)
    d_x_shape, d_router_shape, d_weight_shapes = plan.backward_shapes(tokens)
    schedule_out = _empty(plan.schedule_shapes(tokens), device, torch.int32)
    context_out, out = _empty(context_shapes, device), _empty(out_shape, device)
    d_x_out, d_router_out = _empty(d_x_shape, device), _empty(d_router_shape, device, torch.float32)
    d_weights_out = _empty(d_weight_shapes, device)

    def results(output, schedule, forward_context, grads):
        return [
            t.clone()
            for t in [
                schedule.num_tokens,
                schedule.tokens_per_expert,
                schedule.valid_tokens_per_expert,
                *_forward_results(output, forward_context),
                *list(grads)[:8],
            ]
        ]

    schedule = functional.schedule_execute(plan, config, top, out=schedule_out)
    output, forward_context = functional.forward_execute(plan, config, schedule, x, router_weights, *fw, None, context_out=context_out, out=out)
    grads = functional.backward_execute(
        plan,
        config,
        schedule,
        forward_context,
        d_output,
        x,
        router_weights,
        *bw,
        None,
        out=d_x_out,
        d_router_weights_out=d_router_out,
        d_weights_out=d_weights_out,
    )
    expected = results(output, schedule, forward_context, grads)

    # The same buffers again, through the operator functions: whatever they allocate or compile is their own.
    executors = _executor_counts()
    allocations = torch.cuda.memory_stats(device)["allocation.all.allocated"]
    forward_results = api.moe_dispatch_mlp_combine_forward(
        x, top, router_weights, *fw, plan=plan, config=config, out=out, context_out=context_out, schedule_out=schedule_out
    )
    output, schedule, forward_context = forward_results
    grads = api.moe_dispatch_mlp_combine_backward(
        d_output,
        x,
        router_weights,
        *bw,
        schedule,
        forward_context,
        plan=plan,
        config=config,
        out=d_x_out,
        d_router_weights_out=d_router_out,
        d_weights_out=d_weights_out,
    )
    assert torch.cuda.memory_stats(device)["allocation.all.allocated"] == allocations, f"{precision}: the operator functions allocated device memory"
    assert _executor_counts() == executors, f"{precision}: the operator functions compiled a kernel"
    assert output is out and schedule.peer_rank is schedule_out[0] and grads[0] is d_x_out
    assert isinstance(forward_results, TupleDict) and list(forward_results.keys()) == ["output", "schedule", "forward_context"]
    assert isinstance(grads, TupleDict) and tuple(grads.keys()) == (*BACKWARD_RESULT_NAMES, "d_w_output_gate")
    assert grads["d_x"] is d_x_out and grads["d_w_output_gate"] is None
    _same(expected, results(output, schedule, forward_context, grads), f"{precision}: operator vs execute")
    dist.barrier()
