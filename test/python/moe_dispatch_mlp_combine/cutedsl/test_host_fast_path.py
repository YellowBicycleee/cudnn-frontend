# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The launch-plan fast paths: calls that reuse a plan match calls that build it, bit for bit.

A plan is kept per workspace, weights and geometry; later calls only check their per-call tensors' layouts,
allocate their state and fill the plan's argument template.  MCore passes new Gate / Up row views of its shared
FC1 (and of its main_grad) on every call, so the weights are matched by address and layout, not identity: the
caches must not grow with the number of calls.
"""

import pytest
import torch
import torch.distributed as dist

from cudnn.moe_dispatch_mlp_combine import functional
from cudnn.moe_dispatch_mlp_combine.cutedsl._arch import TARGET_CAPABILITY
from .call_helpers import call_arguments, generate_inputs, shared_output_gate_weight

LOCAL_EXPERTS, HIDDEN, INTERMEDIATE, TOKENS, TOPK = 4, 1024, 512, 1024, 4


def _plan_caches(workspace):
    """Every launch-plan cache a functional call fills, as (name, entries) pairs."""
    from cudnn.moe_dispatch_mlp_combine.cutedsl import barrier_all, persistent_bf16_fwd, persistent_mxfp8_bwd, persistent_mxfp8_fwd

    caches = [("bf16 fwd", persistent_bf16_fwd._LAUNCH_PLANS), ("mxfp8 fwd", persistent_mxfp8_fwd._FAST_PLANS), ("mxfp8 bwd", persistent_mxfp8_bwd._FAST_PLANS)]
    for index, prepared in enumerate((getattr(workspace, "_cutedsl_bwd_preparations", None) or {}).values()):
        caches.append((f"bf16 bwd {index}", prepared.launch_plans))
    return caches, barrier_all._FAST_PLANS


def _clear_plans(workspace):
    caches, barrier = _plan_caches(workspace)
    for _, cache in caches:
        cache.clear()
    barrier.clear()


def _sizes(workspace):
    caches, _ = _plan_caches(workspace)
    return {name: len(cache._entries) for name, cache in caches}


@pytest.mark.parametrize("precision", ["bf16", "mxfp8"])
def test_reused_plans_match_new_plans(context, precision):
    rank, world_size, device = context
    if torch.cuda.get_device_capability(device) != TARGET_CAPABILITY:
        pytest.skip("CuTe DSL kernels")
    config = functional.CuteMoKConfig(fwd_num_comm_sms=8, bwd_num_comm_sms=8, minibatch_size=1024, macrobatch_size=4096, schedule_capacity_multiplier=1.5)
    workspace = functional.create_workspace(config, dist.group.WORLD, device=device, num_local_tokens=TOKENS, hidden_size=HIDDEN, topk=TOPK)
    layers = []
    for layer in range(2):
        inputs = generate_inputs(
            rank,
            device,
            300 + layer,
            num_experts=LOCAL_EXPERTS * world_size,
            local_experts=LOCAL_EXPERTS,
            topk=TOPK,
            tokens=TOKENS,
            hidden=HIDDEN,
            intermediate=INTERMEDIATE,
        )
        gate = shared_output_gate_weight(rank, device, 300 + layer, HIDDEN)
        fw, bw, fk, bk, keepalive = call_arguments("mcore", precision, inputs[3:6], inputs[6:9], output_gate=gate)
        layers.append(dict(fw=fw, bw=bw, fk=fk, bk=bk, keepalive=keepalive, fc1=fw[0]._base, fc1_grad=bk["main_grads"][0]._base))

    def call_weights(layer, which):
        """The call's weights with new shared Gate / Up row views, as MCore makes them on every call."""
        fc1, intermediate = layer["fc1"], INTERMEDIATE
        weights = (fc1.narrow(0, 0, intermediate), fc1.narrow(0, intermediate, intermediate), *layer[which][2:])
        kwargs = dict(layer["fk" if which == "fw" else "bk"])
        if which == "bw":
            grad, main_grads = layer["fc1_grad"], kwargs["main_grads"]
            kwargs["main_grads"] = (grad.narrow(0, 0, intermediate), main_grads[1], grad.narrow(0, intermediate, intermediate), *main_grads[3:])
        return weights, kwargs

    def step(seed):
        x, top, rw, *_, dout = generate_inputs(
            rank, device, seed, num_experts=LOCAL_EXPERTS * world_size, local_experts=LOCAL_EXPERTS, topk=TOPK, tokens=TOKENS, hidden=HIDDEN, intermediate=128
        )
        for layer in layers:
            for grad in (layer["fc1_grad"], layer["bk"]["main_grads"][4], layer["keepalive"][-1], layer["bk"]["shared_output_gate_main_grad"]):
                grad.zero_()
        results, saved = [], []
        for layer in layers:
            schedule = functional.build_schedule(workspace, config, top, num_local_experts=LOCAL_EXPERTS)
            weights, kwargs = call_weights(layer, "fw")
            output, forward_context = functional.forward(config, workspace, schedule, x, rw, *weights, None, **kwargs)
            saved.append((layer, schedule, forward_context))
            results.append(output.clone())
        for layer, schedule, forward_context in reversed(saved):
            weights, kwargs = call_weights(layer, "bw")
            grads = functional.backward(config, workspace, schedule, forward_context, dout, x, rw, *weights, None, **kwargs)
            results += [g.clone() for g in grads[:2]]
        for layer in layers:
            results += [
                layer["fc1_grad"].clone(),
                layer["keepalive"][-1].clone(),
                layer["bk"]["main_grads"][4].clone(),
                layer["bk"]["shared_output_gate_main_grad"].clone(),
            ]
        torch.cuda.synchronize(device)
        return results

    step(10)  # compiles, prepares the backward buffer sets
    for seed in (11, 12):
        _clear_plans(workspace)
        built = step(seed)  # every call builds its plan
        sizes = _sizes(workspace)
        reused = step(seed)  # every call reuses one (new shared views, same addresses)
        again = step(seed + 100)  # new inputs on the kept plans ...
        _clear_plans(workspace)
        fresh = step(seed + 100)  # ... match new plans
        for label, expected, actual in (("reused", built, reused), ("new inputs", fresh, again)):
            for index, (a, b) in enumerate(zip(expected, actual, strict=True)):
                assert torch.equal(a, b), f"{precision} {label}: result {index} differs"
        assert sizes == _sizes(workspace), f"plan caches grew with repeated calls: {sizes} -> {_sizes(workspace)}"
        used = {name: size for name, size in sizes.items() if size}
        expected = {"bf16 fwd": 2, "bf16 bwd 0": 2} if precision == "bf16" else {"mxfp8 fwd": 2, "mxfp8 bwd": 2}
        assert used == expected, f"one kept plan per layer and direction expected: {used}"
    dist.barrier()
