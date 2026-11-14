# Copyright 2026 Cursor Research
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Derived from Mixture-of-Kittens (https://github.com/cursor/mixture-of-kittens, commit 22fc95ae); modified by NVIDIA.

"""The megakernels against the PyTorch reference, and against their own previous run.

Run under torchrun with eight GPUs on SM100 / SM103, e.g.
``torchrun --standalone --nproc-per-node=8 -m pytest -s moe_dispatch_mlp_combine/cutedsl/test_cutedsl_functional.py``.
The CuTe DSL ports are SM100 / SM103 only, so the tests skip elsewhere. Every
(shape, MoK parameter, SwiGLU) combination of the test matrix is checked against
the PyTorch reference with ``BF16_TOLERANCE``, and every backward is run twice and
required to be bit-identical.
"""

import contextlib
import dataclasses
import itertools
import unittest.mock

import pytest  # noqa: F401  (run under torchrun + pytest, see module docstring)
import torch
import torch.distributed as dist

import cudnn.moe_dispatch_mlp_combine
from cudnn.moe_dispatch_mlp_combine import functional
from cudnn.moe_dispatch_mlp_combine.functional import get_workspace
from cudnn.moe_dispatch_mlp_combine.cutedsl._arch import SUPPORTED_CAPABILITIES
from cudnn.moe_dispatch_mlp_combine.ops import mxfp8_quantize

from .utils import (
    ungated,
    BACKWARD_RESULT_NAMES,
    FORWARD_RESULT_NAMES,
    BF16_TOLERANCE,
    MXFP8_TOLERANCE,
    capacity_multiplier,
    check_correctness,
    generate_inputs,
    mok_params,
    mxfp8_tolerance,
    run_forward_reference_bf16,
    run_fwd_epilogue_reference,
    run_routed_topk_reduce_reference,
    run_reference_bf16,
    shapes,
    swiglu_params,
)

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


def test_cutedsl_latent_layer_mxfp8(context: tuple[int, int, torch.device]) -> None:
    """``forward_latent_layer`` / ``backward_latent_layer`` with MXFP8 routed weights (CUDA's tuples from
    ``mxfp8_quantize``), the BF16 latent layer's interface: x -> latent GEMM -> mega (shared in hidden, routed in
    latent on the CuTe DSL MXFP8 forward / backward) -> routed top-k reduction -> latent -> hidden GEMM + y_shared,
    and the backward of that composition.  Against the torch BF16 reference with the unquantized routed weights,
    within ``MXFP8_TOLERANCE``.  The layer's routed chain is a plain layer in the routed hidden; where upstream
    Mixture-of-Kittens is outside the absolute bound on that plain layer, the routed gradient's bound is
    upstream's abs max (``mxfp8_tolerance``, validate/README.md).  Qwen 4096 -> 1024 (one macrobatch) and
    DeepSeek 7168 -> 3584 (multiple macrobatches)."""
    rank, world_size, device = context
    _skip_unless_supported(world_size, device)
    swiglu_param_name, swiglu_limit = swiglu_params()[1]
    minimum_macro = next(p for p in mok_params() if p[0] == "Minimum macrobatch")
    cases = ((shapes(world_size)[2], 4, mok_params()[0]), (shapes(world_size)[3], 2, minimum_macro))
    for shape, divisor, mok_param in cases:
        shape_name, num_experts, hidden_dim, intermediate_dim, topk, num_local_tokens = shape
        routed_hidden_dim = hidden_dim // divisor
        num_local_experts = num_experts // world_size
        label = f"cutedsl-latent-layer-mxfp8/{shape_name}/{routed_hidden_dim}/{swiglu_param_name}"
        config = _config(mok_param, world_size, shape)
        workspace = functional.create_workspace(
            config, dist.group.WORLD, device=device, num_local_tokens=num_local_tokens, hidden_size=hidden_dim, topk=topk, routed_hidden_size=routed_hidden_dim
        )
        hidden_inputs = generate_inputs(rank, device, num_experts, num_local_experts, topk, num_local_tokens, hidden_dim, intermediate_dim)
        routed_inputs = generate_inputs(rank, device, num_experts, num_local_experts, topk, num_local_tokens, routed_hidden_dim, intermediate_dim)
        x, topk_experts, router_weights, w_shared_gate, w_shared_up, w_shared_down = hidden_inputs[:6]
        w_routed_gate, w_routed_up, w_routed_down = routed_inputs[6:9]
        # CUDA's tuples: the forward takes (e4m3, scale), the backward Gate / Up (e4m3, scale, e4m3_t, scale_t) and
        # the transposed Down (e4m3_t, scale_t).
        gate_q, up_q, down_q = (tuple(mxfp8_quantize(weight, True, True)) for weight in (w_routed_gate, w_routed_up, w_routed_down))
        routed_mxfp8 = [gate_q[:2], up_q[:2], down_q[:2]]
        generator = torch.Generator(device=device).manual_seed(1234)
        w_latent_down = (torch.randn(routed_hidden_dim, hidden_dim, device=device, generator=generator) / hidden_dim**0.5).to(torch.bfloat16)
        w_latent_up = (torch.randn(hidden_dim, routed_hidden_dim, device=device, generator=generator) / routed_hidden_dim**0.5).to(torch.bfloat16)
        schedule = functional.build_schedule(workspace, config, topk_experts, num_local_experts=num_local_experts)
        output, layer_context = functional.forward_latent_layer(
            config, workspace, schedule, x, router_weights, w_latent_down, w_latent_up, w_shared_gate, w_shared_up, w_shared_down, *routed_mxfp8, swiglu_limit
        )
        assert tuple(output.shape) == (num_local_tokens, hidden_dim)
        # The MXFP8 forward context in the routed hidden: the transposed e4m3 x and its scales; the routed Gate / Up
        # context e4m3 with its scales (the default gate_up_context="mxfp8").
        x_fp8_t, x_sc_t = layer_context.mok.x_routed
        assert tuple(x_fp8_t.shape) == (routed_hidden_dim, config.macrobatch_size)
        assert x_fp8_t.dtype == torch.float8_e4m3fn
        assert tuple(x_sc_t.shape) == (routed_hidden_dim // 128, config.macrobatch_size // 128, 32, 16)
        gate_fp8_routed, gate_sc_routed = layer_context.mok.gate_routed
        assert gate_fp8_routed.dtype == torch.float8_e4m3fn and gate_sc_routed.dtype == torch.uint8
        assert tuple(layer_context.routed_input.shape) == (num_local_tokens, routed_hidden_dim)
        assert tuple(layer_context.y_routed.shape) == (num_local_tokens, routed_hidden_dim)
        routed_input = torch.matmul(x, w_latent_down.t())
        reference_combine, *_, reference_y_shared = run_forward_reference_bf16(
            x, topk_experts, w_shared_gate, w_shared_up, w_shared_down, w_routed_gate, w_routed_up, w_routed_down, swiglu_limit, routed_input=routed_input
        )
        reference_y_routed = run_routed_topk_reduce_reference(reference_combine, router_weights)
        reference = torch.addmm(reference_y_shared, reference_y_routed, w_latent_up.t())
        check_correctness(f"{label}/output vs reference", reference, output, MXFP8_TOLERANCE, print_stats=rank == 0)
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
            gate_q,
            up_q,
            down_q[2:],
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
        case = f"{shape_name}, routed hidden {routed_hidden_dim}"
        for name, expected, actual in zip(names, reference_grads, layer_grads, strict=True):
            check_correctness(
                f"{label}/{name} vs reference", expected, actual, mxfp8_tolerance(world_size, case, swiglu_param_name, name), print_stats=rank == 0
            )
        del output, layer_context, reference, reference_combine, reference_y_routed, reference_y_shared
        del layer_grads, moe, moe_rest, reference_grads, reference_d_x_shared, reference_d_routed_input
        del hidden_inputs, routed_inputs, routed_mxfp8, gate_q, up_q, down_q, schedule
        torch.cuda.empty_cache()


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


# The CUDA bindings the upstream ``functional`` module imports.  The extension is gone; this list keeps the
# property it defended — the call chain reaches no compiled CUDA kernel — checkable.
REMOVED_CUDA_ENTRIES = (
    "dispatch_mlp_swiglu_combine_fwd_bf16",
    "dispatch_mlp_swiglu_combine_bwd_bf16",
    "dispatch_mlp_swiglu_combine_fwd_mxfp8",
    "dispatch_mlp_swiglu_combine_bwd_mxfp8",
    "dispatch_mlp_swiglu_combine_bf16_fwd_naive",
    "dispatch_mlp_swiglu_combine_bf16_bwd_naive",
    "dispatch_mlp_swiglu_combine_mxfp8_fwd_naive",
    "dispatch_mlp_swiglu_combine_mxfp8_bwd_naive",
    "recompute_forward_context_bf16",
    "recompute_forward_context_mxfp8",
    "forward_naive",
    "backward_naive",
)


def _cutedsl_only_case(context) -> None:
    """Whole call chain (build_schedule, forward, backward) on CuTe DSL kernels only."""
    rank, world_size, device = context
    _skip_unless_supported(world_size, device)
    shape = shapes(world_size)[2]  # Qwen3.5-397B-A17B
    mok_param = mok_params()[0]
    _, num_experts, hidden_dim, intermediate_dim, topk, num_local_tokens = shape
    num_local_experts = num_experts // world_size
    swiglu_limit = swiglu_params()[1][1]
    config = _config(mok_param, world_size, shape)
    workspace = get_workspace(config, dist.group.WORLD, device=device, num_local_tokens=num_local_tokens, hidden_size=hidden_dim, topk=topk)
    inputs = generate_inputs(rank, device, num_experts, num_local_experts, topk, num_local_tokens, hidden_dim, intermediate_dim)
    schedule = functional.build_schedule(workspace, config, inputs[1], num_local_experts=num_local_experts)
    dsl_forward, dsl_grads = _run(config, workspace, schedule, inputs, swiglu_limit)
    reference_combine, reference_gate, reference_up, reference_hidden, reference_y_shared = run_forward_reference_bf16(
        inputs[0], inputs[1], *inputs[3:9], swiglu_limit
    )
    reference_forward = (run_fwd_epilogue_reference(reference_y_shared, reference_combine, inputs[2]), reference_gate, reference_up, reference_hidden)
    reference_grads = run_reference_bf16(*inputs, swiglu_limit)[1:]
    for name, expected, dsl_result in zip(
        FORWARD_RESULT_NAMES + BACKWARD_RESULT_NAMES, reference_forward + tuple(reference_grads), dsl_forward + tuple(dsl_grads), strict=True
    ):
        check_correctness(f"cutedsl-only/{shape[0]}/{name} vs reference", expected, dsl_result, BF16_TOLERANCE, print_stats=rank == 0)


def test_cutedsl_path_uses_no_cuda_kernels(context: tuple[int, int, torch.device]) -> None:
    """The whole call chain runs on the CuTe DSL kernels."""
    for name in REMOVED_CUDA_ENTRIES:
        assert not hasattr(functional, name), f"functional.{name} is a CUDA entry point and was removed"
    assert not hasattr(cudnn.moe_dispatch_mlp_combine, "_C"), "the compiled CUDA extension was removed"
    _cutedsl_only_case(context)


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
