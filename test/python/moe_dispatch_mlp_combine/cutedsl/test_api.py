# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The operator API and its gates: names, signatures and lazy export, plan / execute, the launch-plan fast
paths, the CuTe DSL version gate, the target architecture and the import order.

CPU tests run under plain pytest; the GPU tests run under torchrun on SM100 / SM103, e.g.
``torchrun --standalone --nproc-per-node=8 -m pytest -s moe_dispatch_mlp_combine/cutedsl/test_api.py``.
"""

from __future__ import annotations

import contextlib
import dataclasses
import inspect
import os
import re
import subprocess
import sys
import textwrap
import types
import unittest
from pathlib import Path
from unittest import mock

import pytest
import torch
import torch.distributed as dist

import cudnn.moe_dispatch_mlp_combine
from cudnn._tupledict import TupleDict
from cudnn.frost import buffers
from cudnn.moe_dispatch_mlp_combine import api, functional
from cudnn.moe_dispatch_mlp_combine.cutedsl import _arch
from cudnn.moe_dispatch_mlp_combine.cutedsl._arch import TARGET_CAPABILITY

from ._cutedsl_by_path import load_cutedsl_module
from .call_helpers import call_arguments, generate_inputs, shared_output_gate_weight
from .utils import BACKWARD_RESULT_NAMES

# --------------------------------------------------------------------------------------------------------------------
# CuTe DSL version gate
#
# CPU: the operator's CuTe DSL gate (``cudnn/moe_dispatch_mlp_combine/cutedsl/_version_gate.py``) accepts a CuTe DSL
# at or above its floor (4.8.0) in the cu13 build and declines anything else with the installed and the required
# version, through cuDNN frontend's gate (``cudnn.frost.buffers``, ``python/cudnn/AGENTS.md`` Rule 7).


gate = load_cutedsl_module("_version_gate")


class Floor(unittest.TestCase):
    def test_floor_is_at_least_the_library_floor(self):
        self.assertEqual(gate.CUTEDSL_FLOOR, (4, 8, 0))
        self.assertGreaterEqual(gate.CUTEDSL_FLOOR, tuple(buffers.CUTEDSL_MIN_VERSION))

    def test_cu13_or_unknown_flavor_passes(self):
        self.assertIsNone(gate.flavor_error(13))
        self.assertIsNone(gate.flavor_error(None))  # flavor not known

    def test_cu12_build_is_declined_with_the_reason(self):
        message = gate.flavor_error(12)
        self.assertIn("cu12", message)
        self.assertIn("nvidia-cutlass-dsl[cu13]>=4.8.0", message)


class RequireCutedsl(unittest.TestCase):
    def _require(self, dsl_version, cuda_major, distribution="nvidia-cutlass-dsl"):
        version_info = types.ModuleType("cutlass.base_dsl.version_info")
        version_info.CUDA_VERSION = types.SimpleNamespace(major=cuda_major, minor=0)
        modules = {
            "cutlass": types.ModuleType("cutlass"),
            "cutlass.base_dsl": types.ModuleType("cutlass.base_dsl"),
            "cutlass.base_dsl.version_info": version_info,
        }
        state = (dsl_version is not None, None if dsl_version is None else (distribution, dsl_version))
        with (
            mock.patch.object(gate, "_VERSION", None),
            mock.patch.object(buffers, "_DSL_STATE", state),
            mock.patch.dict(sys.modules, modules),
        ):
            return gate.require_cutedsl()

    def test_supported_install_returns_the_version(self):
        for dsl in ("4.8.0", "4.8.1", "4.9.0", "4.10.0", "5.0.0", "4.8.0.dev1", "4.8.0+local"):
            self.assertEqual(self._require(dsl, 13), dsl)

    def test_older_release_names_both_versions(self):
        for dsl in ("4.5.2", "4.7.0", "4.7.9", "3.0.0"):
            with self.assertRaises(gate.CuteDslRequirementError) as raised:
                self._require(dsl, 13)
            message = str(raised.exception)
            self.assertIn("moe_dispatch_mlp_combine requires nvidia-cutlass-dsl >= 4.8.0", message)
            self.assertIn(f"found {dsl}", message)

    def test_missing_dsl_is_declined(self):
        with self.assertRaisesRegex(gate.CuteDslRequirementError, "not installed"):
            self._require(None, 13)

    def test_cu12_build_raises_before_any_jit(self):
        with self.assertRaisesRegex(gate.CuteDslRequirementError, "cu12"):
            self._require("4.8.0", 12)


# --------------------------------------------------------------------------------------------------------------------
# Target architecture
#
# CPU: the Blackwell target of the CuTe DSL kernels (``cudnn/moe_dispatch_mlp_combine/cutedsl/_arch.py``): the device's compute capability
# picks sm_100a / sm_103a, a ``CUTE_DSL_ARCH`` naming another part is declined, and the target reaches CuTe DSL through
# the compile options, not through the environment.


class ArchResolution(unittest.TestCase):
    def test_device_picks_the_part(self):
        self.assertEqual(_arch.resolve(None, (10, 0)), "sm_100a")
        self.assertEqual(_arch.resolve(None, (10, 3)), "sm_103a")

    def test_no_or_unsupported_device_falls_back_to_the_default(self):
        self.assertEqual(_arch.resolve(None, None), _arch.DEFAULT_ARCH)
        self.assertEqual(_arch.resolve(None, (9, 0)), _arch.DEFAULT_ARCH)
        self.assertEqual(_arch.resolve(None, (12, 0)), _arch.DEFAULT_ARCH)
        self.assertEqual(_arch.resolve("sm_100a", None), _arch.DEFAULT_ARCH)  # nothing to launch on: not used

    def test_matching_request_is_accepted(self):
        self.assertEqual(_arch.resolve("sm_100a", (10, 0)), "sm_100a")
        self.assertEqual(_arch.resolve("sm_103a", (10, 3)), "sm_103a")

    def test_request_for_another_part_is_declined_with_the_reason(self):
        for requested, capability in (
            ("sm_100a", (10, 3)),
            ("sm_103a", (10, 0)),
            ("sm_103", (10, 3)),
            ("sm_100f", (10, 3)),
            ("sm_90a", (10, 3)),
            ("", (10, 0)),
        ):
            with self.assertRaises(RuntimeError) as ctx:
                _arch.resolve(requested, capability)
            message = str(ctx.exception)
            self.assertIn(f"{_arch.ARCH_ENV}={requested}", message)
            self.assertIn(_arch.SUPPORTED_CAPABILITIES[capability], message)

    def test_target_and_tvm_ffi_are_compile_options(self):
        self.assertIn(_arch.TARGET_ARCH, _arch.CAPABILITY_BY_ARCH)
        self.assertEqual(_arch.SUPPORTED_CAPABILITIES[_arch.TARGET_CAPABILITY], _arch.TARGET_ARCH)
        self.assertEqual(set(_arch.SUPPORTED_CAPABILITIES.values()), {"sm_100a", "sm_103a"})
        options = _arch.COMPILE_OPTIONS.split()
        self.assertEqual(options[options.index("--gpu-arch") + 1], _arch.TARGET_ARCH)
        self.assertIn("--enable-tvm-ffi", options)
        self.assertEqual(options[options.index("--opt-level") + 1], "2")


class DeviceCheck(unittest.TestCase):
    def test_check_device_rejects_another_part(self):
        with mock.patch.object(_arch.torch.cuda, "get_device_capability", return_value=(9, 0)):
            with self.assertRaises(NotImplementedError) as ctx:
                _arch.check_device(0, "the test kernel")
        message = str(ctx.exception)
        self.assertIn(_arch.TARGET_ARCH, message)
        self.assertIn("(9, 0)", message)
        with mock.patch.object(_arch.torch.cuda, "get_device_capability", return_value=_arch.TARGET_CAPABILITY):
            _arch.check_device(0, "the test kernel")  # no raise


# --------------------------------------------------------------------------------------------------------------------
# Import order and the process environment
#
# CUTLASS may be imported before the operator's package, with ``CUTE_DSL_ARCH`` and ``CUTE_DSL_ENABLE_TVM_FFI`` unset:
# the target
# and TVM FFI are compile options (``_arch.COMPILE_OPTIONS``), not process environment.
#
# Each case runs in a fresh interpreter (CuTe DSL reads both variables when CUTLASS is first imported) with both
# variables removed from its environment.  CPU: a stand-in ``cutlass`` imported first, then the kernel hosts -- they
# import, set no variable and carry the target in their compile options.  GPU (SM100 / SM103, one device): the real
# ``cutlass.cute`` imported first, then ``cudnn.moe_dispatch_mlp_combine.ops.schedule`` compiles and runs, its executor is a TVM FFI function
# and its per-expert row counts match the routes.


VARIABLES = ("CUTE_DSL_ARCH", "CUTE_DSL_ENABLE_TVM_FFI")


def _run_without_dsl_variables(code: str) -> str:
    env = {name: value for name, value in os.environ.items() if name not in VARIABLES}
    result = subprocess.run([sys.executable, "-c", textwrap.dedent(code)], env=env, capture_output=True, text=True, timeout=900)
    assert result.returncode == 0, result.stdout + result.stderr
    return result.stdout


def test_hosts_import_after_cutlass_without_the_variables():
    out = _run_without_dsl_variables("""
        import os, sys, types
        sys.modules["cutlass"] = types.ModuleType("cutlass")  # CUTLASS is "already imported"
        from cudnn.moe_dispatch_mlp_combine.cutedsl import _arch, persistent_bf16_bwd, persistent_bf16_fwd, persistent_mxfp8_bwd
        from cudnn.moe_dispatch_mlp_combine.cutedsl import persistent_mxfp8_fwd  # noqa: F401
        assert not {"CUTE_DSL_ARCH", "CUTE_DSL_ENABLE_TVM_FFI"} & set(os.environ), "a CuTe DSL variable was set"
        assert f"--gpu-arch {_arch.TARGET_ARCH}" in _arch.COMPILE_OPTIONS and "--enable-tvm-ffi" in _arch.COMPILE_OPTIONS
        print("OK", _arch.COMPILE_OPTIONS)
    """)
    assert out.startswith("OK")


def _blackwell() -> bool:
    return torch.cuda.is_available() and torch.cuda.get_device_capability() in ((10, 0), (10, 3))


@pytest.mark.skipif(not _blackwell(), reason="the CuTe DSL kernels are SM100 / SM103 only")
def test_kernel_compiles_and_runs_after_cutlass_without_the_variables():
    out = _run_without_dsl_variables("""
        import os, sys
        import cutlass.cute  # noqa: F401  first, before the operator, with neither variable set
        import torch
        from cutlass.cutlass_dsl.tvm_ffi_provider import TVMFFIJitCompiledFunctionBase
        from cudnn.moe_dispatch_mlp_combine import ops
        from cudnn.moe_dispatch_mlp_combine.cutedsl._version_gate import CuteDslRequirementError, require_cutedsl

        try:
            require_cutedsl()
        except CuteDslRequirementError as error:
            print("SKIP", error)
            sys.exit(0)
        from cudnn.moe_dispatch_mlp_combine.cutedsl import _arch, scheduler
        experts, tokens, topk = 8, 512, 4
        device = torch.device("cuda", torch.cuda.current_device())
        generator = torch.Generator(device=device).manual_seed(0)
        routes = torch.randint(0, experts, (1, tokens, topk), dtype=torch.int32, device=device, generator=generator)
        valid = ops.schedule(routes, experts, 2 * tokens * topk, 0)[4]  # capacity: the factor-2 floor of a workspace
        torch.cuda.synchronize(device)
        assert valid.tolist() == torch.bincount(routes.flatten(), minlength=experts).tolist(), valid
        executors = list(scheduler._EXECUTOR_CACHE.values())
        # TVM FFI executors (CuTe DSL 4.8 compiles a signature with defaults to its kwargs-wrapper subclass, which is
        # not a tvm_ffi.Function itself), not the DSL's own call path
        assert executors and all(isinstance(e.executor, TVMFFIJitCompiledFunctionBase) for e in executors), executors
        assert not {"CUTE_DSL_ARCH", "CUTE_DSL_ENABLE_TVM_FFI"} & set(os.environ)
        print("OK", _arch.COMPILE_OPTIONS)
    """)
    if out.startswith("SKIP"):
        pytest.skip(out[5:].strip())
    assert out.startswith("OK"), out


# --------------------------------------------------------------------------------------------------------------------
# Plan / execute
#
# The plan / execute form of a layer: ``schedule_execute`` / ``forward_execute`` / ``backward_execute`` on a
# ``make_plan`` match ``build_schedule`` / ``forward`` / ``backward`` bit for bit.
#
# The plan compiles every kernel of its configuration and owns the counters and the scratch; the execute steps take
# the caller's output buffers, compile nothing, allocate nothing and never wait for the device (no device-to-host read,
# no synchronization), so the first execute after a plan can be captured in a CUDA graph.  One plan serves every token
# count up to its maximum.  BF16 and MXFP8 routed weights (the MXFP8 Gate / Up context in both precisions).


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


# --------------------------------------------------------------------------------------------------------------------
# Operator API
#
# The operator API (``cudnn.moe_dispatch_mlp_combine.api``): ``moe_dispatch_mlp_combine_{prepare, forward, backward}``.
#
# CPU: the names follow cuDNN frontend's op naming rules, the signatures (data tensors positional, everything else
# keyword-only; ``prepare`` takes exactly ``make_plan``'s arguments), the lazy export from the package and from
# ``cudnn.ops`` in every import order, the results' names, and the decline of an unsupported layout or dtype before
# anything is launched.
# GPU: the operator functions return what ``schedule_execute`` + ``forward_execute`` and ``backward_execute`` return,
# bit for bit, for BF16 and MXFP8 routed weights, compiling nothing and allocating no device memory.


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


def test_cutedsl_path_uses_no_cuda_kernels() -> None:
    """No CUDA entry point or compiled extension is left (the call chain on the CuTe DSL kernels runs in
    test_backward.py's test_cutedsl_forward_backward_bf16)."""
    for name in REMOVED_CUDA_ENTRIES:
        assert not hasattr(functional, name), f"functional.{name} is a CUDA entry point and was removed"
    assert not hasattr(cudnn.moe_dispatch_mlp_combine, "_C"), "the compiled CUDA extension was removed"


# --------------------------------------------------------------------------------------------------------------------
# Launch-plan fast paths
#
# The launch-plan fast paths: calls that reuse a plan match calls that build it, bit for bit.
#
# A plan is kept per workspace, weights and geometry; later calls only check their per-call tensors' layouts,
# allocate their state and fill the plan's argument template.  MCore passes new Gate / Up row views of its shared
# FC1 (and of its main_grad) on every call, so the weights are matched by address and layout, not identity: the
# caches must not grow with the number of calls.


TOKENS = 1024  # LOCAL_EXPERTS, HIDDEN, INTERMEDIATE and TOPK: the plan / execute section's


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


def test_kernel_sources_have_no_quack_imports():
    package_root = Path(cudnn.moe_dispatch_mlp_combine.__file__).resolve().parent
    sources = "\n".join(path.read_text(encoding="utf-8") for path in package_root.rglob("*.py"))
    assert re.search(r"^\s*(?:from|import)\s+quack\b", sources, flags=re.MULTILINE) is None
    assert "quack." not in sources
