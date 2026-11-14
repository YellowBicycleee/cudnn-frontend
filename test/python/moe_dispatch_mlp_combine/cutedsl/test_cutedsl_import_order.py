# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CUTLASS may be imported before the operator's package, with ``CUTE_DSL_ARCH`` and ``CUTE_DSL_ENABLE_TVM_FFI`` unset: the target
and TVM FFI are compile options (``_arch.COMPILE_OPTIONS``), not process environment.

Each case runs in a fresh interpreter (CuTe DSL reads both variables when CUTLASS is first imported) with both
variables removed from its environment.  CPU: a stand-in ``cutlass`` imported first, then the kernel hosts -- they
import, set no variable and carry the target in their compile options.  GPU (SM100 / SM103, one device): the real
``cutlass.cute`` imported first, then ``cudnn.moe_dispatch_mlp_combine.ops.schedule`` compiles and runs, its executor is a TVM FFI function
and its per-expert row counts match the routes.
"""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap

import pytest
import torch

VARIABLES = ("CUTE_DSL_ARCH", "CUTE_DSL_ENABLE_TVM_FFI")


def _run(code: str) -> str:
    env = {name: value for name, value in os.environ.items() if name not in VARIABLES}
    result = subprocess.run([sys.executable, "-c", textwrap.dedent(code)], env=env, capture_output=True, text=True, timeout=900)
    assert result.returncode == 0, result.stdout + result.stderr
    return result.stdout


def test_hosts_import_after_cutlass_without_the_variables():
    out = _run("""
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
    out = _run("""
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
