# Copyright 2026 Cursor Research
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Derived from Mixture-of-Kittens (https://github.com/cursor/mixture-of-kittens, commit 22fc95ae); modified by NVIDIA.

import gc
import os
from collections.abc import Iterator
from pathlib import Path

import pytest
import torch
import torch.distributed as dist

from cudnn.moe_ep.cutemok import functional
from cudnn.moe_ep.cutemok.functional import clear_workspace_cache
from cudnn.moe_ep.cutemok.cutedsl._version_gate import CuteDslRequirementError


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_makereport(item, call):
    """A test that reaches the kernels under an unsupported CuTe DSL (older than the floor, or its cu12 build) is
    skipped with the version gate's message instead of failing."""
    outcome = yield
    report = outcome.get_result()
    if call.excinfo is not None and call.excinfo.errisinstance(CuteDslRequirementError):
        report.outcome = "skipped"
        report.longrepr = (str(item.path), item.location[1] or 0, f"Skipped: {call.excinfo.value}")


_TORCHRUN_VARIABLES = ("RANK", "WORLD_SIZE", "LOCAL_RANK")


@pytest.hookimpl(tryfirst=True)
def pytest_collection_modifyitems(config, items):
    """Every test of this directory is L0 (``pytest.ini`` selects ``-m L0`` by default); marked before the mark
    plugin deselects.  The tests that need an EP group skip without one (the ``context`` fixture)."""
    here = Path(__file__).resolve().parent
    for item in items:
        if here in Path(str(item.path)).resolve().parents:
            item.add_marker(pytest.mark.L0)


@pytest.fixture(scope="session")
def context() -> Iterator[tuple[int, int, torch.device]]:
    """(rank, world size, device) of a torchrun process group; tests that take it skip on a node without a GPU and
    outside torchrun (``torchrun --nproc-per-node=<ranks> -m pytest moe_ep/cutemok/<file>``)."""
    if not torch.cuda.is_available():
        pytest.skip("needs a CUDA GPU (run under torchrun on a GPU node)")
    missing = [name for name in _TORCHRUN_VARIABLES if name not in os.environ]
    if missing:
        pytest.skip("needs a torchrun process group; not set: " + ", ".join(missing))
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])

    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)
    dist.init_process_group(
        backend="nccl",
        rank=rank,
        world_size=world_size,
        device_id=device,
    )

    try:
        yield rank, world_size, device
    finally:
        dist.barrier()
        clear_workspace_cache()
        gc.collect()
        dist.destroy_process_group()


@pytest.fixture(autouse=True)
def _release_workspaces() -> Iterator[None]:
    """Release the cached workspaces after every test.  A test module runs many tests in one process; each workspace
    holds symmetric memory with multicast addresses, and an EP group runs out of them when the workspaces of many
    shapes stay cached until the session ends.  The compiled executors stay cached."""
    yield
    if not (dist.is_available() and dist.is_initialized()) or not functional._WORKSPACE_CACHE:
        return
    from cudnn.moe_ep.cutemok.cutedsl.barrier_all import barrier_all

    for workspace in functional._WORKSPACE_CACHE.values():
        barrier_all(workspace.barrier_buffer, workspace.barrier_buffer_ptrs, workspace.barrier_buffer_multicast_ptr, workspace.barrier_target)
        torch.cuda.synchronize(workspace.device)
    functional._WORKSPACE_CACHE.clear()
    gc.collect()
