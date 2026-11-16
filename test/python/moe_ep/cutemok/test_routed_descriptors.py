# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""A TMA descriptor per expert reads that expert's own allocation.

This is the mechanism MCore's split routed weights need: the experts are independent allocations
agreeing on dtype, shape and stride, so one descriptor schema fits them all and only the base
address differs.  ``routed_descriptors`` builds the array once and the mainloop names one entry
through ``tma_desc_ptr``.

The test gives every expert a distinguishable pattern and loads a tile from each through the
descriptor array, so a descriptor that kept the atom's original address -- the failure that
matters, since it is silent -- shows up as expert 0's data appearing everywhere.

No ``from __future__ import annotations`` here: the DSL reads a kernel's annotations through
``inspect.signature``, and stringified ones get evaluated against module globals.

GPU only: ``python -m pytest moe_ep/cutemok/test_routed_descriptors.py``
"""

import pytest
import torch

from cudnn.moe_ep.cutemok.cutedsl._version_gate import CuteDslRequirementError, require_cutedsl

try:
    require_cutedsl()
except CuteDslRequirementError as error:  # this module imports CUTLASS and defines kernels
    pytest.skip(str(error), allow_module_level=True)

import cutlass
import cutlass.cute as cute
from cutlass import Int32
from cutlass.cute.nvgpu import cpasync
from cutlass.cute.runtime import from_dlpack

from cudnn.moe_ep.cutemok import ops
from cudnn.moe_ep.cutemok.cutedsl import routed_descriptors as rd
from cudnn.moe_ep.cutemok.cutedsl._host_common import current_stream_handle

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="TMA needs a GPU")

EXPERTS, ROWS, COLUMNS = 5, 128, 64
TILE = (64, 64)
THREADS = 32  # one warp: elect_one issues the copy, the warp copies out


@cute.kernel
def _read_back(atom: cute.CopyAtom, tma_tensor: cute.Tensor, descriptors: cute.Pointer, out: cute.Tensor, expert: Int32, retarget: cutlass.Constexpr[bool]):
    """Load tile (0, 0) through one expert's descriptor and write it out.

    The barrier and arena follow the mega kernels' SwiGLU loads: an mbarrier at the start
    of a 1024-byte aligned dynamic shared-memory region, the payload after it, thread 0 doing
    init -> init_fence -> arrive_and_expect_tx, and then **the whole warp** issuing the copy.

    That last part is what two earlier versions of this file got wrong, at the cost of a day:
    ``cute.copy`` on a TMA atom elects a thread itself and synchronises its warp, so calling it
    under ``if thread == 0`` deadlocks the other 31 threads on a warp sync that the elected thread
    never reaches -- which surfaces as a hang in ``mbarrier_wait``, pointing at the barrier rather
    than at the copy.  Issuing it from all threads of one warp is the only correct form.

    One expert per launch, so the barrier is used once and there is no phase parity to get wrong.
    A kernel is an isolated region, so the layout is rebuilt here from the same constants the host
    used.
    """
    smem_layout = cute.make_layout(TILE, stride=(TILE[1], 1))
    base = cute.arch.get_dyn_smem(cutlass.Uint8, alignment=1024)
    mbar = cute.recast_ptr(base, dtype=cutlass.Int64)
    arena = (base + cutlass.Int64(1024)).align(min_align=1024)
    tile = cute.make_tensor(cute.recast_ptr(arena, dtype=cutlass.BFloat16), smem_layout)

    # One thread initialises and arrives; the whole warp issues the copy (see the docstring).
    thread = cute.arch.thread_idx()[0]
    if thread == Int32(0):
        cute.arch.mbarrier_init(mbar, 1)
    cute.arch.mbarrier_init_fence()
    cute.arch.sync_threads()
    if thread == Int32(0):
        cute.arch.mbarrier_arrive_and_expect_tx(mbar, Int32(TILE[0] * TILE[1] * 2))
    cute.arch.sync_warp()
    global_tile = cute.local_tile(tma_tensor, TILE, (Int32(0), Int32(0)))
    smem_part, global_part = cpasync.tma_partition(
        atom, 0, cute.make_layout(1), cute.group_modes(tile, 0, cute.rank(tile)), cute.group_modes(global_tile, 0, cute.rank(global_tile))
    )
    if cutlass.const_expr(retarget):
        rd.acquire_descriptors(descriptors, Int32(EXPERTS))
        cute.copy(atom, global_part, smem_part, tma_bar_ptr=mbar, tma_desc_ptr=rd.descriptor_argument(descriptors, expert))
    else:
        cute.copy(atom, global_part, smem_part, tma_bar_ptr=mbar)
    cute.arch.mbarrier_wait(mbar, 0)
    cute.arch.fence_proxy("async.shared", space="cta")
    for index in cutlass.range(thread, Int32(TILE[0] * TILE[1]), Int32(THREADS)):
        row = index // Int32(TILE[1])
        column = index % Int32(TILE[1])
        out[row, column] = tile[row, column]


@cute.jit
def _run(
    representative: cute.Tensor,
    addresses: cute.Tensor,
    descriptors_address: cutlass.Int64,
    out: cute.Tensor,
    num_experts: Int32,
    expert: Int32,
    stream,
    retarget: cutlass.Constexpr[bool] = True,
):
    # Layouts and pointers are trace-time objects, so they are built here, not at the call site.
    smem_layout = cute.make_layout(TILE, stride=(TILE[1], 1))
    descriptors = cute.make_ptr(cutlass.Uint8, descriptors_address, cute.AddressSpace.gmem, assumed_align=128)
    atom, tma_tensor = cpasync.make_tiled_tma_atom(cpasync.CopyBulkTensorTileG2SOp(), representative, smem_layout, TILE)
    rd.build_routed_descriptors((atom,), addresses, descriptors, num_experts, stream)
    _read_back(atom, tma_tensor, descriptors, out, expert, retarget).launch(
        grid=(1, 1, 1), block=(THREADS, 1, 1), stream=stream, smem=1024 + TILE[0] * TILE[1] * 2 + 1024
    )


@cute.jit
def _build_only(representative: cute.Tensor, addresses: cute.Tensor, descriptors_address: cutlass.Int64, num_experts: Int32, stream):
    """Just build the array -- no load, so nothing but the retargeting is under test."""
    smem_layout = cute.make_layout(TILE, stride=(TILE[1], 1))
    descriptors = cute.make_ptr(cutlass.Uint8, descriptors_address, cute.AddressSpace.gmem, assumed_align=128)
    atom, _ = cpasync.make_tiled_tma_atom(cpasync.CopyBulkTensorTileG2SOp(), representative, smem_layout, TILE)
    rd.build_routed_descriptors((atom,), addresses, descriptors, num_experts, stream)


def test_build_points_each_descriptor_at_its_expert():
    """Does update_tma_descriptor actually rewrite the base address?

    A TMA descriptor's first eight bytes are the global address of the tensor it describes, so
    the array can be read back on the host and checked directly.  That keeps the one thing this
    module does -- retargeting -- separate from the TMA machinery that reads the result, which is
    what the skipped test above never managed to disentangle.
    """
    experts, addresses = _distinct_experts()
    device = experts[0].device
    descriptors = torch.zeros(rd.descriptor_bytes(EXPERTS), dtype=torch.uint8, device=device)
    _build_only(
        from_dlpack(experts[0], assumed_align=16),
        from_dlpack(addresses, assumed_align=16),
        descriptors.data_ptr(),
        EXPERTS,
        current_stream_handle(device.index or 0),
    )
    torch.cuda.synchronize(device)

    seen = descriptors.view(EXPERTS, rd.BYTES_PER_DESCRIPTOR)[:, :8].cpu().view(torch.int64).flatten()
    assert seen.tolist() == addresses[0].cpu().tolist(), (
        f"descriptors point at {[hex(a) for a in seen.tolist()]}, " f"experts are at {[hex(a) for a in addresses[0].cpu().tolist()]}"
    )
    # And they are not all the atom's original tensor, which is what a no-op retarget would give.
    assert len(set(seen.tolist())) == EXPERTS


def _load(experts, addresses, retarget):
    device = experts[0].device
    descriptors = torch.zeros(rd.descriptor_bytes(len(experts)), dtype=torch.uint8, device=device)
    out = torch.zeros(len(experts), *TILE, dtype=torch.bfloat16, device=device)
    for expert in range(len(experts)):
        _run(
            from_dlpack(experts[0], assumed_align=16),
            from_dlpack(addresses, assumed_align=16),
            descriptors.data_ptr(),
            from_dlpack(out[expert], assumed_align=16),
            len(experts),
            expert,
            current_stream_handle(device.index or 0),
            retarget,
        )
    torch.cuda.synchronize(device)
    return out


def _distinct_experts():
    device = torch.device("cuda", torch.cuda.current_device())
    experts = [torch.full((ROWS, COLUMNS), float(index + 1), dtype=torch.bfloat16, device=device) for index in range(EXPERTS)]
    # (operands, experts): one operand here, the shape the builder indexes by row.
    return experts, ops.make_routed_weight_storage_table_bf16(experts).unsqueeze(0)


def test_without_retargeting_every_read_is_the_representative():
    """The control: the atom's own descriptor names expert 0, so every read should be expert 0.

    This is what the split-weight path would silently do if the retargeting were a no-op, and it
    also separates a problem in the retargeting from one in the surrounding TMA plumbing.
    """
    experts, addresses = _distinct_experts()
    out = _load(experts, addresses, retarget=False)
    for index in range(EXPERTS):
        assert torch.equal(out[index], experts[0][: TILE[0], : TILE[1]])


def test_each_descriptor_reads_its_own_expert():
    experts, addresses = _distinct_experts()
    assert len(set(addresses[0].tolist())) == EXPERTS, "the allocations must really be distinct"
    out = _load(experts, addresses, retarget=True)
    for index, expert in enumerate(experts):
        assert torch.equal(out[index], expert[: TILE[0], : TILE[1]]), (
            f"expert {index} read {out[index].flatten()[0].item()} where {index + 1} was written; " "its descriptor kept another expert's base address"
        )
