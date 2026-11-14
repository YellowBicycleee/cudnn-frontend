# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The host contract for MCore's split routed weights.

MCore holds the routed experts as independent per-expert allocations whenever the grouped
single-parameter layout is unavailable -- which is the common case, since the LayerWise
distributed optimizer refuses ``moe_single_grouped_weight`` and the DeepSeek recipes run Muon.
It hands MOK a ``SplitRoutedWeight``: one representative expert plus a table of base addresses.

What is checked here is the part that can be wrong silently. The table is only usable if every
expert agrees on dtype, shape, stride and device, because the kernel keeps one TMA descriptor and
only retargets its address; a table built from mismatched allocations would read the right address
with the wrong schema. The validators therefore have to reject that, and ``_routed_weight_payload``
has to hand the rest of the stack exactly the shape the non-split forms do, so that every
downstream check stays the one it already was.

GPU only (the tables hold device addresses):
``python -m pytest moe_dispatch_mlp_combine/cutedsl/test_split_routed_weights.py``
"""

from __future__ import annotations

import pytest
import torch

from cudnn.moe_dispatch_mlp_combine import functional as f
from cudnn.moe_dispatch_mlp_combine import ops

EXPERTS, INTERMEDIATE, HIDDEN = 4, 256, 512

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="storage tables hold device addresses")


def _experts(dtype=torch.bfloat16, shape=(INTERMEDIATE, HIDDEN), count=EXPERTS):
    return [torch.zeros(*shape, dtype=dtype, device="cuda") for _ in range(count)]


def _scales(count=EXPERTS):
    return [torch.zeros(INTERMEDIATE // 128, HIDDEN // 128, 32, 16, dtype=torch.uint8, device="cuda") for _ in range(count)]


def test_table_holds_each_expert_address():
    tensors = _experts()
    table = ops.make_routed_weight_storage_table_bf16(tensors)
    assert table.dtype is torch.int64 and table.device == tensors[0].device
    assert table.tolist() == [tensor.data_ptr() for tensor in tensors]
    # Distinct allocations: a table that collapsed them would read one expert everywhere.
    assert len(set(table.tolist())) == len(tensors)


def test_table_rejects_mismatched_experts():
    """The whole point of the table is that one descriptor schema fits every entry."""
    good = _experts()
    for label, bad in (
        ("dtype", torch.zeros(INTERMEDIATE, HIDDEN, dtype=torch.float16, device="cuda")),
        ("shape", torch.zeros(INTERMEDIATE, HIDDEN // 2, dtype=torch.bfloat16, device="cuda")),
    ):
        with pytest.raises((ValueError, TypeError)):
            ops.make_routed_weight_storage_table_bf16([*good[:-1], bad])
    with pytest.raises(ValueError, match="contiguous"):
        ops.make_routed_weight_storage_table_bf16([*good[:-1], torch.zeros(INTERMEDIATE, HIDDEN * 2, dtype=torch.bfloat16, device="cuda")[:, ::2]])
    with pytest.raises(TypeError):
        ops.make_routed_weight_storage_table_bf16([])


def test_scale_table_demands_the_tcgen05_shape():
    ops.make_routed_scale_storage_table(_scales())
    for bad in (torch.zeros(4, 4, 32, 8, dtype=torch.uint8, device="cuda"), torch.zeros(4, 4, 32, 16, dtype=torch.int8, device="cuda")):
        with pytest.raises((ValueError, TypeError)):
            ops.make_routed_scale_storage_table([bad])


def test_mxfp8_and_main_grad_tables_check_their_dtypes():
    ops.make_routed_weight_storage_table_mxfp8(_experts(dtype=torch.float8_e4m3fn))
    with pytest.raises(TypeError):
        ops.make_routed_weight_storage_table_mxfp8(_experts(dtype=torch.bfloat16))
    for dtype in (torch.float32, torch.bfloat16):
        ops.make_routed_d_weight_storage_table(_experts(dtype=dtype))
    with pytest.raises(TypeError):
        ops.make_routed_d_weight_storage_table(_experts(dtype=torch.float8_e4m3fn))


def _split(dtype=torch.bfloat16, **extra):
    tensors = _experts(dtype=dtype)
    table = (ops.make_routed_weight_storage_table_mxfp8 if dtype is torch.float8_e4m3fn else ops.make_routed_weight_storage_table_bf16)(tensors)
    return f.SplitRoutedWeight(data=tensors[0], storage_table=table, **extra)


def test_payload_is_the_shape_the_rest_of_the_stack_expects():
    """BF16 gives a tensor, MXFP8 a pair, and the backward's MXFP8 the five-tuple with its flag."""
    bf16 = _split()
    assert f._routed_weight_payload(bf16) is bf16.data

    scales = _scales()
    forward = _split(dtype=torch.float8_e4m3fn, scale=scales[0], scale_storage_table=ops.make_routed_scale_storage_table(scales), scale_tensors=tuple(scales))
    assert f._routed_weight_payload(forward) == (forward.data, forward.scale)

    transposed = _experts(dtype=torch.float8_e4m3fn)
    backward = _split(
        dtype=torch.float8_e4m3fn,
        scale=scales[0],
        scale_storage_table=ops.make_routed_scale_storage_table(scales),
        scale_tensors=tuple(scales),
        transposed_data=transposed[0],
        transposed_scale=scales[0],
        native_columnwise=True,
    )
    assert f._routed_weight_payload(backward) == (backward.data, backward.scale, backward.transposed_data, backward.transposed_scale, True)

    # Anything that is not a SplitRoutedWeight travels untouched.
    plain = torch.zeros(EXPERTS, INTERMEDIATE, HIDDEN, dtype=torch.bfloat16, device="cuda")
    assert f._routed_weight_payload(plain) is plain


def test_splitting_is_all_or_nothing():
    """One descriptor per operand means one addressing scheme; a mix has no meaning."""
    split = _split()
    plain = torch.zeros(EXPERTS, INTERMEDIATE, HIDDEN, dtype=torch.bfloat16, device="cuda")
    assert f._routed_storage_tables(plain, plain, plain) is None
    tables = f._routed_storage_tables(split, split, split)
    assert tables.gate is split.storage_table and tables.transposed is None
    with pytest.raises(ValueError, match="all be SplitRoutedWeight"):
        f._routed_storage_tables(split, plain, split)
