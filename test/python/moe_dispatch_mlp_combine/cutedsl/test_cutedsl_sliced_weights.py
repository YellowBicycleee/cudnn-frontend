# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""A routed weight may be a slice of a larger parameter, and its strides must survive.

MCore owns the routed experts' Gate and Up as the two halves of one combined ``[E, 2I, H]``
parameter and hands MOK each half as a view.  The kernels build one TMA descriptor per weight and
never assume two are adjacent — the Gate/Up interleave exists in shared memory only — so a view is
addressable as it stands and copying the halves apart every step is pure waste.

Two host-side rules make that safe, and both are checked here:

``row_major_view`` must leave a strided tensor alone.  It rewrites the strides of a *contiguous*
tensor whose singleton or empty dimensions carry arbitrary ones, which the ABI descriptors need;
applied to a genuine slice it would describe a different region of memory and return wrong data
with no error at all.  That is the failure this file exists to prevent.

``validate_tensor(sliced=True)`` must accept the view and still reject what TMA cannot address:
a non-contiguous innermost dimension, an outer stride that is not 16-byte aligned, a self-
overlapping layout, and a base pointer off a 16-byte boundary.

CPU only, no CUTLASS and no GPU for the stride rules: ``python -m pytest moe_dispatch_mlp_combine/cutedsl/test_cutedsl_sliced_weights.py``.
"""

from __future__ import annotations

import pytest
import torch

from ._cutedsl_by_path import load_cutedsl_module

_host_common = load_cutedsl_module("_host_common")
row_major_view = _host_common.row_major_view
validate_tensor = _host_common.validate_tensor
TMA_ALIGNMENT_BYTES = _host_common.TMA_ALIGNMENT_BYTES

# A small stand-in for a combined routed FC1: [E, 2I, H], Gate the first I rows, Up the second.
EXPERTS, INTERMEDIATE, HIDDEN = 3, 8, 16


def _combined() -> torch.Tensor:
    count = EXPERTS * 2 * INTERMEDIATE * HIDDEN
    return torch.arange(count, dtype=torch.bfloat16).reshape(EXPERTS, 2 * INTERMEDIATE, HIDDEN)


def _halves(combined: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    return combined[:, :INTERMEDIATE, :], combined[:, INTERMEDIATE:, :]


def test_row_major_view_keeps_a_slice_addressing_its_own_rows():
    """The regression this file is named for: rewriting a slice's strides reads other rows."""
    combined = _combined()
    gate, up = _halves(combined)
    assert not gate.is_contiguous() and not up.is_contiguous()

    for name, half in (("gate", gate), ("up", up)):
        viewed = row_major_view(half)
        assert viewed.stride() == half.stride(), f"{name}: strides were rewritten"
        assert viewed.data_ptr() == half.data_ptr(), f"{name}: base moved"
        # The values are the test: canonical strides would read a contiguous run instead.
        assert torch.equal(viewed, half), f"{name}: reads different memory"

    # And the two halves really are different rows of the parameter.
    assert not torch.equal(gate, up)


def test_row_major_view_still_canonicalizes_singleton_strides():
    """What the function is for: a contiguous tensor whose singleton dimension carries any stride."""
    contiguous = torch.zeros(4, 1, 6)
    odd = contiguous.as_strided(contiguous.shape, (6, 999, 1))
    assert odd.is_contiguous()  # a singleton dimension's stride does not affect contiguity
    assert row_major_view(odd).stride() == (6, 6, 1)


def test_row_major_view_returns_a_plain_contiguous_tensor_unchanged():
    contiguous = torch.zeros(4, 5, 6)
    assert row_major_view(contiguous) is contiguous


@pytest.mark.skipif(not torch.cuda.is_available(), reason="validate_tensor requires a CUDA tensor")
class TestValidateTensor:
    """``validate_tensor`` needs CUDA tensors; the stride rules above do not."""

    @staticmethod
    def _cuda_combined() -> torch.Tensor:
        return _combined().cuda()

    def test_accepts_either_half_as_sliced(self):
        device = torch.device("cuda", torch.cuda.current_device())
        gate, up = _halves(self._cuda_combined())
        for name, half in (("gate", gate), ("up", up)):
            validate_tensor(name, half, shape=(EXPERTS, INTERMEDIATE, HIDDEN), dtype=torch.bfloat16, device=device, sliced=True)

    def test_rejects_a_slice_when_not_sliced(self):
        device = torch.device("cuda", torch.cuda.current_device())
        gate, _ = _halves(self._cuda_combined())
        with pytest.raises(ValueError, match="contiguous"):
            validate_tensor("gate", gate, shape=(EXPERTS, INTERMEDIATE, HIDDEN), dtype=torch.bfloat16, device=device)

    def test_rejects_a_non_contiguous_last_dimension(self):
        device = torch.device("cuda", torch.cuda.current_device())
        # Every other column: the innermost stride is 2, which TMA's box cannot express here.
        strided = self._cuda_combined()[:, :INTERMEDIATE, ::2]
        with pytest.raises(ValueError, match="last dimension"):
            validate_tensor("gate", strided, shape=(EXPERTS, INTERMEDIATE, HIDDEN // 2), dtype=torch.bfloat16, device=device, sliced=True)

    def test_rejects_an_outer_stride_that_is_not_aligned(self):
        device = torch.device("cuda", torch.cuda.current_device())
        elements = TMA_ALIGNMENT_BYTES // 2  # bfloat16
        assert HIDDEN % elements == 0, "pick a HIDDEN whose rows are aligned to start with"
        base = torch.zeros(EXPERTS * INTERMEDIATE * HIDDEN * 4, dtype=torch.bfloat16, device=device)
        # A row stride one element past the aligned one.
        misaligned = base.as_strided((EXPERTS, INTERMEDIATE, HIDDEN), ((HIDDEN + 1) * INTERMEDIATE, HIDDEN + 1, 1))
        with pytest.raises(ValueError, match="multiple of"):
            validate_tensor("gate", misaligned, shape=(EXPERTS, INTERMEDIATE, HIDDEN), dtype=torch.bfloat16, device=device, sliced=True)

    def test_rejects_a_self_overlapping_layout(self):
        device = torch.device("cuda", torch.cuda.current_device())
        base = torch.zeros(EXPERTS * INTERMEDIATE * HIDDEN, dtype=torch.bfloat16, device=device)
        # Every expert starts at the same place: the experts alias each other.
        overlapping = base.as_strided((EXPERTS, INTERMEDIATE, HIDDEN), (0, HIDDEN, 1))
        with pytest.raises(ValueError, match="overlaps itself"):
            validate_tensor("gate", overlapping, shape=(EXPERTS, INTERMEDIATE, HIDDEN), dtype=torch.bfloat16, device=device, sliced=True)

    def test_rejects_a_misaligned_base_pointer(self):
        device = torch.device("cuda", torch.cuda.current_device())
        elements = TMA_ALIGNMENT_BYTES // 2
        base = torch.zeros(EXPERTS * INTERMEDIATE * HIDDEN + elements, dtype=torch.bfloat16, device=device)
        offset = base[1:]  # one bfloat16 past the aligned base
        assert offset.data_ptr() % TMA_ALIGNMENT_BYTES
        with pytest.raises(ValueError, match="aligned"):
            validate_tensor(
                "gate",
                offset[: EXPERTS * INTERMEDIATE * HIDDEN].view(EXPERTS, INTERMEDIATE, HIDDEN),
                shape=(EXPERTS, INTERMEDIATE, HIDDEN),
                dtype=torch.bfloat16,
                device=device,
                sliced=True,
            )
