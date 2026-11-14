# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CuTe DSL MXFP8 quantization against the PyTorch reference, byte for byte.

``mok::mxfp8_quantize`` (``csrc/mxfp8.cuh``) is the reference.  MXFP8's end-to-end
tolerance is loose enough to hide a wrong scale rounding, so the port's quantization
is pinned here at the byte level instead: same input, identical payload and scale
bytes in both the normal and the transposed layout.

One GPU, no distributed setup: ``pytest moe_dispatch_mlp_combine/cutedsl/test_cutedsl_mxfp8_quantize.py``.
"""

import pytest
import torch
from cudnn.moe_dispatch_mlp_combine.cutedsl._arch import SUPPORTED_CAPABILITIES

requires_blackwell = pytest.mark.skipif(
    not (torch.cuda.is_available() and torch.cuda.get_device_capability() in SUPPORTED_CAPABILITIES),
    reason="the CuTe DSL MXFP8 quantizer is SM100/SM103 only",
)
OUTPUT_NAMES = ("x_fp8", "x_sc", "x_fp8_t", "x_sc_t")


def _reference(x, return_normal=True, return_transposed=True):
    """The PyTorch model of the quantization, not another call into the kernel under test."""
    from .utils import run_mxfp8_quantize_reference

    return run_mxfp8_quantize_reference(x, return_normal, return_transposed)


def _ported(x, return_normal=True, return_transposed=True):
    from cudnn.moe_dispatch_mlp_combine.cutedsl.mxfp8_quantize import mxfp8_quantize

    return mxfp8_quantize(x, return_normal, return_transposed)


def _assert_same_bytes(name, expected, actual):
    assert tuple(actual.shape) == tuple(expected.shape), f"{name}: shape"
    assert actual.dtype == expected.dtype, f"{name}: dtype"
    expected_bytes = expected.reshape(-1).view(torch.uint8)
    actual_bytes = actual.reshape(-1).view(torch.uint8)
    differing = expected_bytes != actual_bytes
    count = int(differing.sum().item())
    if count:
        first = int(differing.nonzero()[0].item())
        pytest.fail(
            f"{name}: {count}/{expected_bytes.numel()} bytes differ; first at "
            f"{first}: reference {int(expected_bytes[first])} vs kernel {int(actual_bytes[first])}"
        )


def _assert_same_quantization(x):
    for name, expected, actual in zip(OUTPUT_NAMES, _reference(x), _ported(x), strict=True):
        _assert_same_bytes(name, expected, actual)


@requires_blackwell
@pytest.mark.parametrize("shape", [(128, 128), (128, 512), (384, 256), (1024, 4096), (3, 256, 384)])
def test_random(shape):
    """Random normal activations over several geometries, 2-D and 3-D."""
    torch.manual_seed(0)
    _assert_same_quantization(torch.randn(shape, dtype=torch.bfloat16, device="cuda"))


@requires_blackwell
def test_block_boundaries():
    """Values that exercise the scale rounding, the floor and e4m3 saturation.

    One row per pattern, every value exactly representable in BF16 (the kernel's input
    type, so the pattern survives the conversion): zeros (amax 0 takes the 1e-12 floor),
    exact powers of two (the e8m0 exponent is exact), e4m3's 448 maximum and its BF16
    neighbours 446 and 450, magnitudes around the scale floor, a block whose amax sits one
    BF16 ulp below a power of two (127.5, where round-toward-+inf and round-to-nearest
    disagree), and two ramps that mix magnitudes within a block.
    """
    exact_rows = [
        torch.zeros(128),
        torch.full((128,), 448.0),
        torch.full((128,), 450.0),
        torch.full((128,), 446.0),
        2.0 ** torch.arange(-20, 108, dtype=torch.float32),
        torch.full((128,), 2.0**-40),
        torch.full((128,), 2.0**-14),
        torch.full((128,), 127.5),
    ]
    for row in exact_rows:
        assert torch.equal(row.to(torch.bfloat16).float(), row), "pattern is not BF16-exact"
    rows = exact_rows + [
        torch.linspace(-448.0, 448.0, 128),
        torch.linspace(-1e-6, 1e-6, 128),
    ]
    block = torch.stack(rows)
    # 128 rows: the patterns, their negatives and both column-reversed, repeated.
    variants = torch.cat([block, -block, block.flip(1), -block.flip(1)])
    x = variants.repeat(-(-128 // variants.shape[0]), 1)[:128]
    _assert_same_quantization(x.to(torch.bfloat16).cuda().contiguous())


@requires_blackwell
def test_single_layouts():
    """``return_normal`` / ``return_transposed`` select exactly one pair each."""
    torch.manual_seed(1)
    x = torch.randn((256, 256), dtype=torch.bfloat16, device="cuda")
    both = _ported(x)

    normal = _ported(x, True, False)
    assert normal[2] is None and normal[3] is None
    _assert_same_bytes("x_fp8", both[0], normal[0])
    _assert_same_bytes("x_sc", both[1], normal[1])

    transposed = _ported(x, False, True)
    assert transposed[0] is None and transposed[1] is None
    _assert_same_bytes("x_fp8_t", both[2], transposed[2])
    _assert_same_bytes("x_sc_t", both[3], transposed[3])
