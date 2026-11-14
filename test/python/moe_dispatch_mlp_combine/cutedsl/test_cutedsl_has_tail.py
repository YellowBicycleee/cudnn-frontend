# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""``_host_common.has_tail``: the host's derivation of the megakernels' compile-time ``has_tail`` flag.

CPU-only.  The tail instantiations of the kernels are generated for a tail of exactly ``TAIL_WIDTH`` = 128
columns, half a 256-wide tile, so the helper that derives the flag accepts only extents that are whole
128-wide halves: a multiple of 256 (no tail) or 128 more than one (a tail).
"""

from __future__ import annotations

import unittest

from ._cutedsl_by_path import load_cutedsl_module

host_common = load_cutedsl_module("_host_common")


class HasTail(unittest.TestCase):
    def test_the_tail_is_half_a_tile(self):
        self.assertEqual(host_common.TAIL_WIDTH, 128)

    def test_aligned_sizes_have_no_tail(self):
        for sizes in ((256,), (512, 512), (2048, 2048, 1024), (7168, 5120)):
            self.assertIs(host_common.has_tail(*sizes), False, sizes)

    def test_a_128_wide_tail_in_any_size(self):
        for sizes in ((384,), (640, 512), (512, 640), (512, 512, 384), (2688, 2048, 1280), (640, 640, 640)):
            self.assertIs(host_common.has_tail(*sizes), True, sizes)

    def test_sizes_off_the_128_grid_are_rejected(self):
        for size in (192, 320, 384 + 64, 100, 257):
            for sizes in ((size,), (512, size), (size, 640), (512, 640, size)):
                with self.assertRaises(ValueError, msg=sizes):
                    host_common.has_tail(*sizes)


if __name__ == "__main__":
    unittest.main()
