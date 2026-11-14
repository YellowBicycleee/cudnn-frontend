# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Every ``TmemAllocator`` the CuTe DSL kernels construct names its options by keyword.

CPU-only, over the sources.  nvidia-cutlass-dsl releases differ in the allocator's signature: the older one
takes the retrieval barrier and the options positionally or by keyword, the newer one makes everything after
the shared-memory pointer keyword-only.  One spelling serves both -- the pointer positional, the rest by
keyword -- and a positional barrier fails only when the kernel is traced under the newer release.
"""

from __future__ import annotations

import ast
import unittest

from ._cutedsl_by_path import CUTEDSL_DIR

CUTEDSL = CUTEDSL_DIR


def _allocator_calls():
    for path in sorted(CUTEDSL.glob("*.py")):
        for node in ast.walk(ast.parse(path.read_text())):
            if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)):
                continue
            if node.func.attr == "TmemAllocator":
                yield path.name, node


class TmemAllocatorCalls(unittest.TestCase):
    def test_only_the_pointer_is_positional(self):
        calls = list(_allocator_calls())
        self.assertGreaterEqual(len(calls), 4)  # the four megakernels
        for name, call in calls:
            where = f"{name}:{call.lineno}"
            self.assertLessEqual(len(call.args), 1, where)
            self.assertIn("barrier_for_retrieve", [keyword.arg for keyword in call.keywords], where)


if __name__ == "__main__":
    unittest.main()
