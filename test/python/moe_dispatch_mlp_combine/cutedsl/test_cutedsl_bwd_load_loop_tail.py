# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU: the backward Wgrad K64 load loop peeks the ring once per iteration, as the first statement of the loop body and
never under an ``if`` (source form; no torch, no CuTe DSL).

A peek guarded by "is there a next K64 block" shares its predicate register with the loop-exit test: the guard's last
reader is the try_wait, so ptxas places the exit compare behind it with a 13-cycle stall on every K64 block.
"""

import ast
import unittest

from ._cutedsl_by_path import CUTEDSL_DIR

SOURCE = CUTEDSL_DIR / "_persistent_bf16_bwd_mega.py"


def _is_peek(node):
    return (
        isinstance(node, ast.Assign)
        and isinstance(node.value, ast.Call)
        and isinstance(node.value.func, ast.Attribute)
        and node.value.func.attr == "producer_try_acquire"
    )


class WgradLoadLoopPeek(unittest.TestCase):
    def test_the_only_peek_is_the_first_statement_of_the_loop_body(self):
        tree = ast.parse(SOURCE.read_text())
        functions = [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "_load_one_atb_task"]
        self.assertEqual(len(functions), 1)
        loops = [n for n in functions[0].body if isinstance(n, ast.For)]
        self.assertEqual(len(loops), 1, "one K64 load loop expected in _load_one_atb_task")
        peeks = [n for n in ast.walk(functions[0]) if _is_peek(n)]
        self.assertEqual(len(peeks), 1, "one producer_try_acquire per K64 block: nothing in the prologue or the tail")
        self.assertIs(peeks[0], loops[0].body[0], "the peek must be the first statement of the loop body, not under an if")


if __name__ == "__main__":
    unittest.main()
