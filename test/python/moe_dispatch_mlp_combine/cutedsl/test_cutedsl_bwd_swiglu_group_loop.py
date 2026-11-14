# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU: the routed Down dgrad epilogue runs its fused SwiGLU-backward in a runtime loop over feature groups and stores
each group's results inside that loop (source form; no torch, no CuTe DSL).

With the 32 features of a thread in one straight-line block the results of all of them are live at once, the register
peak of the backward megakernel; storing earlier within the block does not help, the assembler orders a basic block as
it likes.  The loop bounds the lifetimes to one group.  Register buffers indexed by the runtime loop variable would
fall into local memory, so only the global / staging tiles may depend on it.
"""

import ast
import unittest

from ._cutedsl_by_path import CUTEDSL_DIR

SOURCE = CUTEDSL_DIR / "_persistent_bf16_bwd_mega.py"
EPILOGUE = "_run_routed_down_swapab_epilogue"
RESULT_VECTORS = 3  # d_gate, d_up, w * Hidden


def _calls(node, name):
    return [n for n in ast.walk(node) if isinstance(n, ast.Call) and getattr(n.func, "attr", getattr(n.func, "id", "")) == name]


def _is_runtime_group_loop(node):
    return (
        isinstance(node, ast.For)
        and isinstance(node.target, ast.Name)
        and node.target.id == "group"
        and ast.unparse(node.iter.func) == "cutlass.range"
        and [(k.arg, ast.unparse(k.value)) for k in node.iter.keywords] == [("unroll", "1")]
    )


def _is_register_buffer(node):
    return isinstance(node, ast.Name) and node.id.startswith("r_")


class RoutedSwigluGroupLoop(unittest.TestCase):
    def test_each_group_is_computed_and_stored_in_one_iteration_of_a_runtime_loop(self):
        tree = ast.parse(SOURCE.read_text())
        functions = [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == EPILOGUE]
        self.assertEqual(len(functions), 1)
        loops = [n for n in ast.walk(functions[0]) if _is_runtime_group_loop(n)]
        self.assertEqual(len(loops), 1, "one cutlass.range(..., unroll=1) loop over the feature groups")
        loop = loops[0]
        # the element math of the whole function is in the loop
        self.assertEqual(len(_calls(functions[0], "_swiglu_backward_element")), 1)
        self.assertEqual(len(_calls(loop, "_swiglu_backward_element")), 1)
        # the group's results are stored by statements of the loop body itself, behind the element math
        stores = [
            n
            for n in loop.body
            if isinstance(n, ast.Expr) and _calls(n, "autovec_copy") and _is_register_buffer(n.value.args[0]) and not _is_register_buffer(n.value.args[1])
        ]
        self.assertEqual(len(stores), RESULT_VECTORS)
        math = [n for n in loop.body if _calls(n, "_swiglu_backward_element")]
        self.assertTrue(all(loop.body.index(s) > loop.body.index(math[0]) for s in stores))
        # no other copy of the function writes anything but a register buffer
        for call in _calls(functions[0], "autovec_copy"):
            if not any(call is s.value for s in stores):
                self.assertTrue(_is_register_buffer(call.args[1]), f"{ast.unparse(call)}: a store outside the loop")
        # register buffers are indexed with compile-time constants only, never with the runtime group index
        for node in ast.walk(loop):
            if isinstance(node, ast.Subscript) and _is_register_buffer(node.value):
                self.assertNotIn("group", {n.id for n in ast.walk(node.slice) if isinstance(n, ast.Name)}, ast.unparse(node))

    def test_the_pieces_run_through_one_runtime_loop_with_one_body(self):
        """One body in the binary: the staging tile is a runtime value of the piece loop, not a compile-time half."""
        tree = ast.parse(SOURCE.read_text())
        function = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == EPILOGUE)
        group_loop = next(n for n in ast.walk(function) if _is_runtime_group_loop(n))
        enclosing = [n for n in ast.walk(function) if isinstance(n, ast.For) and n is not group_loop and any(m is group_loop for m in ast.walk(n))]
        self.assertEqual([(n.target.id, ast.unparse(n.iter)) for n in enclosing], [("piece", "cutlass.range(pieces, unroll=1)")])
        self.assertIn("half = piece & Int32(1)", [ast.unparse(n) for n in enclosing[0].body])


if __name__ == "__main__":
    unittest.main()
