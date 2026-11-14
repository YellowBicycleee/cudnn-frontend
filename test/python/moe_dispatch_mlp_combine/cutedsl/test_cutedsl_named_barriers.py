# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""No aligned named barrier is reachable from a warp-role branch of the CuTe DSL kernels.

CPU-only, over the sources.  CuTe DSL 4.8 lowers ``NamedBarrier.arrive_and_wait`` / ``wait_unaligned`` /
``sync`` / ``wait`` / ``arrive`` and ``TmemAllocator.wait_for_alloc`` to the aligned PTX forms (``bar.sync``,
``bar.arrive``), which promise that every thread of the CTA executes that instruction.  The megakernels use
their named barriers inside warp-role branches (the epilogue warps, the MMA warp), so they go through
``_bwd_device_helpers.named_barrier_sync_unaligned`` and ``NamedBarrier.arrive_unaligned``.  Synccheck and the
numerical tests do not see the difference; the sources and the emitted PTX do.
"""

from __future__ import annotations

import ast
import re

from ._cutedsl_by_path import CUTEDSL_DIR

CUTEDSL = CUTEDSL_DIR
VENDORED = CUTEDSL / "_quack"
ALIGNED_METHODS = {"arrive_and_wait", "wait_unaligned", "wait_for_alloc"}
ALIGNED_WITHOUT_ARGUMENTS = {"arrive", "sync", "wait"}  # NamedBarrier's; the mbarrier forms take arguments
ALIGNED_PTX = re.compile(r"\bbar\.(sync|arrive)\b|\bbarrier\.(cta\.)?(sync|arrive)\.aligned\b")
# Reached by all 256 threads of the CTA before any warp-role branch (the MXFP8 megakernels' TMEM allocation).
WHOLE_CTA = {
    ("_persistent_mxfp8_fwd_mega.py", "wait_for_alloc"): 1,
    ("_persistent_mxfp8_bwd_mega.py", "wait_for_alloc"): 1,
}


def _aligned_barrier_calls(tree: ast.AST):
    """Calls of the aligned NamedBarrier / TmemAllocator methods and inline PTX with an aligned barrier
    (docstrings may name the instructions)."""
    docstrings = {
        id(node.body[0].value) for node in ast.walk(tree) if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef)) and ast.get_docstring(node)
    }
    for node in ast.walk(tree):
        if id(node) in docstrings:
            continue
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute):
            name = node.func.attr
            if (
                name in ALIGNED_METHODS
                or (name in ALIGNED_WITHOUT_ARGUMENTS and not node.args and not node.keywords)
                or (name == "barrier" and any(keyword.arg == "barrier_id" for keyword in node.keywords))
            ):
                yield node, name
        elif isinstance(node, ast.Constant) and isinstance(node.value, str) and ALIGNED_PTX.search(node.value):
            yield node, "inline PTX"


def _kernel_trees():
    return [(path.name, ast.parse(path.read_text())) for path in sorted(CUTEDSL.glob("*.py"))]


def test_kernels_issue_no_aligned_named_barrier_from_a_role_branch():
    found = {}
    for name, tree in _kernel_trees():
        for node, kind in _aligned_barrier_calls(tree):
            found.setdefault((name, kind), []).append(node.lineno)
    assert {key: len(lines) for key, lines in found.items()} == WHOLE_CTA, found


def test_the_shared_helper_builds_the_unaligned_sync():
    helpers = ast.parse((CUTEDSL / "_bwd_device_helpers.py").read_text())
    helper = next(node for node in helpers.body if isinstance(node, ast.FunctionDef) and node.name == "named_barrier_sync_unaligned")
    syncs = [node for node in ast.walk(helper) if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "barrier_cta_sync"]
    assert len(syncs) == 1
    aligned = {keyword.arg: keyword.value for keyword in syncs[0].keywords}.get("aligned")
    assert isinstance(aligned, ast.Constant) and aligned.value is False
    users = {
        name
        for name, tree in _kernel_trees()
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "named_barrier_sync_unaligned"
    }
    assert {"_persistent_bf16_fwd_mega.py", "_persistent_bf16_bwd_mega.py", "_persistent_mxfp8_fwd_gemm.py", "_mxfp8_gemm_base.py"} <= users, users


def test_no_vendored_function_with_an_aligned_barrier_is_called():
    """The vendored QuACK keeps its aligned barriers (its own kernel and epilogue); the kernels call none of
    those functions, directly or through another vendored function.  ``self.<name>`` inside a class that defines
    ``<name>`` is that class's own method (every kernel class has a ``kernel``)."""
    functions = {}
    for path in sorted(VENDORED.glob("*.py")):
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.FunctionDef):
                functions.setdefault(node.name, []).append(node)
    barrier_names = {name for name, nodes in functions.items() if any(True for node in nodes for _ in _aligned_barrier_calls(node))}
    assert {"epilogue", "kernel", "epi_begin"} <= barrier_names, barrier_names  # the scan sees the vendored barriers
    grew = True
    while grew:  # callers of those names inside the vendored package
        grew = False
        for name, nodes in functions.items():
            if name in barrier_names:
                continue
            callees = {
                call.func.attr if isinstance(call.func, ast.Attribute) else getattr(call.func, "id", None)
                for node in nodes
                for call in ast.walk(node)
                if isinstance(call, ast.Call)
            }
            if callees & barrier_names:
                barrier_names.add(name)
                grew = True

    offenders = []

    def visit(node: ast.AST, own_methods: frozenset, name: str) -> None:
        if isinstance(node, ast.ClassDef):
            own_methods = frozenset(item.name for item in node.body if isinstance(item, ast.FunctionDef))
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr in barrier_names:
            receiver = node.func.value
            own = isinstance(receiver, ast.Name) and receiver.id == "self" and node.func.attr in own_methods
            if not own:
                offenders.append(f"{name}:{node.lineno}: {ast.unparse(node.func)}")
        for child in ast.iter_child_nodes(node):
            visit(child, own_methods, name)

    for name, tree in _kernel_trees():
        visit(tree, frozenset(), name)
    assert not offenders, offenders
