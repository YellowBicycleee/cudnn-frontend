# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""A warp consumes the decoded-task record before it releases the record's stage.

CPU-only, over the sources of the BF16 megakernels.  Every consumer warp loads the record, releases the stage and
uses the values afterwards, while the CLC warp refills the stage.  A shared-memory load that nothing has read when
the release executes can return the next record, and the compiler moves plain uses behind the release, so
``_load_task_info`` stores the sum of every loaded word (one word per warp, never read) before it returns: the
release that follows the call is behind that store.  The disassembly is the proof; this pins the source form.
"""

from __future__ import annotations

import ast

import pytest

from ._cutedsl_by_path import CUTEDSL_DIR

CUTEDSL = CUTEDSL_DIR
LOAD = "_load_task_info(s_task_info, task_consumer_state.index, s_task_taken, warp)"


@pytest.mark.parametrize("name,releases", [("_persistent_bf16_fwd_mega.py", 4), ("_persistent_bf16_bwd_mega.py", 1)])
def test_task_record_release_follows_the_consuming_load(name, releases):
    tree = ast.parse((CUTEDSL / name).read_text())
    load = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "_load_task_info")
    stores = [
        node
        for node in ast.walk(load)
        if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Subscript) and ast.unparse(node.targets[0].value) == "s_task_taken"
    ]
    assert len(stores) == 1 and ast.unparse(stores[0].value) == "sum(words[1:], words[0])"
    assert isinstance(load.body[-1], ast.Return) and stores[0].lineno < load.body[-1].lineno

    found = 0
    for node in ast.walk(tree):
        for body in (getattr(node, "body", None), getattr(node, "orelse", None)):
            for position, statement in enumerate(body if isinstance(body, list) else []):
                if isinstance(statement, ast.With) and "task_pipeline.consumer_release(" in ast.unparse(statement):
                    found += 1
                    before = "\n".join(ast.unparse(earlier) for earlier in body[:position])
                    assert LOAD in before, f"{name}:{statement.lineno}: released without the consuming load in front"
    assert found == releases
