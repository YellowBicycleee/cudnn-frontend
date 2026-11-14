# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The kernels use the QuACK modules vendored in ``cudnn/moe_dispatch_mlp_combine/cutedsl/_quack``, never an installed ``quack``, and the
vendored copy changes no CuTe DSL global when it is imported.

CPU, source only: no file of the package or of these tests imports ``quack``; every import inside ``_quack`` resolves to a file
of ``_quack``; no module-level statement of ``_quack`` assigns an attribute of a CUTLASS module (QuACK's own
package replaces CuTe DSL functions at import; the vendored copy must not).  CPU with CuTe DSL >= 4.8 installed
(no GPU needed): a fresh interpreter in which importing ``quack`` raises imports every module of
``cudnn.moe_dispatch_mlp_combine.cutedsl`` and checks that the CuTe DSL functions QuACK releases are known to replace are unchanged.
"""

from __future__ import annotations

import ast
import os
from pathlib import Path
import subprocess
import sys
import textwrap

import pytest

from ._cutedsl_by_path import CUTEDSL_DIR, PACKAGE_DIR

VENDORED = CUTEDSL_DIR / "_quack"


def _python_files():
    return [path for root in (PACKAGE_DIR, Path(__file__).resolve().parent) for path in root.rglob("*.py")]


def _imported_modules(tree: ast.AST):
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                yield node, alias.name
        elif isinstance(node, ast.ImportFrom):
            yield node, node.module or ""


def test_no_file_imports_quack():
    offenders = []
    for path in _python_files():
        for node, module in _imported_modules(ast.parse(path.read_text())):
            if getattr(node, "level", 0) == 0 and module.split(".")[0] == "quack":
                offenders.append(f"{path}:{node.lineno}: {module}")
    assert not offenders, offenders


def test_vendored_imports_stay_inside_the_vendored_package():
    for path in sorted(VENDORED.glob("*.py")):
        for node, module in _imported_modules(ast.parse(path.read_text())):
            if not isinstance(node, ast.ImportFrom) or node.level == 0:
                continue
            assert node.level == 1, f"{path.name}:{node.lineno} leaves _quack"
            names = [module] if module else [alias.name for alias in node.names]
            for name in names:
                assert (VENDORED / f"{name}.py").is_file(), f"{path.name}:{node.lineno}: _quack.{name} is not vendored"


def test_vendored_modules_assign_no_cutlass_attribute_at_import():
    for path in sorted(VENDORED.glob("*.py")):
        tree = ast.parse(path.read_text())
        cutlass_names = set()
        for node in tree.body:
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name.split(".")[0] == "cutlass":
                        cutlass_names.add(alias.asname or alias.name.split(".")[0])
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and (node.module or "").split(".")[0] == "cutlass":
                cutlass_names.update(alias.asname or alias.name for alias in node.names)
        for node in tree.body:
            targets = node.targets if isinstance(node, ast.Assign) else [node.target] if isinstance(node, (ast.AugAssign, ast.AnnAssign)) else []
            calls = [node.value] if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call) else []
            for target in targets:
                root = target
                while isinstance(root, (ast.Attribute, ast.Subscript)):
                    root = root.value
                assert not (
                    target is not root and isinstance(root, ast.Name) and root.id in cutlass_names
                ), f"{path.name}:{node.lineno} assigns {ast.unparse(target)} at import"
            for call in calls:
                if ast.unparse(call.func) == "setattr" and call.args:
                    root = call.args[0]
                    while isinstance(root, ast.Attribute):
                        root = root.value
                    assert not (isinstance(root, ast.Name) and root.id in cutlass_names), f"{path.name}:{node.lineno} calls {ast.unparse(call)} at import"


def test_kernel_modules_import_without_quack():
    code = textwrap.dedent("""
        import importlib, importlib.abc, pkgutil, sys

        class NoQuack(importlib.abc.MetaPathFinder):
            def find_spec(self, name, path=None, target=None):
                if name == "quack" or name.startswith("quack."):
                    raise ImportError(f"the operator imported {name}")
                return None

        sys.meta_path.insert(0, NoQuack())
        try:
            import cutlass  # noqa: F401
        except ImportError as error:
            print("SKIP CuTe DSL is not installed:", error)
            sys.exit(0)
        import cudnn.moe_dispatch_mlp_combine.cutedsl as package
        from cudnn.moe_dispatch_mlp_combine.cutedsl._version_gate import CuteDslRequirementError, require_cutedsl
        try:
            require_cutedsl()
        except CuteDslRequirementError as error:
            print("SKIP", error)
            sys.exit(0)

        def watched():
            from cutlass.cute import _tvm_ffi_args_spec_converter as converter
            functions = {"_convert_single_arg": converter._convert_single_arg}
            try:
                from cutlass.base_dsl.ast_preprocessor import DSLPreprocessor
            except ImportError:
                pass
            else:
                functions["DSLPreprocessor.visit_If"] = DSLPreprocessor.visit_If
            return functions

        before = watched()
        names = sorted(info.name for info in pkgutil.walk_packages(package.__path__, package.__name__ + "."))
        for name in names:
            importlib.import_module(name)
        assert "cudnn.moe_dispatch_mlp_combine.cutedsl._quack.gemm_default_epi" in sys.modules
        assert not [name for name in sys.modules if name == "quack" or name.startswith("quack.")]
        changed = [key for key, function in watched().items() if before[key] is not function]
        assert not changed, f"importing the kernel modules replaced {changed}"
        print("OK", len(names))
    """)
    result = subprocess.run([sys.executable, "-c", code], env=dict(os.environ), capture_output=True, text=True, timeout=900)
    assert result.returncode == 0, result.stdout + result.stderr
    if result.stdout.startswith("SKIP"):
        pytest.skip(result.stdout[5:].strip())
    assert result.stdout.startswith("OK"), result.stdout
