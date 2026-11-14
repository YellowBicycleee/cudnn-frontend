# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Load ``cudnn/moe_dispatch_mlp_combine/cutedsl`` modules by path, without importing the
``cudnn.moe_dispatch_mlp_combine`` package (which imports torch).

The CPU-only tests check the contract and host helper modules (standard library at import time) on
machines without torch.  ``spec_from_file_location`` gives such a module no
parent package, so a relative import inside it (``persistent_bf16_bwd_contract`` imports the forward
contract) fails with "attempted relative import with no known parent package".  Here a stand-in
package whose ``__path__`` is the ``cutedsl`` directory is registered in ``sys.modules`` once, and
the modules are imported through it with the ordinary import machinery: relative imports between
them resolve, ``dataclasses`` finds the defining module, and a module is loaded once per process.
"""

from __future__ import annotations

import importlib
import importlib.util
from pathlib import Path
import sys
import types


def _cudnn_dir() -> Path:
    """The ``cudnn`` package under test: the one the interpreter imports, else this checkout's sources."""
    try:
        spec = importlib.util.find_spec("cudnn")
    except (ImportError, ValueError):
        spec = None
    if spec is not None and spec.origin:
        return Path(spec.origin).resolve().parent
    return Path(__file__).resolve().parents[4] / "python" / "cudnn"


PACKAGE_DIR = _cudnn_dir() / "moe_dispatch_mlp_combine"
CUTEDSL_DIR = PACKAGE_DIR / "cutedsl"
PACKAGE = "moe_dispatch_mlp_combine_cutedsl_by_path"


def load_cutedsl_module(name: str) -> types.ModuleType:
    """Return ``cudnn/moe_dispatch_mlp_combine/cutedsl/<name>.py`` imported as ``<PACKAGE>.<name>``."""
    if PACKAGE not in sys.modules:
        package = types.ModuleType(PACKAGE)
        package.__path__ = [str(CUTEDSL_DIR)]
        sys.modules[PACKAGE] = package
    return importlib.import_module(f"{PACKAGE}.{name}")
