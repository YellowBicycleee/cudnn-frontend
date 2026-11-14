# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The peer count (EP size) is a runtime value taken from the workspace.

CPU-only contract checks, parameterized over ep_size in {1, 4, 8, 16, 64}:
  * every host plan/geometry accepts the size and threads it to the pointer-list
    and rank validation (no `8` literal anywhere on the host path);
  * the peer selectors in the shared BWD contract use the given ep_size;
  * the four kernel sources carry no EP8 constant: no `EP_SIZE`, no
    `peer_rank < Int32(8)` guard, no 8-way pointer select; peer bases come from
    an address table indexed by the runtime rank (CUDA `peer_buf[peer_rank]`);
  * the host builds the peer address table with the pointer-list length.
"""

from __future__ import annotations

import ast
import re
import unittest

from ._cutedsl_by_path import CUTEDSL_DIR, load_cutedsl_module

_CUTEDSL = CUTEDSL_DIR
EP_SIZES = (1, 4, 8, 16, 64)
# Device-only communication prefixes (the mega kernel hands them the peer table): the dynamic task stream the
# kernels call.  The shared pool module touches no peer memory.
COMM_MODULES = (
    "_dynamic_task_stream.py",
    "_fwd_dynamic_task_stream.py",
    "_bwd_dynamic_task_stream.py",
)
SHARED_POOLS = ("_dynamic_task_stream.py",)
KERNELS = (
    ("_persistent_bf16_fwd_mega.py",)
    + COMM_MODULES
    + (
        "_persistent_bf16_bwd_mega.py",
        "_persistent_mxfp8_fwd_mega.py",
        "_persistent_mxfp8_bwd_mega.py",
    )
)


def _load(name):
    return load_cutedsl_module(name)


class EpSizeRuntime(unittest.TestCase):
    def test_bwd_contract_geometry_and_peer_selectors_take_ep_size(self):
        c = _load("persistent_bf16_bwd_contract")
        for ep in EP_SIZES:
            g = c.BwdGeometry(ep_size=ep)
            self.assertEqual(g.ep_size, ep)
            # peer_rank selectors: -1 padding never selects; ranks in [0, ep) select; ep is out of range
            self.assertIsNone(c.selected_peer(-1, ep))
            self.assertEqual(c.selected_peer(ep - 1, ep), ep - 1)
            self.assertIsNone(c.selected_peer(ep, ep))
            self.assertFalse(c.reads_peer_token_index(-1, ep))
            self.assertTrue(c.reads_peer_token_index(0, ep))
        with self.assertRaises(ValueError):
            c.BwdGeometry(ep_size=0)
        with self.assertRaises(TypeError):
            c.selected_peer(0, 0)

    def test_kernel_sources_carry_no_ep8_constant(self):
        for name in KERNELS:
            text = (_CUTEDSL / name).read_text()
            with self.subTest(kernel=name):
                self.assertIsNone(re.search(r"(?<![A-Z_])EP_SIZE", text), name)  # _RT_EP_SIZE is the runtime slot
                self.assertNotIn("_select_peer_address", text)
                self.assertNotIn("_persistent_bf16_ptrs", text)
                self.assertNotIn("_peer_ptrs(", text)
                # No upper-bound peer guard against a literal: `peer... < Int32(8)`
                self.assertIsNone(re.search(r"peer\w*(\[\w+\])?\s*<\s*Int32\(8\)", text), name)
                self.assertIsNone(re.search(r"\]\s*<\s*Int32\(8\)", text), name)
                # Every kernel takes peer bases as tensors (address tables), never pointer lists.
                self.assertIsNone(re.search(r"peer_\w*:\s*list\[cute\.Pointer\]", text), name)
                if name not in SHARED_POOLS:
                    self.assertRegex(text, r"peer_table:\s*cute\.Tensor")
                else:
                    self.assertIsNone(re.search(r"\bpeer_\w+", text), name)
                # The host builds those tables from the pointer lists with runtime length
                # (device-only comm modules receive the table from the mega kernel; the MXFP8
                # forward's builder wraps its base module's, the MXFP8 backward's builds its own).
                if "_mxfp8_fwd" in name:
                    self.assertIn("make_mega_args(", text)
                    self.assertIn("peer_address_table(", (_CUTEDSL / "_mxfp8_fwd_base.py").read_text())
                elif name not in COMM_MODULES:
                    self.assertIn("peer_address_table(", text)

    def test_bwd_kernels_bound_peer_rank_by_runtime_ep_size(self):
        opt = (_CUTEDSL / "_persistent_bf16_bwd_mega.py").read_text()
        # The opt communication prefix treats peer ranks >= the runtime EP size as padding (load and
        # store), with the size handed down from the launch (runtime[_RT_EP_SIZE] = peer table length).
        comm = (_CUTEDSL / "_bwd_dynamic_task_stream.py").read_text()
        self.assertEqual(len(re.findall(r"if peer_rank >= args.ep_size:", comm)), 1)  # _routed
        self.assertEqual(len(re.findall(r"routed = _routed\(args, peer_rank\)", comm)), 2)  # load and store
        self.assertIn("ep_size=runtime[_RT_EP_SIZE]", opt)
        # compute side: the SwiGLU-backward router partial bounds the peer rank by the same runtime size,
        # inline or through a local `ep_size = runtime[_RT_EP_SIZE]` read in the same function (#33).
        guard = re.compile(r"peer_rank < (?:ep_size\b|runtime\[_RT_EP_SIZE\])")
        functions = [block for block in re.split(r"\n(?=    def |def )", opt) if guard.search(block)]
        self.assertTrue(functions, "no peer_rank guard against the runtime EP size")
        for block in functions:
            self.assertTrue("ep_size = runtime[_RT_EP_SIZE]" in block or "peer_rank < runtime[_RT_EP_SIZE]" in block, block.splitlines()[0])
        self.assertIn("ep_size = Int32(x_peer_table.shape[0])", opt)
        # Shared helpers: table lookup by rank, no branch chain.
        helpers = (_CUTEDSL / "_bwd_device_helpers.py").read_text()
        tree = ast.parse(helpers)
        fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "_peer_address")
        self.assertEqual(sum(isinstance(n, ast.If) for n in ast.walk(fn)), 0)
        self.assertIn("peer_table[peer_rank]", ast.unparse(fn))
        store = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "_store_peer_float")
        self.assertIn("ep_size = Int32(peer_table.shape[0])", ast.unparse(store))

    def test_host_peer_address_table_has_runtime_length(self):
        common = _load("_host_common")

        class Device:
            def __str__(self):
                return "test-device"

        built = []

        class FakeTorch:
            int64 = "int64"

            @staticmethod
            def tensor(values, dtype, device):
                built.append((tuple(values), dtype, str(device)))
                return ("table", tuple(values))

        import sys

        saved = sys.modules.get("torch")
        sys.modules["torch"] = FakeTorch
        try:
            self.assertFalse(hasattr(common, "_PEER_TABLE_CACHE"))  # no process-global CUDA tensor cache
            tables = {}  # owned by one workspace
            for ep in EP_SIZES:
                addresses = tuple(4096 * (i + 1) for i in range(ep))
                table = common.peer_address_table(addresses, Device(), tables)
                self.assertEqual(table, ("table", addresses))
                self.assertEqual(len(table[1]), ep)
                # same addresses, same workspace -> reused, no second construction
                self.assertIs(common.peer_address_table(list(addresses), Device(), tables), table)
            self.assertEqual(len(built), len(EP_SIZES))
            self.assertEqual(len(tables), len(EP_SIZES))
            # Another workspace's cache builds its own table.
            other = common.peer_address_table(addresses, Device(), {})
            self.assertEqual(other, table)
            self.assertEqual(len(built), len(EP_SIZES) + 1)
            with self.assertRaises(ValueError):
                common.peer_address_table((), Device(), {})
        finally:
            if saved is None:
                del sys.modules["torch"]
            else:
                sys.modules["torch"] = saved

    def test_host_validators_reference_plan_ep_size_not_a_literal(self):
        for name in ("persistent_bf16_fwd.py", "persistent_bf16_bwd.py", "persistent_mxfp8_fwd.py", "persistent_mxfp8_bwd.py"):
            text = (_CUTEDSL / name).read_text()
            with self.subTest(host=name):
                self.assertNotIn("EP_SIZE", text)
                self.assertIsNone(re.search(r"ep_size\s*!=\s*8|len\(pointers\)\s*!=\s*8|ep_rank\s*<\s*8", text), name)
                self.assertNotIn("eight aligned", text)
                self.assertRegex(text, r"len\(pointers\) != (plan|config|workspace)\.ep_size")
