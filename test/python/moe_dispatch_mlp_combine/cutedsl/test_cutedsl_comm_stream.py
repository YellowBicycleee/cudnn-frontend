# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The communication task streams of the forward and backward prefixes, checked on the CPU.

``_stream_task`` (and the helpers ``_reverse_tile`` / ``_step_task``) are integer arithmetic on runtime
values; here they run as plain Python (``Int32`` -> ``int``) over a grid of macrobatch / minibatch shapes,
including the short last macrobatch / generation, one macrobatch, ``mpm`` = 1 and odd tile counts, and
are checked against a readable reference enumeration of the same order.  Three properties per shape:
every task appears exactly once (bijection); the stream matches the reference; a task comes after
every task it depends on through the kernel's readiness counters, so the smallest unfinished task is
always ready (no deadlock).  A task is one tile row (all hidden columns); the edges, at the
granularity the counters have:
  forward   push(m, row in minibatch j) waits the routed Down of (m, j), which needs every pull of
            (m, j) [x_routed_ready] and, with a next macrobatch, every push of (m + 1, j)
            [y_routed_done]; pull(m - 1, row in j) waits the same Down
  backward  dX(g) waits the Gate/Up dgrad of g, which needs every X and dY row of g and every dX of
            g - mpm [reverse_dispatch_done]; X / dY(g) for g >= mpm wait the compute readers of
            g - mpm [routed_buffers_done], which need every pull of g - mpm and every dX of g - 2 mpm
            (recompute_forward_context: every generation has X rows, the first macrobatch's included)

The tile shape (rows x columns of one slot, the contract's ``CommTileShape``) is the compile-time parameter of
the communication CTAs; ``SlotShape`` runs the same checks at real geometries for both shapes and several hidden sizes,
and checks the counts that follow the shape: the arrivals of every readiness counter against what the compute
side waits for, and the bytes of the column blocks.
"""

from __future__ import annotations

from collections import Counter
import re
import textwrap
import types
import unittest

from ._cutedsl_by_path import CUTEDSL_DIR, load_cutedsl_module

PULL, PUSH = 0, 1
TASK_X, TASK_DY, TASK_DX = 0, 1, 2


def _device_functions(module: str, *names: str, **constants) -> dict:
    """The named ``@cute.jit`` functions of ``cudnn/moe_dispatch_mlp_combine/cutedsl/<module>.py`` as plain Python (Int32 -> int);
    ``constants`` are the module constants they read."""
    source = (CUTEDSL_DIR / f"{module}.py").read_text()
    code = []
    for name in names:
        heads = [m.start() for m in re.finditer(rf"\n@cute\.jit\ndef {name}\(", source)]
        assert len(heads) == 1, (module, name, len(heads))  # the one definition the kernel calls
        start = source.index(f"\ndef {name}(", heads[0]) + 1
        end = source.find("\n\n\n", start)
        code.append(source[start:end])
    text = re.sub(r"\) -> tuple\[[^\]]*\]:", "):", "\n\n".join(code)).replace(": Int32", "")
    text = text.replace(" -> Int32:", ":")
    namespace = dict(
        Int32=int,
        TASK_X=TASK_X,
        TASK_DY=TASK_DY,
        TASK_DX=TASK_DX,
        PULL=PULL,
        PUSH=PUSH,
        cutlass=types.SimpleNamespace(min=min, max=max, const_expr=bool, Constexpr=bool),
        **constants,
    )
    exec(text, namespace)
    return {name: namespace[name] for name in names}


def _slot_pool_method(name: str):
    """The ``@cute.jit`` method ``SlotPool.<name>`` of ``_dynamic_task_stream`` as plain Python; ``self`` is any
    object with a ``shape``."""
    source = (CUTEDSL_DIR / "_dynamic_task_stream.py").read_text()
    heads = [m.start() for m in re.finditer(rf"\n    @cute\.jit\n    def {name}\(", source)]
    assert len(heads) == 1, (name, len(heads))
    start = source.index(f"\n    def {name}(", heads[0]) + 1
    end = source.find("\n\n", start)
    text = textwrap.dedent(source[start:end])
    text = re.sub(r": (Int32|cute\.Tensor)", "", text)
    text = re.sub(r"\) -> [\w.]+:", "):", text)
    namespace = dict(Int32=int, WARP_LANES=32, cutlass=types.SimpleNamespace(min=min, const_expr=bool))
    exec(text, namespace)
    return namespace[name]


class ForwardStream(unittest.TestCase):
    """``_fwd_dynamic_task_stream._stream_task``: the forward order of the module docstring."""

    @staticmethod
    def reference(last_macro, last_tiles, full_tiles, mini_tiles):
        def minibatches_desc(count):  # tile rows of a macrobatch of ``count`` rows, last minibatch first
            n = (count + mini_tiles - 1) // mini_tiles
            for j in range(n - 1, -1, -1):
                yield j, list(range(j * mini_tiles, min(count, (j + 1) * mini_tiles)))

        out = [(PULL, last_macro, r) for _, rows in minibatches_desc(last_tiles) for r in rows]
        for m in range(last_macro, 0, -1):
            count = last_tiles if m == last_macro else full_tiles
            used = (count + mini_tiles - 1) // mini_tiles
            for j in range(full_tiles // mini_tiles - 1, used - 1, -1):  # ring positions m never used
                out += [(PULL, m - 1, r) for r in range(j * mini_tiles, (j + 1) * mini_tiles)]
            for j, push_rows in minibatches_desc(count):
                pull_rows = list(range(j * mini_tiles, (j + 1) * mini_tiles))
                k = 0
                for p in push_rows:
                    out.append((PUSH, m, p))
                    if k < len(pull_rows):
                        out.append((PULL, m - 1, pull_rows[k]))
                        k += 1
                out += [(PULL, m - 1, r) for r in pull_rows[k:]]
        out += [(PUSH, 0, r) for _, rows in minibatches_desc(full_tiles if last_macro else last_tiles) for r in rows]
        return out

    def check(self, f, last_macro, last_tiles, full_tiles, mt):
        """Bijection, the reference order and the dependency order of one shape (in tile rows)."""
        ref = self.reference(last_macro, last_tiles, full_tiles, mt)
        tasks = 2 * (last_tiles + last_macro * full_tiles)
        self.assertEqual(len(ref), tasks)
        got = [f(i, last_macro, last_tiles, full_tiles, mt) for i in range(tasks)]
        self.assertEqual(got, ref, (last_macro, last_tiles, full_tiles, mt))
        self.assertEqual(len(set(got)), tasks)  # bijection
        pos = {t: i for i, t in enumerate(got)}
        rows_of = lambda m: last_tiles if m == last_macro else full_tiles

        def minibatch_rows(m, r):  # tile rows of the minibatch of row r that macrobatch m has
            j = r // mt
            return range(j * mt, min((j + 1) * mt, rows_of(m)))

        for (d, m, r), i in pos.items():
            # the routed Down of (m, j) that a push of (m, j) / a pull of (m - 1, j) waits for
            m_down = m if d == PUSH else m + 1
            if m_down > last_macro:
                continue  # the last macrobatch's pulls wait for nothing
            for q in minibatch_rows(m_down, r):
                self.assertLess(pos[(PULL, m_down, q)], i, (d, m, r))  # x_routed_ready of (m_down, j)
            if m_down < last_macro:  # y_routed_done: the pushes of m_down + 1 on the same rows
                for q in minibatch_rows(m_down + 1, r):
                    self.assertLess(pos[(PUSH, m_down + 1, q)], i, (d, m, r))
        return got

    def test_matches_reference_and_dependencies(self):
        f = _device_functions("_fwd_dynamic_task_stream", "_reverse_tile", "_stream_task")["_stream_task"]
        cases = 0
        for mini_tiles in (1, 2, 3, 4):
            for full_mult in (1, 2, 3):
                full_tiles = mini_tiles * full_mult
                for last_macro in range(0, 4):
                    for last_tiles in range(1, full_tiles + 1):
                        mt = full_tiles if last_macro == 0 else mini_tiles  # one macrobatch: natural order
                        self.check(f, last_macro, last_tiles, full_tiles, mt)
                        cases += 1
        self.assertGreater(cases, 100)


class ForwardPullStream(unittest.TestCase):
    """``_fwd_dynamic_task_stream._pull_stream_task`` (epilogue_combine): exactly the pulls of ``_stream_task``, in the
    same order, so the pulls keep every dependency of the full stream (the pushes are the routed Down epilogue's)."""

    def test_pulls_of_the_full_stream(self):
        fns = _device_functions("_fwd_dynamic_task_stream", "_reverse_tile", "_stream_task", "_pull_stream_task")
        cases = 0
        for mini_tiles in (1, 2, 3, 4):
            for full_mult in (1, 2, 3):
                full_tiles = mini_tiles * full_mult
                for last_macro in range(0, 4):
                    for last_tiles in range(1, full_tiles + 1):
                        mt = full_tiles if last_macro == 0 else mini_tiles
                        full = [fns["_stream_task"](i, last_macro, last_tiles, full_tiles, mt) for i in range(2 * (last_tiles + last_macro * full_tiles))]
                        pulls = [t for t in full if t[0] == PULL]
                        got = [fns["_pull_stream_task"](i, last_macro, last_tiles, full_tiles, mt) for i in range(last_tiles + last_macro * full_tiles)]
                        self.assertEqual(got, pulls, (last_macro, last_tiles, full_tiles, mt))
                        cases += 1
        self.assertGreater(cases, 100)


class BackwardStream(unittest.TestCase):
    """``_bwd_dynamic_task_stream._stream_task``: the backward order of the module docstring."""

    @staticmethod
    def reference(G, mpm, mt, last, recompute_forward_context=False):
        rows = lambda g: last if g == G - 1 else mt
        # recompute_forward_context: X rows, then dY rows, per generation
        first = (TASK_X, TASK_DY) if recompute_forward_context else (TASK_DY,)
        out = [(kind, g, r) for g in range(min(mpm, G)) for kind in first for r in range(rows(g))]
        if G > mpm:
            lag = max(mpm - 1, 1)
            out += [(TASK_DX, g, r) for g in range(mpm - lag) for r in range(mt)]
            for g in range(mpm, G):
                n, p = rows(g), rows(g - lag)
                nxt = 0
                for kind in (TASK_X, TASK_DY):  # X half then dY half, half the pushes mixed into each
                    for r in range(n):
                        out.append((kind, g, r))
                        if r % 2 == 0 and nxt < p:
                            out.append((TASK_DX, g - lag, nxt))
                            nxt += 1
                out += [(TASK_DX, g - lag, r) for r in range(nxt, p)]
            tail = range(G - lag, G)
        else:
            tail = range(G)
        out += [(TASK_DX, g, r) for g in tail for r in range(rows(g))]
        return out

    def check(self, f, G, mpm, mt, last, recompute_forward_context):
        """Bijection, the reference order and the dependency order of one shape (in tile rows)."""
        ref = self.reference(G, mpm, mt, last, recompute_forward_context)
        rows = lambda g: last if g == G - 1 else mt
        first_x = 0 if recompute_forward_context else mpm  # first generation whose X is pulled
        expected = {(TASK_DY, g, r) for g in range(G) for r in range(rows(g))}
        expected |= {(TASK_DX, g, r) for g in range(G) for r in range(rows(g))}
        expected |= {(TASK_X, g, r) for g in range(first_x, G) for r in range(rows(g))}
        self.assertEqual(set(ref), expected)
        self.assertEqual(len(ref), len(expected))
        got = [f(i, G, mpm, mt, last, recompute_forward_context) for i in range(len(ref))]
        self.assertEqual(got, ref, (G, mpm, mt, last, recompute_forward_context))
        pos = {t: i for i, t in enumerate(got)}

        def pulls(g):
            kinds = (TASK_X, TASK_DY) if g >= first_x else (TASK_DY,)
            return [(k, g, q) for k in kinds for q in range(rows(g))]

        for (k, g, r), i in pos.items():
            if k == TASK_DX:  # Gate/Up dgrad of g: every pull of g, every dX of g - mpm
                before = pulls(g) + ([(TASK_DX, g - mpm, q) for q in range(rows(g - mpm))] if g >= mpm else [])
            elif g >= mpm:  # compute readers of g - mpm: every pull of g - mpm, every dX of g - 2 mpm
                before = pulls(g - mpm) + ([(TASK_DX, g - 2 * mpm, q) for q in range(rows(g - 2 * mpm))] if g >= 2 * mpm else [])
            else:
                before = []
            for t in before:
                self.assertLess(pos[t], i, ((k, g, r), t))
        return got

    def test_matches_reference_and_dependencies(self):
        f = _device_functions("_bwd_dynamic_task_stream", "_step_task", "_stream_task")["_stream_task"]
        cases = 0
        for mpm in (1, 2, 3, 4, 5):
            for mt in (1, 2, 3, 4):
                for G in range(1, 3 * mpm + 3):
                    for last, recompute_forward_context in ((last, flag) for last in range(1, mt + 1) for flag in (False, True)):
                        self.check(f, G, mpm, mt, last, recompute_forward_context)
                        cases += 1
        self.assertGreater(cases, 600)


class SlotShape(unittest.TestCase):
    """The communication tile shape (contract ``CommTileShape``: 32 rows x 512 columns or 16 x 1024, one 32 KiB
    slot) is the one compile-time parameter; every count derived from it follows it.  For both shapes, hidden sizes
    with and without a short last column block and several (routed rows on the rank, macrobatch, minibatch):
    the streams, sized as ``run_comm_prefix`` sizes them, move every row once in dependency order; a finished
    tile is one arrival, and the arrivals of every counter are what the compute side waits for; the column
    blocks of a row cover its H columns exactly."""

    HIDDEN = (640, 1152, 4096, 7168)
    # (routed rows on the rank, macrobatch, minibatch)
    GEOMETRIES = ((256, 256, 256), (1024, 1024, 256), (1280, 1024, 256), (2816, 1024, 512), (8192, 2048, 2048), (5376, 4096, 1024), (12288, 4096, 4096))

    @classmethod
    def setUpClass(cls):
        cls.shapes = load_cutedsl_module("persistent_bf16_fwd_contract").COMM_TILE_SHAPES
        cls.pairs = tuple((shape.rows, shape.columns) for shape in cls.shapes)
        cls.tile_bytes = staticmethod(_slot_pool_method("tile_bytes"))
        cls.lane_entry = staticmethod(_slot_pool_method("lane_entry"))

    def test_shapes_fill_one_slot(self):
        self.assertEqual(self.pairs, ((32, 512), (16, 1024)))
        for shape in self.shapes:
            self.assertEqual((shape.row_chunk_bytes, shape.slot_bytes), (shape.columns * 2, 32768))
            self.assertEqual(shape.rows * shape.row_chunk_bytes, shape.slot_bytes)  # the lanes' chunks fill the slot

    def test_column_blocks_cover_the_hidden_columns(self):
        for shape in self.shapes:
            pool, rows, columns = types.SimpleNamespace(shape=shape), shape.rows, shape.columns
            for hidden in self.HIDDEN:
                self.assertEqual(shape.column_blocks(hidden), -(-hidden // columns))
                blocks = [self.tile_bytes(pool, hidden, block) for block in range(shape.column_blocks(hidden))]
                self.assertEqual(sum(blocks), hidden * 2, (rows, columns, hidden))  # no padding column moves
                self.assertTrue(all(0 < b <= columns * 2 and b % 16 == 0 for b in blocks), blocks)
                self.assertTrue(all(b == columns * 2 for b in blocks[:-1]), blocks)  # only the last is short
                self.assertLessEqual(rows * max(blocks), 32768)  # a tile's transaction bytes fit its slot

    def test_lanes_past_the_tile_rows_read_no_schedule_entry(self):
        table = list(range(100, 132))
        for shape in self.shapes:
            pool, rows = types.SimpleNamespace(shape=shape), shape.rows
            entries = [self.lane_entry(pool, table[:rows], lane, lane) for lane in range(32)]  # IndexError past rows
            self.assertEqual(entries, table[:rows] + [-1] * (32 - rows))

    def test_forward_stream_and_arrivals(self):
        fns = _device_functions("_fwd_dynamic_task_stream", "_reverse_tile", "_stream_task", "_pull_stream_task", "_arrive_unit", DONE_BLOCK_ROWS=128)
        mega = (CUTEDSL_DIR / "_persistent_bf16_fwd_mega.py").read_text()
        # What the compute side waits for: x_routed_ready of a minibatch (Gate/Up) and y_routed_done of a 128-row
        # block (the next macrobatch's Down), one arrival per tile of the kernel's shape.
        self.assertRegex(mega, r"dispatch_required = _ceil_div_i32\(\s*minibatch_rows, " r"Int32\(comm_tile_shape\.rows\)\s*\) \* routed_comm_tile_columns")
        self.assertIn("_dynamic_task_stream.column_blocks(self.comm_tile_shape, routed_hidden_size)", mega)
        self.assertRegex(mega, r"Int32\(self\.comm_tile_shape\.row_tiles\(MLP_TILE_ROWS // CLUSTER_SIZE\)\)\s*" r"\* routed_comm_tile_columns")
        for (tile_rows, tile_columns), (tokens, macro, mini) in ((s, g) for s in self.pairs for g in self.GEOMETRIES):
            last_macro = -(-tokens // macro) - 1
            last_tiles = (tokens - last_macro * macro) // tile_rows
            full_tiles, mini_tiles = macro // tile_rows, mini // tile_rows
            if last_macro == 0:
                mini_tiles = full_tiles
            stream = ForwardStream().check(fns["_stream_task"], last_macro, last_tiles, full_tiles, mini_tiles)
            pulls = [fns["_pull_stream_task"](i, last_macro, last_tiles, full_tiles, mini_tiles) for i in range(last_tiles + last_macro * full_tiles)]
            self.assertEqual(pulls, [task for task in stream if task[0] == PULL])  # epilogue_combine
            # every routed row of the rank is pulled once and pushed once
            moved = sorted((d, m * macro + tile * tile_rows + lane) for d, m, tile in stream for lane in range(tile_rows))
            self.assertEqual(moved, [(d, row) for d in (PULL, PUSH) for row in range(tokens)])
            for hidden in self.HIDDEN:
                column_blocks = -(-hidden // tile_columns)
                arrivals = Counter()
                for d, m, tile in stream:
                    arrivals[d, fns["_arrive_unit"](d, m, tile * tile_rows, macro, mini)] += column_blocks
                expected = Counter()
                for minibatch in range(-(-tokens // mini)):  # dispatch_required
                    expected[PULL, minibatch] = -(-min(mini, tokens - minibatch * mini) // tile_rows) * column_blocks
                for block in range(macro // 128, -(-tokens // 128)):  # the blocks of macrobatches 1 .. last
                    expected[PUSH, block] = (128 // tile_rows) * column_blocks
                expected[PUSH, -1] = (min(macro, tokens) // tile_rows) * column_blocks  # macrobatch 0: nobody waits
                self.assertEqual(arrivals, expected, (tile_rows, tile_columns, tokens, macro, mini, hidden))

    def test_backward_stream_and_arrivals(self):
        fns = _device_functions("_bwd_dynamic_task_stream", "_step_task", "_stream_task", "_arrive_unit")
        contract = load_cutedsl_module("persistent_bf16_bwd_contract")
        mega = (CUTEDSL_DIR / "_persistent_bf16_bwd_mega.py").read_text()
        # What the compute side waits for (_comm_tiles): one arrival per tile of the kernel's shape.
        self.assertRegex(
            mega,
            r"return _dynamic_task_stream\.row_tiles\(self\.comm_tile_shape, rows\) \* "
            r"_dynamic_task_stream\.column_blocks\(\s*self\.comm_tile_shape, "
            r"runtime\[_RT_ROUTED_HIDDEN_SIZE\]\s*\)",
        )
        cases = ((s, g, flag) for s in self.shapes for g in self.GEOMETRIES for flag in (False, True))
        for shape, (tokens, macro, mini), recompute_forward_context in cases:
            tile_rows, tile_columns = shape.rows, shape.columns
            generations, mpm = -(-tokens // mini), macro // mini
            rows = lambda g: min(mini, tokens - g * mini)
            mini_tiles, last_tiles = mini // tile_rows, rows(generations - 1) // tile_rows
            stream = BackwardStream().check(fns["_stream_task"], generations, mpm, mini_tiles, last_tiles, recompute_forward_context)
            first_x = 0 if recompute_forward_context else mpm
            moved = sorted((k, g * mini + tile * tile_rows + lane) for k, g, tile in stream for lane in range(tile_rows))
            self.assertEqual(moved, [(k, row) for k in (TASK_X, TASK_DY, TASK_DX) for row in range(first_x * mini if k == TASK_X else 0, tokens)])
            for hidden in self.HIDDEN:
                column_blocks = -(-hidden // tile_columns)
                arrivals = Counter()
                for k, g, tile in stream:
                    arrivals[k, fns["_arrive_unit"](k, g, generations, mpm)] += column_blocks
                comm_tiles = lambda g: (rows(g) // tile_rows) * column_blocks  # _comm_tiles
                expected = Counter()
                for g in range(generations):
                    expected[TASK_DY, g] = comm_tiles(g)
                    if g >= first_x:
                        expected[TASK_X, g] = comm_tiles(g)
                    # reverse_dispatch_done: only while a later generation reuses the dX ring position
                    expected[TASK_DX, g if g + mpm < generations else -1] += comm_tiles(g)
                self.assertEqual(arrivals, expected, (tile_rows, tile_columns, tokens, macro, mini, hidden))
                geometry = contract.BwdGeometry(
                    schedule_capacity=tokens,
                    num_local_tokens=256,
                    macrobatch_size=macro,
                    minibatch_size=mini,
                    hidden_size=hidden,
                    topk=1,
                    comm_tile_shape=shape,
                )
                self.assertEqual(geometry.d_y_ready_required, (mini // tile_rows) * column_blocks)
                self.assertEqual(shape.tiles(mini, hidden), (mini // tile_rows) * column_blocks)
                self.assertEqual(geometry.reverse_dispatch_done_required, geometry.d_y_ready_required)
        with self.assertRaises(ValueError):
            contract.BwdGeometry(comm_tile_shape=(8, 2048))

    def test_stream_modules_carry_no_tile_constant(self):
        """The shape reaches the stream modules through ``SlotPool.shape`` only, and the megakernels through their
        ``comm_tile_shape``; no module keeps a tile constant of its own."""
        for name in ("_dynamic_task_stream.py", "_fwd_dynamic_task_stream.py", "_bwd_dynamic_task_stream.py"):
            text = (CUTEDSL_DIR / name).read_text()
            with self.subTest(module=name):
                self.assertIsNone(re.search(r"\bTILE_ROWS\b|\bTILE_COLUMNS\b|\bROW_CHUNK_BYTES\b", text))
                self.assertIsNone(re.search(r"Int(32|64)\((16|32|512|1024|2048)\)", text))
        for name in (
            "_persistent_bf16_fwd_mega.py",
            "_persistent_bf16_bwd_mega.py",
            "persistent_bf16_fwd_contract.py",
            "persistent_bf16_bwd_contract.py",
            "persistent_bf16_fwd.py",
            "persistent_bf16_bwd.py",
        ):
            text = (CUTEDSL_DIR / name).read_text()
            self.assertIsNone(re.search(r"dynamic_task_stream\.TILE_|COMM_TILE_(ROWS|COLUMNS)|DISPATCH_TILE", text), name)
        pool = (CUTEDSL_DIR / "_dynamic_task_stream.py").read_text()
        self.assertIn("SLOT_BYTES = COMM_SLOT_BYTES", pool)  # derived from the shape, not a second definition

    def test_host_selects_the_default_shape(self):
        select = load_cutedsl_module("_host_common").comm_tile_shape
        for hidden in self.HIDDEN:
            self.assertIs(select(hidden), self.shapes[0])
            for shape, pair in zip(self.shapes, self.pairs):
                self.assertEqual(select(hidden, shape), shape)
                self.assertEqual(select(hidden, pair), shape)
            with self.assertRaises(ValueError):
                select(hidden, (64, 256))


def _compute_decode():
    """``_decode_compute_task`` of the forward megakernel (a ``@cute.jit`` method) as plain Python, ``self`` dropped."""
    source = (CUTEDSL_DIR / "_persistent_bf16_fwd_mega.py").read_text()
    heads = [m.start() for m in re.finditer(r"\n    @cute\.jit\n    def _decode_compute_task\(", source)]
    assert len(heads) == 1, len(heads)
    start = source.index("\n    def _decode_compute_task(", heads[0]) + 1
    end = source.find("\n\n    @cute.jit", start)
    text = textwrap.dedent(source[start:end]).replace("    self,\n", "", 1)
    text = re.sub(r"\) -> tuple\[[^\]]*\]:", "):", text).replace(": Int32", "")
    namespace = dict(Int32=int, _ceil_div_i32=lambda a, b: -(-a // b))
    exec(text, namespace)
    return namespace["_decode_compute_task"]


class ForwardComputeOrder(unittest.TestCase):
    """The forward compute decode takes the routed minibatches in the stream's order (_fwd_dynamic_task_stream):
    macrobatches from the last to the first and, with several macrobatches, the minibatches of a macrobatch from the
    last to the first; one macrobatch keeps CUDA's natural order.  Every (macrobatch, minibatch, task) appears once."""

    def test_bijection_and_order(self):
        decode = _compute_decode()
        shared_gate_up, shared_down = 4, 2
        mini_gate_up, mini_down = 3, 2
        shared_tasks, minibatch_tasks = shared_gate_up + shared_down, mini_gate_up + mini_down
        cases = 0
        for minibatch_size in (256, 512):
            for macro_mult in (1, 2, 3, 4):
                macrobatch_size = minibatch_size * macro_mult
                for num_tokens in range(256, 4 * macrobatch_size + 1, 256):
                    num_macrobatches = -(-num_tokens // macrobatch_size)
                    true_minibatches = -(-num_tokens // minibatch_size)
                    last_macro_minibatches = true_minibatches - (num_macrobatches - 1) * macro_mult
                    expected = []
                    for m in range(num_macrobatches - 1, -1, -1):
                        count = last_macro_minibatches if m == num_macrobatches - 1 else macro_mult
                        order = range(count) if num_macrobatches == 1 else range(count - 1, -1, -1)
                        for j in order:
                            expected += [(2, t, m, j) for t in range(mini_gate_up)]
                            expected += [(3, t, m, j) for t in range(mini_down)]
                    shared = [(0, c, -1, -1) for c in range(shared_gate_up)] + [(1, c, -1, -1) for c in range(shared_down)]
                    got = [
                        decode(
                            c, num_tokens, shared_gate_up, shared_down, shared_tasks, mini_gate_up, mini_down, minibatch_tasks, macrobatch_size, minibatch_size
                        )
                        for c in range(shared_tasks + true_minibatches * minibatch_tasks)
                    ]
                    self.assertEqual(got, shared + expected, (num_tokens, macrobatch_size, minibatch_size))
                    self.assertEqual(len(set(got)), len(got))  # bijection
                    cases += 1
        self.assertGreater(cases, 40)


if __name__ == "__main__":
    unittest.main()
