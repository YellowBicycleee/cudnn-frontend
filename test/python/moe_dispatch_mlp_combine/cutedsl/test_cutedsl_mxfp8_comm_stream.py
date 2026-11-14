# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The MXFP8 megas' communication task stream decode, checked on the CPU.

``_stream_task`` (the forward stream's task -> (queue, macrobatch, tile) decode), ``_reciprocal_u32`` and
``_divide_u32`` (the division-free block decode) of ``cudnn/moe_dispatch_mlp_combine/cutedsl/_persistent_mxfp8_fwd_mega.py``, and
``_backward_stream_task`` (the backward stream's per-macrobatch task -> (queue, tile) decode) of
``_persistent_mxfp8_bwd_mega.py``, are integer arithmetic on runtime values; here they run as plain Python
(``Int32`` / ``Int64`` -> ``int``, ``mul.hi.u32`` as a Python product) over a grid of macrobatch shapes, including
one macrobatch, a short last macrobatch and tile counts that are not powers of two, and are checked against a
readable reference enumeration of the order the docstrings describe.  Three properties per shape: every task
appears exactly once and their number is the block total the kernel computes (bijection, totals match); the stream
equals the reference; every task comes after the tasks its readiness counter depends on (forward: a push of
macrobatch m and a pull of m - 1 wait for the routed Down of m, which needs every pull of m and, with a macrobatch
after m, every push of m + 1; backward: a push waits for its macrobatch's Gate / Up dgrad, which needs every pull
of the macrobatch), so the smallest unfinished task is always ready.  The division is checked against ``//`` for
divisors from 1 (where the reciprocal wraps to 0xFFFFFFFF) up, over the dividends the streams produce.

The backward's compute-task decode (``_task_table`` / ``_decode_mxfp8_task``, with the cluster count of
``_true_clusters``) runs the same way for both values of the kernel's compile-time ``recompute_forward_context`` flag
and is checked against the task order of its docstring: with ``recompute_forward_context`` macrobatch 0 has the replay
tasks of a replayed macrobatch (and its stream the x pulls: ``_backward_stream_task`` with twice the tiles as pull
tasks).

CPU only: ``python -m pytest moe_dispatch_mlp_combine/cutedsl/test_cutedsl_mxfp8_comm_stream.py`` (no torch, no GPU).
"""

from __future__ import annotations

import re
import types
import unittest

from ._cutedsl_by_path import CUTEDSL_DIR

STREAM_PULL, STREAM_PUSH, STREAM_PULL_X = 0, 1, 2
MODULE = "_persistent_mxfp8_fwd_mega"
BWD_MODULE = "_persistent_mxfp8_bwd_mega"


def _mul_hi_u32(a: int, b: int) -> int:
    """``mul.hi.u32``: the high 32 bits of the 64-bit product of the operands' low 32 bits."""
    return ((a & 0xFFFFFFFF) * (b & 0xFFFFFFFF)) >> 32


def _device_functions(*names: str, module: str = MODULE) -> dict:
    """The named ``@cute.jit`` functions of a mega module as plain Python (Int32 / Int64 -> int)."""
    source = (CUTEDSL_DIR / f"{module}.py").read_text()
    code = []
    for name in names:
        heads = [m.start() for m in re.finditer(rf"\n@cute\.jit\ndef {name}\(", source)]
        assert len(heads) == 1, (name, len(heads))
        start = source.index(f"\ndef {name}(", heads[0]) + 1
        end = source.find("\n\n\n", start)
        code.append(source[start:end])
    text = re.sub(r"\) -> tuple\[[^\]]*\]:", "):", "\n\n".join(code))
    text = text.replace(": Int32", "").replace(" -> Int32:", ":")
    namespace = dict(
        Int32=int,
        Int64=int,
        STREAM_PULL=STREAM_PULL,
        STREAM_PUSH=STREAM_PUSH,
        STREAM_PULL_X=STREAM_PULL_X,
        _mul_hi_u32=_mul_hi_u32,
        cutlass=types.SimpleNamespace(min=min, max=max),
    )
    exec(text, namespace)
    return {name: namespace[name] for name in names}


class ReciprocalDivision(unittest.TestCase):
    """``_divide_u32(n, d, _reciprocal_u32(d))`` is ``n // d`` for 0 <= n < 2^31 and every divisor the streams use."""

    def test_matches_floor_division(self):
        f = _device_functions("_reciprocal_u32", "_divide_u32")
        divisors = [1, 2, 3, 4, 5, 6, 7, 8, 9, 12, 16, 17, 31, 32, 33, 64, 100, 255, 256, 257, 1023, 4096, 4097, 65536, (1 << 20) + 3, (1 << 31) - 1]
        dividends = list(range(0, 4200)) + [(1 << 31) - 1 - k for k in range(64)] + [(1 << 20) * k + 7 for k in range(1, 2048)]
        self.assertEqual(f["_reciprocal_u32"](1), 0xFFFFFFFF)  # the Int32 truncation keeps the bits mul.hi reads
        checks = 0
        for d in divisors:
            r = f["_reciprocal_u32"](d)
            for n in dividends:
                self.assertEqual(f["_divide_u32"](n, d, r), n // d, (n, d))
                checks += 1
        self.assertGreater(checks, 100000)


class ForwardStream(unittest.TestCase):
    """``_stream_task``: the forward order of its docstring."""

    @staticmethod
    def reference(last_macro, last_tiles, full_tiles):
        tiles = lambda m: last_tiles if m == last_macro else full_tiles  # noqa: E731
        out = [(STREAM_PULL, last_macro, t) for t in range(last_tiles)]
        for m in range(last_macro, 0, -1):  # segment m: the pushes of m interleaved with the pulls of m - 1
            for t in range(tiles(m)):
                out.append((STREAM_PUSH, m, t))
                out.append((STREAM_PULL, m - 1, t))
            out += [(STREAM_PULL, m - 1, t) for t in range(tiles(m), full_tiles)]  # only the last macrobatch is short
        out += [(STREAM_PUSH, 0, t) for t in range(tiles(0))]
        return out

    def test_matches_reference_totals_and_dependencies(self):
        f = _device_functions("_reciprocal_u32", "_divide_u32", "_stream_task")
        cases = 0
        for full_tiles in (1, 2, 3, 4, 5, 8, 33):
            reciprocal_2f = f["_reciprocal_u32"](2 * full_tiles)
            for last_macro in range(0, 4):
                for last_tiles in range(1, full_tiles + 1):
                    ref = self.reference(last_macro, last_tiles, full_tiles)
                    blocks = 2 * (last_tiles + last_macro * full_tiles)  # the kernel's block total (per column group)
                    self.assertEqual(len(ref), blocks)
                    got = [tuple(f["_stream_task"](i, last_macro, last_tiles, full_tiles, reciprocal_2f)) for i in range(blocks)]
                    self.assertEqual(got, ref, (last_macro, last_tiles, full_tiles))
                    self.assertEqual(len(set(got)), blocks)  # bijection
                    pos = {task: i for i, task in enumerate(got)}
                    tiles = lambda m: last_tiles if m == last_macro else full_tiles  # noqa: E731
                    for (queue, m, t), i in pos.items():
                        m_down = m if queue == STREAM_PUSH else m + 1  # the routed Down the task waits for
                        if m_down > last_macro:
                            continue  # the last macrobatch's pulls wait for nothing
                        for q in range(tiles(m_down)):  # x_routed_ready of m_down: every pull of m_down
                            self.assertLess(pos[(STREAM_PULL, m_down, q)], i, (queue, m, t))
                        if m_down < last_macro:  # y_routed_done: every push of m_down + 1 on the same rows
                            for q in range(tiles(m_down + 1)):
                                self.assertLess(pos[(STREAM_PUSH, m_down + 1, q)], i, (queue, m, t))
                    cases += 1
        self.assertGreater(cases, 100)


class BackwardStream(unittest.TestCase):
    """``_backward_stream_task``: one macrobatch's order of its docstring (macrobatch 0: dY pulls then pushes; a
    replayed macrobatch, which macrobatch 0 is with ``recompute_forward_context``: x and dY pulls tile by tile, then
    pushes)."""

    @staticmethod
    def reference(tiles, replay):
        out = []
        for t in range(tiles):
            if replay:
                out.append((STREAM_PULL_X, t))
            out.append((STREAM_PULL, t))
        out += [(STREAM_PUSH, t) for t in range(tiles)]
        return out

    def test_matches_reference_totals_and_dependencies(self):
        f = _device_functions("_backward_stream_task", module=BWD_MODULE)
        cases = 0
        for tiles in (1, 2, 3, 4, 5, 8, 33, 128):
            for replay in (False, True):
                pull_tasks = 2 * tiles if replay else tiles
                ref = self.reference(tiles, replay)
                blocks = pull_tasks + tiles  # the kernel's block total (per column group)
                self.assertEqual(len(ref), blocks)
                got = [tuple(f["_backward_stream_task"](i, tiles, pull_tasks)) for i in range(blocks)]
                self.assertEqual(got, ref, (tiles, replay))
                self.assertEqual(len(set(got)), blocks)  # bijection
                pos = {task: i for i, task in enumerate(got)}
                for t in range(tiles):  # a push's Gate / Up dgrad needs the macrobatch's dY (and x) pulls
                    for q in range(tiles):
                        self.assertLess(pos[(STREAM_PULL, q)], pos[(STREAM_PUSH, t)])
                        if replay:
                            self.assertLess(pos[(STREAM_PULL_X, q)], pos[(STREAM_PUSH, t)])
                cases += 1
        self.assertGreater(cases, 10)


def _backward_task_model(recompute_forward_context: bool):
    """The backward kernel's compute-task arithmetic as plain Python: an object with the kernel class's
    ``_minibatches_of``, ``_true_clusters``, ``_task_table`` and ``_decode_mxfp8_task`` (source text, ``Int32`` ->
    ``int``, ``cutlass.const_expr`` the identity) and its compile-time flags, plus the module's ``_task_counts``
    and the task kinds by name."""
    source = (CUTEDSL_DIR / f"{BWD_MODULE}.py").read_text()
    base = (CUTEDSL_DIR / "_mxfp8_bwd_base.py").read_text()
    namespace = dict(
        Int32=int,
        cute=types.SimpleNamespace(jit=lambda function: function),
        cutlass=types.SimpleNamespace(min=min, max=max, const_expr=lambda value: value),
        _ceil_div_i32=lambda value, divisor: -(-value // divisor),
    )
    for text, names in (
        (source, ("MLP_TILE_M", "MLP_TILE_N", "SWIGLU_TILE", "SWIGLU_FWD_STAGES", "SWIGLU_BWD_STAGES", "CLUSTER_SIZE", "REPLAY_SWIGLU")),
        (
            base,
            (
                "SHARED_DOWN_DGRAD",
                "SHARED_SWIGLU_BWD",
                "SHARED_GATE_UP_DGRAD",
                "SHARED_WGRAD_DOWN",
                "REPLAY_GATE",
                "REPLAY_UP",
                "ROUTED_DOWN_DGRAD",
                "ROUTED_SWIGLU_BWD",
                "ROUTED_GATE_UP_DGRAD",
                "ROUTED_WGRAD_DOWN",
            ),
        ),
    ):
        for name in names:
            namespace[name] = int(re.search(rf"^{name} = (\d+)\b", text, re.MULTILINE).group(1))
    code = []
    for name in ("_tile_blocks", "_task_counts"):  # module functions (host arithmetic, not jit)
        start = source.index(f"\ndef {name}(") + 1
        code.append(source[start : source.index("\n\n\n", start)])
    code.append("class Model:")
    for name in ("_minibatches_of", "_true_clusters", "_task_table", "_decode_mxfp8_task"):
        heads = [m.start() for m in re.finditer(rf"\n    @cute\.jit\n    def {name}\(", source)]
        assert len(heads) == 1, (name, len(heads))
        code.append(source[heads[0] + 1 : source.index("\n\n    @cute.jit\n", heads[0] + 1)])
    exec("\n\n".join(code), namespace)
    model = namespace["Model"]()
    model.recompute_forward_context, model.has_shared, model.has_tail = recompute_forward_context, True, False
    return model, namespace["_task_counts"], types.SimpleNamespace(**namespace)


class BackwardTaskDecode(unittest.TestCase):
    """``_decode_mxfp8_task`` over ``_task_table``: the compute tasks in the order of the decode's docstring, as
    many as ``_true_clusters`` counts.  ``recompute_forward_context``: macrobatch 0's minibatches have the replay tasks
    too."""

    @staticmethod
    def reference(kinds, counts, num_tokens, macrobatch, minibatch, recompute_forward_context):
        (
            shared_down,
            shared_swiglu,
            shared_gate_up,
            _,
            routed_down,
            routed_swiglu,
            routed_gate_up,
            _,
            replay_gate_up,
            replay,
            weight_tiles,
            wgrad_matrix,
            _,
        ) = counts
        out = [(kinds.SHARED_DOWN_DGRAD, 0, 0, local) for local in range(shared_down)]
        out += [(kinds.SHARED_SWIGLU_BWD, 0, 0, local) for local in range(shared_swiglu)]
        out += [(kinds.SHARED_GATE_UP_DGRAD, 0, 0, local) for local in range(shared_gate_up)]
        out += [(kinds.SHARED_WGRAD_DOWN + matrix, 0, 0, local) for matrix in range(3) for local in range(weight_tiles)]
        for macro in range(-(-num_tokens // macrobatch)):
            rows = min(macrobatch, num_tokens - macro * macrobatch)
            generations = [macro * (macrobatch // minibatch) + mini for mini in range(-(-rows // minibatch))]
            if macro > 0 or recompute_forward_context:  # a replayed macrobatch: every minibatch's replay first
                for generation in generations:
                    out += [(kinds.REPLAY_GATE, macro, generation, local) for local in range(replay_gate_up)]
                    out += [(kinds.REPLAY_UP, macro, generation, local) for local in range(replay_gate_up)]
                    out += [(kinds.REPLAY_SWIGLU, macro, generation, local) for local in range(replay - 2 * replay_gate_up)]
            for generation in generations:
                out += [(kinds.ROUTED_DOWN_DGRAD, macro, generation, local) for local in range(routed_down)]
                out += [(kinds.ROUTED_SWIGLU_BWD, macro, generation, local) for local in range(routed_swiglu)]
                out += [(kinds.ROUTED_GATE_UP_DGRAD, macro, generation, local) for local in range(routed_gate_up)]
            out += [(kinds.ROUTED_WGRAD_DOWN + matrix, macro, 0, local) for matrix in range(3) for local in range(wgrad_matrix)]
        return out

    def test_matches_reference_and_cluster_count(self):
        cases = replayed_first = 0
        for recompute_forward_context in (False, True):
            model, task_counts, kinds = _backward_task_model(recompute_forward_context)
            for hidden, intermediate, experts in ((256, 256, 2), (512, 768, 3), (384, 640, 2)):
                # The kernel's compile-time flag: with a 128-wide H / I tail the tile counts are ceilings.
                model.has_tail = hidden % 256 != 0 or intermediate % 256 != 0
                for macrobatch, minibatch in ((256, 256), (512, 256), (1024, 512), (768, 256)):
                    for num_tokens in range(0, 3 * macrobatch + 512, 256):  # 0, one macrobatch, a short last one
                        sizes = (256, hidden, hidden, intermediate, experts)  # shared tokens, hiddens, I, E
                        counts = task_counts(*sizes, minibatch, True, model.has_tail)
                        ref = self.reference(kinds, counts, num_tokens, macrobatch, minibatch, recompute_forward_context)
                        self.assertEqual(model._true_clusters(0, num_tokens, *sizes, macrobatch, minibatch), len(ref))
                        table = model._task_table(num_tokens, *sizes, macrobatch, minibatch)
                        got = [tuple(model._decode_mxfp8_task(task, *table)) for task in range(len(ref))]
                        self.assertEqual(got, ref, (recompute_forward_context, sizes, macrobatch, minibatch, num_tokens))
                        replayed_first += any(kind == kinds.REPLAY_GATE and macro == 0 for kind, macro, _, _ in got)
                        cases += 1
        self.assertGreater(cases, 100)
        self.assertGreater(replayed_first, 50)  # recompute_forward_context's macrobatch 0 ran its replay tasks


if __name__ == "__main__":
    unittest.main()
