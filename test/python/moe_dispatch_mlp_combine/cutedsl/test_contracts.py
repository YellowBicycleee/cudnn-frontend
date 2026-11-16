# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Host-side contracts and the megakernels' task arithmetic, on the CPU.

The contract and host helper modules are loaded by path (``_cutedsl_by_path``); the integer arithmetic of the
megakernels' communication streams and task decode is run as plain Python.  A few cases need a CUDA device for
their tensors and skip without one.  ``python -m pytest moe_dispatch_mlp_combine/cutedsl/test_contracts.py``.
"""

from __future__ import annotations

import bisect
import random
import re
import textwrap
import types
import unittest
import unittest.mock
from collections import Counter
from dataclasses import replace

import pytest
import torch

from ._cutedsl_by_path import CUTEDSL_DIR, load_cutedsl_module

# --------------------------------------------------------------------------------------------------------------------
# Tail flag of the hidden / intermediate sizes
#
# ``_host_common.has_tail``: the host's derivation of the megakernels' compile-time ``has_tail`` flag.
#
# CPU-only.  The tail instantiations of the kernels are generated for a tail of exactly ``TAIL_WIDTH`` = 128
# columns, half a 256-wide tile, so the helper that derives the flag accepts only extents that are whole
# 128-wide halves: a multiple of 256 (no tail) or 128 more than one (a tail).


host_common = load_cutedsl_module("_host_common")


class HasTail(unittest.TestCase):
    def test_the_tail_is_half_a_tile(self):
        self.assertEqual(host_common.TAIL_WIDTH, 128)

    def test_aligned_sizes_have_no_tail(self):
        for sizes in ((256,), (512, 512), (2048, 2048, 1024), (7168, 5120)):
            self.assertIs(host_common.has_tail(*sizes), False, sizes)

    def test_a_128_wide_tail_in_any_size(self):
        for sizes in ((384,), (640, 512), (512, 640), (512, 512, 384), (2688, 2048, 1280), (640, 640, 640)):
            self.assertIs(host_common.has_tail(*sizes), True, sizes)

    def test_sizes_off_the_128_grid_are_rejected(self):
        for size in (192, 320, 384 + 64, 100, 257):
            for sizes in ((size,), (512, size), (size, 640), (512, 640, size)):
                with self.assertRaises(ValueError, msg=sizes):
                    host_common.has_tail(*sizes)


# --------------------------------------------------------------------------------------------------------------------
# Launch-plan cache, executor keys and the communication tile option
#
# Host launch path of the CuTe DSL operators.
#
# CPU: the launch-plan cache (``_host_common.LaunchPlanCache``) reuses a plan only for the same live
# source tensors with unchanged address, shape, strides and dtype, and typed keys keep ``True``, ``1``
# and ``1.0`` (also inside lists) apart; an executor miss during a CUDA graph capture raises.  GPU: the scheduler's call-signature cache validates a signature once and
# in full on every new one.


def _load_host_common():
    return load_cutedsl_module("_host_common")


class LaunchPlanCache(unittest.TestCase):
    def test_plan_hits_only_for_unchanged_sources(self):
        common = _load_host_common()
        plans = common.LaunchPlanCache()
        a, b = torch.zeros(4, 8), torch.zeros(16)
        plans.put((a, b), ("geometry", 1), "plan")
        self.assertEqual(plans.get((a, b), ("geometry", 1)), "plan")
        self.assertIsNone(plans.get((a, b), ("geometry", 2)))  # other scalars
        self.assertIsNone(plans.get((a.clone(), b), ("geometry", 1)))  # other tensor object
        a.resize_(8, 4)  # same object, same storage, new shape
        self.assertIsNone(plans.get((a, b), ("geometry", 1)))
        a.resize_(4, 8)
        self.assertEqual(plans.get((a, b), ("geometry", 1)), "plan")
        a.data = a.view(torch.int32)  # same object and address, new dtype
        self.assertIsNone(plans.get((a, b), ("geometry", 1)))
        b.set_(torch.zeros(16).untyped_storage())  # same object, new address
        self.assertIsNone(plans.get((a, b), ("geometry", 1)))

    def test_freed_source_is_a_miss(self):
        common = _load_host_common()
        plans = common.LaunchPlanCache()
        a = torch.zeros(4)
        plans.put((a,), (), "plan")
        identity = id(a)
        del a
        reused = [torch.zeros(4) for _ in range(64)]  # likely to reuse the freed id
        for tensor in reused:
            if id(tensor) == identity:
                self.assertIsNone(plans.get((tensor,), ()))
        self.assertTrue(all(plans.get((tensor,), ()) is None for tensor in reused))
        held = torch.zeros(4)
        plans.put((held,), ("other",), "other")  # a put drops the plans of freed sources
        self.assertEqual([entry[2] for entry in plans._entries.values()], ["other"])

    def test_typed_keys_keep_bool_int_float_and_containers_apart(self):
        typed_key = _load_host_common().typed_key
        self.assertNotEqual(typed_key(True), typed_key(1))
        self.assertNotEqual(typed_key(1), typed_key(1.0))
        self.assertNotEqual(typed_key([1, 2]), typed_key((1, 2)))
        self.assertNotEqual(typed_key([1, 2]), typed_key([1.0, 2]))  # element types too
        self.assertEqual(typed_key([1, 2], None), typed_key([1, 2], None))


class CompileDuringCapture(unittest.TestCase):
    def test_executor_miss_during_a_capture_raises_without_compiling(self):
        # Every compile goes through persistent_executor: a miss while the current stream captures a CUDA graph
        # raises instead of compiling (or loading a kept executor) in the middle of the recording.
        common = _load_host_common()
        compiled = []
        with unittest.mock.patch.object(torch.cuda, "is_current_stream_capturing", return_value=True):
            with self.assertRaisesRegex(RuntimeError, "before CUDA graph capture"):
                common.persistent_executor(("kernel", 1), lambda: compiled.append(True))
        self.assertEqual(compiled, [])


class FwdEpilogueCombineVariant(unittest.TestCase):
    """CuteMoKConfig.epilogue_combine is a compiled variant of the BF16 forward mega: the two combine placements
    never share an executor.  (The launch-plan key is built inside ``forward_bf16``; that a call with the option does
    not reuse the default's plan is checked on the GPU, test_backward.py::_assert_variants_ran.)"""

    def test_executor_key_separates_the_variants(self):
        host = load_cutedsl_module("persistent_bf16_fwd")
        tensor = torch.zeros(2, 2)
        workspace = types.SimpleNamespace(x_buffer=tensor, combine_buffer=tensor)
        schedule = types.SimpleNamespace(peer_rank=tensor, peer_token_idx=tensor, num_tokens=tensor, tokens_per_expert=tensor, valid_tokens_per_expert=tensor)
        state = host.PersistentBf16State(*[tensor] * 14)  # nine outputs, five counters

        def key(**variant):
            return host._persistent_executor_key(
                workspace,
                schedule,
                (tensor,) * 6,
                state,
                plan=host.PersistentBf16Plan(),
                swiglu_limit_value=0.0,
                is_clamped=False,
                task_decode=host.TaskDecode.SMEM_BINARY,
                runtime_environment=(),
                device_index=0,
                context_key=0,
                **variant,
            )

        self.assertEqual(key(), key(epilogue_combine=False))  # the default is the base placement
        self.assertNotEqual(key(epilogue_combine=True), key(epilogue_combine=False))


class CommTileShapeVariant(unittest.TestCase):
    """The communication tile shape (contract COMM_TILE_SHAPES, ``CuteMoKConfig.comm_tile_shape``) is a compiled
    variant of the BF16 forward and backward megas: the two shapes never share an executor, and the default shape
    is the plans' and the config's default.  (That a call with the other shape runs its own executor is checked on
    the GPU, test_geometry.py.)"""

    def test_config_field_selects_the_shape(self):
        from cudnn.moe_dispatch_mlp_combine import functional
        from cudnn.moe_dispatch_mlp_combine.cutedsl.persistent_bf16_fwd_contract import CommTileShape

        workspace = types.SimpleNamespace(device=None, num_local_tokens=4096, topk=8, routed_hidden_size=4096)
        device = types.SimpleNamespace(multi_processor_count=148, shared_memory_per_block_optin=232448)

        def selected(config):
            with unittest.mock.patch.object(functional.torch.cuda, "get_device_properties", return_value=device):
                functional._validate_schedule_config(workspace, config)  # the options' validation path
            shape = functional._comm_tile_shape(workspace, config)
            return shape.rows, shape.columns

        default = functional.CuteMoKConfig()
        self.assertEqual(default.comm_tile_shape, (32, 512))
        self.assertEqual(selected(default), (32, 512))
        wide = selected(functional.CuteMoKConfig(comm_tile_shape=(16, 1024)))
        self.assertEqual(wide, (16, 1024))
        self.assertEqual(selected(functional.CuteMoKConfig(comm_tile_shape=CommTileShape(16, 1024))), wide)
        from_json = functional.CuteMoKConfig(comm_tile_shape=[16, 1024])  # a list: kept hashable as a tuple
        self.assertEqual(from_json.comm_tile_shape, (16, 1024))
        hash(from_json)
        for invalid in ((64, 256), (16, 512), (32,), (32, 512, 1), 16, "32x512", "Auto"):
            with self.assertRaisesRegex(ValueError, "comm_tile_shape must be"):
                selected(functional.CuteMoKConfig(comm_tile_shape=invalid))
        # "auto": 16 x 1024 for a hidden size that is a multiple of 1024, 32 x 512 otherwise.
        for hidden, expected in ((4096, (16, 1024)), (7168, (16, 1024)), (1024, (16, 1024)), (1152, (32, 512)), (640, (32, 512)), (3584, (32, 512))):
            auto_workspace = types.SimpleNamespace(**{**vars(workspace), "routed_hidden_size": hidden})
            shape = functional._comm_tile_shape(auto_workspace, functional.CuteMoKConfig(comm_tile_shape="auto"))
            self.assertEqual((shape.rows, shape.columns), expected, hidden)
        # MXFP8 routed weights have their own stream tile: only the default and "auto" pass.
        functional._require_default_comm_tile_shape(workspace, default)
        functional._require_default_comm_tile_shape(workspace, functional.CuteMoKConfig(comm_tile_shape="auto"))
        with self.assertRaisesRegex(ValueError, "BF16 megakernels"):
            functional._require_default_comm_tile_shape(workspace, functional.CuteMoKConfig(comm_tile_shape=(16, 1024)))

    def test_forward_executor_key_separates_the_shapes(self):
        host = load_cutedsl_module("persistent_bf16_fwd")
        tensor = torch.zeros(2, 2)
        workspace = types.SimpleNamespace(x_buffer=tensor, combine_buffer=tensor)
        schedule = types.SimpleNamespace(peer_rank=tensor, peer_token_idx=tensor, num_tokens=tensor, tokens_per_expert=tensor, valid_tokens_per_expert=tensor)
        state = host.PersistentBf16State(*[tensor] * 14)  # nine outputs, five counters

        def key(**plan):
            return host._persistent_executor_key(
                workspace,
                schedule,
                (tensor,) * 6,
                state,
                plan=host.PersistentBf16Plan(**plan),
                swiglu_limit_value=0.0,
                is_clamped=False,
                task_decode=host.TaskDecode.SMEM_BINARY,
                runtime_environment=(),
                device_index=0,
                context_key=0,
            )

        default, wide = host.COMM_TILE_SHAPES
        self.assertEqual(key(), key(comm_tile_shape=default))
        self.assertNotEqual(key(comm_tile_shape=wide), key())
        self.assertIn(("comm_tile_shape", (16, 1024)), key(comm_tile_shape=wide))
        host.PersistentBf16Plan(comm_tile_shape=wide).validate()
        with self.assertRaises(ValueError):
            host.PersistentBf16Plan(comm_tile_shape=host.CommTileShape(8, 2048)).validate()

    def test_backward_executor_key_separates_the_shapes(self):
        host = load_cutedsl_module("persistent_bf16_bwd")
        tensor = torch.zeros(2, 2)
        inputs = unittest.mock.Mock()  # every input tensor: dtype and rank only
        inputs.configure_mock(**{name: tensor for name in host.PersistentBf16BwdInputs.__dataclass_fields__})

        def key(**plan):
            return host._executor_cache_key(0, 0, host.PersistentBf16BwdPlan(**plan), inputs, host.TaskDecode.SMEM_BINARY)

        default, wide = load_cutedsl_module("persistent_bf16_fwd_contract").COMM_TILE_SHAPES
        self.assertEqual(key(), key(comm_tile_shape=default))
        self.assertNotEqual(key(comm_tile_shape=wide), key())
        self.assertIn(("comm_tile_shape", (16, 1024)), key(comm_tile_shape=wide))
        plan = host.PersistentBf16BwdPlan(comm_tile_shape=wide)
        self.assertIs(plan.validate(schedule_capacity=plan.schedule_capacity).comm_tile_shape, wide)


@unittest.skipUnless(torch.cuda.is_available(), "needs a CUDA device")
class SignatureCacheAddressChecks(unittest.TestCase):
    def test_scheduler_validates_once_per_signature_and_rejects_a_new_one(self):
        from cudnn.moe_dispatch_mlp_combine.cutedsl import scheduler as module

        device = torch.device("cuda", torch.cuda.current_device())
        experts, tokens, top_k, capacity = 4, 8, 2, 4 * 256
        topk = (torch.arange(tokens * top_k, dtype=torch.int32, device=device) % experts).view(1, tokens, top_k)
        module.scheduler(topk, experts, capacity, 0)
        torch.cuda.synchronize(device)
        # Same layouts and scalars (a signature hit): the full validation does not run again.
        original = module._validate_inputs
        module._validate_inputs = lambda *args: (_ for _ in ()).throw(AssertionError("validated twice"))
        try:
            module.scheduler(topk, experts, capacity, 0)
            # A new scalar (a different capacity) is a new signature: validated in full, so the bad rank is rejected.
            module._validate_inputs = original
            with self.assertRaisesRegex(ValueError, "rank"):
                module.scheduler(topk, experts, capacity + 256, 1)
        finally:
            module._validate_inputs = original
        torch.cuda.synchronize(device)


# --------------------------------------------------------------------------------------------------------------------
# Sliced (strided) weight views
#
# A routed weight may be a slice of a larger parameter, and its strides must survive.
#
# MCore owns the routed experts' Gate and Up as the two halves of one combined ``[E, 2I, H]``
# parameter and hands MOK each half as a view.  The kernels build one TMA descriptor per weight and
# never assume two are adjacent — the Gate/Up interleave exists in shared memory only — so a view is
# addressable as it stands and copying the halves apart every step is pure waste.
#
# Two host-side rules make that safe, and both are checked here:
#
# ``row_major_view`` must leave a strided tensor alone.  It rewrites the strides of a *contiguous*
# tensor whose singleton or empty dimensions carry arbitrary ones, which the ABI descriptors need;
# applied to a genuine slice it would describe a different region of memory and return wrong data
# with no error at all.  That is the failure this file exists to prevent.
#
# ``validate_tensor(sliced=True)`` must accept the view and still reject what TMA cannot address:
# a non-contiguous innermost dimension, an outer stride that is not 16-byte aligned, a self-
# overlapping layout, and a base pointer off a 16-byte boundary.
#
# CPU only, no CUTLASS and no GPU for the stride rules: ``python -m pytest moe_dispatch_mlp_combine/cutedsl/test_contracts.py``.


_host_common = load_cutedsl_module("_host_common")


row_major_view = _host_common.row_major_view


validate_tensor = _host_common.validate_tensor


TMA_ALIGNMENT_BYTES = _host_common.TMA_ALIGNMENT_BYTES


# A small stand-in for a combined routed FC1: [E, 2I, H], Gate the first I rows, Up the second.
EXPERTS, INTERMEDIATE, HIDDEN = 3, 8, 16


def _combined() -> torch.Tensor:
    count = EXPERTS * 2 * INTERMEDIATE * HIDDEN
    return torch.arange(count, dtype=torch.bfloat16).reshape(EXPERTS, 2 * INTERMEDIATE, HIDDEN)


def _halves(combined: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    return combined[:, :INTERMEDIATE, :], combined[:, INTERMEDIATE:, :]


def test_row_major_view_keeps_a_slice_addressing_its_own_rows():
    """The regression this file is named for: rewriting a slice's strides reads other rows."""
    combined = _combined()
    gate, up = _halves(combined)
    assert not gate.is_contiguous() and not up.is_contiguous()

    for name, half in (("gate", gate), ("up", up)):
        viewed = row_major_view(half)
        assert viewed.stride() == half.stride(), f"{name}: strides were rewritten"
        assert viewed.data_ptr() == half.data_ptr(), f"{name}: base moved"
        # The values are the test: canonical strides would read a contiguous run instead.
        assert torch.equal(viewed, half), f"{name}: reads different memory"

    # And the two halves really are different rows of the parameter.
    assert not torch.equal(gate, up)


def test_row_major_view_still_canonicalizes_singleton_strides():
    """What the function is for: a contiguous tensor whose singleton dimension carries any stride."""
    contiguous = torch.zeros(4, 1, 6)
    odd = contiguous.as_strided(contiguous.shape, (6, 999, 1))
    assert odd.is_contiguous()  # a singleton dimension's stride does not affect contiguity
    assert row_major_view(odd).stride() == (6, 6, 1)


def test_row_major_view_returns_a_plain_contiguous_tensor_unchanged():
    contiguous = torch.zeros(4, 5, 6)
    assert row_major_view(contiguous) is contiguous


@pytest.mark.skipif(not torch.cuda.is_available(), reason="validate_tensor requires a CUDA tensor")
class TestValidateTensor:
    """``validate_tensor`` needs CUDA tensors; the stride rules above do not."""

    @staticmethod
    def _cuda_combined() -> torch.Tensor:
        return _combined().cuda()

    def test_accepts_either_half_as_sliced(self):
        device = torch.device("cuda", torch.cuda.current_device())
        gate, up = _halves(self._cuda_combined())
        for name, half in (("gate", gate), ("up", up)):
            validate_tensor(name, half, shape=(EXPERTS, INTERMEDIATE, HIDDEN), dtype=torch.bfloat16, device=device, sliced=True)

    def test_rejects_a_slice_when_not_sliced(self):
        device = torch.device("cuda", torch.cuda.current_device())
        gate, _ = _halves(self._cuda_combined())
        with pytest.raises(ValueError, match="contiguous"):
            validate_tensor("gate", gate, shape=(EXPERTS, INTERMEDIATE, HIDDEN), dtype=torch.bfloat16, device=device)

    def test_rejects_a_non_contiguous_last_dimension(self):
        device = torch.device("cuda", torch.cuda.current_device())
        # Every other column: the innermost stride is 2, which TMA's box cannot express here.
        strided = self._cuda_combined()[:, :INTERMEDIATE, ::2]
        with pytest.raises(ValueError, match="last dimension"):
            validate_tensor("gate", strided, shape=(EXPERTS, INTERMEDIATE, HIDDEN // 2), dtype=torch.bfloat16, device=device, sliced=True)

    def test_rejects_an_outer_stride_that_is_not_aligned(self):
        device = torch.device("cuda", torch.cuda.current_device())
        elements = TMA_ALIGNMENT_BYTES // 2  # bfloat16
        assert HIDDEN % elements == 0, "pick a HIDDEN whose rows are aligned to start with"
        base = torch.zeros(EXPERTS * INTERMEDIATE * HIDDEN * 4, dtype=torch.bfloat16, device=device)
        # A row stride one element past the aligned one.
        misaligned = base.as_strided((EXPERTS, INTERMEDIATE, HIDDEN), ((HIDDEN + 1) * INTERMEDIATE, HIDDEN + 1, 1))
        with pytest.raises(ValueError, match="multiple of"):
            validate_tensor("gate", misaligned, shape=(EXPERTS, INTERMEDIATE, HIDDEN), dtype=torch.bfloat16, device=device, sliced=True)

    def test_rejects_a_self_overlapping_layout(self):
        device = torch.device("cuda", torch.cuda.current_device())
        base = torch.zeros(EXPERTS * INTERMEDIATE * HIDDEN, dtype=torch.bfloat16, device=device)
        # Every expert starts at the same place: the experts alias each other.
        overlapping = base.as_strided((EXPERTS, INTERMEDIATE, HIDDEN), (0, HIDDEN, 1))
        with pytest.raises(ValueError, match="overlaps itself"):
            validate_tensor("gate", overlapping, shape=(EXPERTS, INTERMEDIATE, HIDDEN), dtype=torch.bfloat16, device=device, sliced=True)

    def test_rejects_a_misaligned_base_pointer(self):
        device = torch.device("cuda", torch.cuda.current_device())
        elements = TMA_ALIGNMENT_BYTES // 2
        base = torch.zeros(EXPERTS * INTERMEDIATE * HIDDEN + elements, dtype=torch.bfloat16, device=device)
        offset = base[1:]  # one bfloat16 past the aligned base
        assert offset.data_ptr() % TMA_ALIGNMENT_BYTES
        with pytest.raises(ValueError, match="aligned"):
            validate_tensor(
                "gate",
                offset[: EXPERTS * INTERMEDIATE * HIDDEN].view(EXPERTS, INTERMEDIATE, HIDDEN),
                shape=(EXPERTS, INTERMEDIATE, HIDDEN),
                dtype=torch.bfloat16,
                device=device,
                sliced=True,
            )


# --------------------------------------------------------------------------------------------------------------------
# Backward geometry with a routed hidden size
#
# CPU checks of the backward contract with separate shared and routed hiddens (no GPU, no torch, no
# CuTe DSL import: the contract modules use the standard library only and are loaded by path).
#
# The shared chain (shared Gate/Up dgrad, shared Wgrads, d_x_shared, shared weights) sizes from hidden_size,
# the routed chain (rings, communication readiness, routed tasks and Wgrads, routed weights) from
# routed_hidden_size; equal hiddens give exactly the ordinary geometry.


contract = load_cutedsl_module("persistent_bf16_bwd_contract")


PAIRS = ((4096, 1024), (7168, 2048), (4096, 1280), (4096, 5120), (2048, 2048))


def _geometry(hidden, routed):
    return contract.BwdGeometry(
        num_local_tokens=8192,
        hidden_size=hidden,
        routed_hidden_size=routed,
        intermediate_size=1024,
        schedule_capacity=81920,
        macrobatch_size=32768,
        minibatch_size=4096,
        num_local_experts=64,
        topk=10,
        num_comm_sms=22,
    )


class RoutedHiddenContract(unittest.TestCase):
    def test_default_and_equal_hiddens_are_the_ordinary_geometry(self):
        default = replace(_geometry(4096, 1024), routed_hidden_size=None)
        self.assertEqual(default.routed_hidden_size, 4096)
        equal = _geometry(4096, 4096)
        self.assertEqual(contract.task_counts(default), contract.task_counts(equal))
        self.assertEqual(contract.raw13_descriptors(default), contract.raw13_descriptors(equal))
        with self.assertRaises(ValueError):
            _geometry(4096, 1000)  # 256-aligned like every CUDA tile dimension
        with self.assertRaises(ValueError):
            _geometry(4096, -256)

    def test_task_counts_follow_each_chain(self):
        for hs, hr in PAIRS:
            g = _geometry(hs, hr)
            counts = contract.task_counts(g)
            i_cols = 1024 // 256
            self.assertEqual(counts.shared_gate_up_dgrad, (8192 // 256) * (hs // 256))
            self.assertEqual((counts.shared_wgrad_down, counts.shared_wgrad_gate, counts.shared_wgrad_up), (i_cols * (hs // 256),) * 3)
            self.assertEqual(counts.routed_gate_up_dgrad, (4096 // 256) * (hr // 256))
            self.assertEqual(counts.wgrad_tile_tasks, i_cols * (hr // 256))
            # routed readiness: communication tiles (32 x 512, short last block) and Gate/Up dgrad tiles
            self.assertEqual(g.d_y_ready_required, (4096 // 32) * -(-hr // 512))
            self.assertEqual(g.d_x_ready_required, (4096 // 256) * (hr // 256) * 2)

    def test_raw13_shapes_follow_each_chain(self):
        for hs, hr in PAIRS:
            shapes = {d.name: d.shape for d in contract.raw13_descriptors(_geometry(hs, hr))}
            self.assertEqual(len(shapes), 13)
            self.assertNotIn("d_hidden_shared", shapes)
            self.assertNotIn("d_hidden_routed", shapes)
            t, b, e, i = 8192, 32768, 64, 1024
            self.assertEqual(shapes["d_x_shared"], (t, hs))
            self.assertEqual((shapes["d_x_routed"], shapes["d_y_routed"]), ((b, hr), (b, hr)))
            self.assertEqual((shapes["d_w_shared_gate"], shapes["d_w_shared_up"], shapes["d_w_shared_down"]), ((i, hs), (i, hs), (hs, i)))
            self.assertEqual((shapes["d_w_routed_gate"], shapes["d_w_routed_up"], shapes["d_w_routed_down"]), ((e, i, hr), (e, i, hr), (e, hr, i)))

    def test_wgrad_tile_coordinates_cover_each_chain(self):
        family = contract.TaskFamily
        for hs, hr in PAIRS:
            g = _geometry(hs, hr)
            for fam, hidden in ((family.SHARED_WGRAD_DOWN, hs), (family.ROUTED_WGRAD_DOWN, hr)):
                tiles = (hidden // 256) * 4
                coords = {contract._wgrad_coord(g, fam, tile) for tile in range(tiles)}
                self.assertEqual(coords, {(r, c) for r in range(hidden // 256) for c in range(4)})
            for fam, hidden in ((family.SHARED_WGRAD_GATE, hs), (family.ROUTED_WGRAD_UP, hr)):
                tiles = (hidden // 256) * 4
                coords = {contract._wgrad_coord(g, fam, tile) for tile in range(tiles)}
                self.assertEqual(coords, {(r, c) for r in range(4) for c in range(hidden // 256)})


# --------------------------------------------------------------------------------------------------------------------
# EP size as a runtime value
#
# The peer count (EP size) is a runtime value taken from the workspace.
#
# CPU-only contract checks, parameterized over ep_size in {1, 4, 8, 16, 64}:
#   * every host plan/geometry accepts the size and threads it to the pointer-list
#     and rank validation (no `8` literal anywhere on the host path);
#   * the peer selectors in the shared BWD contract use the given ep_size;
#   * the four kernel sources carry no EP8 constant: no `EP_SIZE`, no
#     `peer_rank < Int32(8)` guard, no 8-way pointer select; peer bases come from
#     an address table indexed by the runtime rank (CUDA `peer_buf[peer_rank]`);
#   * the host builds the peer address table with the pointer-list length.


EP_SIZES = (1, 4, 8, 16, 64)


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


# --------------------------------------------------------------------------------------------------------------------
# Communication task streams of the BF16 megakernels
#
# The communication task streams of the forward and backward prefixes, checked on the CPU.
#
# ``_stream_task`` (and the helpers ``_reverse_tile`` / ``_step_task``) are integer arithmetic on runtime
# values; here they run as plain Python (``Int32`` -> ``int``) over a grid of macrobatch / minibatch shapes,
# including the short last macrobatch / generation, one macrobatch, ``mpm`` = 1 and odd tile counts, and
# are checked against a readable reference enumeration of the same order.  Three properties per shape:
# every task appears exactly once (bijection); the stream matches the reference; a task comes after
# every task it depends on through the kernel's readiness counters, so the smallest unfinished task is
# always ready (no deadlock).  A task is one tile row (all hidden columns); the edges, at the
# granularity the counters have:
#   forward   push(m, row in minibatch j) waits the routed Down of (m, j), which needs every pull of
#             (m, j) [x_routed_ready] and, with a next macrobatch, every push of (m + 1, j)
#             [y_routed_done]; pull(m - 1, row in j) waits the same Down
#   backward  dX(g) waits the Gate/Up dgrad of g, which needs every X and dY row of g and every dX of
#             g - mpm [reverse_dispatch_done]; X / dY(g) for g >= mpm wait the compute readers of
#             g - mpm [routed_buffers_done], which need every pull of g - mpm and every dX of g - 2 mpm
#             (recompute_forward_context: every generation has X rows, the first macrobatch's included)
#
# The tile shape (rows x columns of one slot, the contract's ``CommTileShape``) is the compile-time parameter of
# the communication CTAs; ``SlotShape`` runs the same checks at real geometries for both shapes and several hidden sizes,
# and checks the counts that follow the shape: the arrivals of every readiness counter against what the compute
# side waits for, and the bytes of the column blocks.


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


# --------------------------------------------------------------------------------------------------------------------
# Communication task streams and task decode of the MXFP8 megakernels
#
# The MXFP8 megas' communication task stream decode, checked on the CPU.
#
# ``_stream_task`` (the forward stream's task -> (queue, macrobatch, tile) decode), ``_reciprocal_u32`` and
# ``_divide_u32`` (the division-free block decode) of ``cudnn/moe_dispatch_mlp_combine/cutedsl/_persistent_mxfp8_fwd_mega.py``, and
# ``_backward_stream_task`` (the backward stream's per-macrobatch task -> (queue, tile) decode) of
# ``_persistent_mxfp8_bwd_mega.py``, are integer arithmetic on runtime values; here they run as plain Python
# (``Int32`` / ``Int64`` -> ``int``, ``mul.hi.u32`` as a Python product) over a grid of macrobatch shapes, including
# one macrobatch, a short last macrobatch and tile counts that are not powers of two, and are checked against a
# readable reference enumeration of the order the docstrings describe.  Three properties per shape: every task
# appears exactly once and their number is the block total the kernel computes (bijection, totals match); the stream
# equals the reference; every task comes after the tasks its readiness counter depends on (forward: a push of
# macrobatch m and a pull of m - 1 wait for the routed Down of m, which needs every pull of m and, with a macrobatch
# after m, every push of m + 1; backward: a push waits for its macrobatch's Gate / Up dgrad, which needs every pull
# of the macrobatch), so the smallest unfinished task is always ready.  The division is checked against ``//`` for
# divisors from 1 (where the reciprocal wraps to 0xFFFFFFFF) up, over the dividends the streams produce.
#
# The backward's compute-task decode (``_task_table`` / ``_decode_mxfp8_task``, with the cluster count of
# ``_true_clusters``) runs the same way for both values of the kernel's compile-time ``recompute_forward_context`` flag
# and is checked against the task order of its docstring: with ``recompute_forward_context`` macrobatch 0 has the replay
# tasks of a replayed macrobatch (and its stream the x pulls: ``_backward_stream_task`` with twice the tiles as pull
# tasks).
#
# CPU only: ``python -m pytest moe_dispatch_mlp_combine/cutedsl/test_contracts.py`` (no torch, no GPU).


STREAM_PULL, STREAM_PUSH, STREAM_PULL_X = 0, 1, 2


MODULE = "_persistent_mxfp8_fwd_mega"


BWD_MODULE = "_persistent_mxfp8_bwd_mega"


def _mul_hi_u32(a: int, b: int) -> int:
    """``mul.hi.u32``: the high 32 bits of the 64-bit product of the operands' low 32 bits."""
    return ((a & 0xFFFFFFFF) * (b & 0xFFFFFFFF)) >> 32


def _mxfp8_device_functions(*names: str, module: str = MODULE) -> dict:
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
        f = _mxfp8_device_functions("_reciprocal_u32", "_divide_u32")
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


class Mxfp8ForwardStream(unittest.TestCase):
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
        f = _mxfp8_device_functions("_reciprocal_u32", "_divide_u32", "_stream_task")
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


class Mxfp8BackwardStream(unittest.TestCase):
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
        f = _mxfp8_device_functions("_backward_stream_task", module=BWD_MODULE)
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


# --------------------------------------------------------------------------------------------------------------------
# EP8 expert and routed-task decoders (CPU references)
#
# CPU references for EP8 expert and routed-task decoders.


MLP_MB = 256


def _expert_blocks(tokens_per_expert: list[int]) -> list[int]:
    assert all(rows >= 0 and rows % MLP_MB == 0 for rows in tokens_per_expert)
    return [rows // MLP_MB for rows in tokens_per_expert]


def _legacy_decode(
    tokens_per_expert: list[int],
    minibatch_blocks: int,
    global_minibatch: int,
    col_blocks: int,
    task: int,
) -> tuple[int, int, int, int] | None:
    """Mirror the original per-expert scan in grouped_gemm.cuh."""
    minibatch_begin = global_minibatch * minibatch_blocks
    minibatch_end = minibatch_begin + minibatch_blocks
    expert_begin = 0
    for expert, blocks in enumerate(_expert_blocks(tokens_per_expert)):
        first_block = max(minibatch_begin, expert_begin)
        row_blocks = max(0, min(minibatch_end, expert_begin + blocks) - first_block)
        expert_tasks = row_blocks * col_blocks
        if task < expert_tasks:
            return expert, first_block, row_blocks, task
        task -= expert_tasks
        expert_begin += blocks
    return None


def _prefix_decode(
    tokens_per_expert: list[int],
    minibatch_blocks: int,
    global_minibatch: int,
    col_blocks: int,
    task: int,
) -> tuple[tuple[int, int, int, int] | None, int]:
    """Mirror six unrolled upper-bound steps and the E>64 fallback loop."""
    offsets = [0]
    for blocks in _expert_blocks(tokens_per_expert):
        offsets.append(offsets[-1] + blocks)

    minibatch_begin = global_minibatch * minibatch_blocks
    global_task_block = minibatch_begin + task // col_blocks
    if global_task_block >= offsets[-1]:
        return None, 0

    expert_begin = 0
    expert_end = len(tokens_per_expert) - 1
    comparisons = 0
    for _ in range(6):
        if expert_begin < expert_end:
            expert_mid = (expert_begin + expert_end) // 2
            comparisons += 1
            if offsets[expert_mid + 1] <= global_task_block:
                expert_begin = expert_mid + 1
            else:
                expert_end = expert_mid
    while expert_begin < expert_end:
        expert_mid = (expert_begin + expert_end) // 2
        comparisons += 1
        if offsets[expert_mid + 1] <= global_task_block:
            expert_begin = expert_mid + 1
        else:
            expert_end = expert_mid

    expert = expert_begin
    first_block = max(minibatch_begin, offsets[expert])
    row_blocks = min(minibatch_begin + minibatch_blocks, offsets[expert + 1]) - first_block
    expert_task = task - (first_block - minibatch_begin) * col_blocks
    return (expert, first_block, row_blocks, expert_task), comparisons


def _assert_matches_legacy(tokens_per_expert: list[int], minibatch_blocks: int, col_blocks: int) -> list[int]:
    total_blocks = sum(_expert_blocks(tokens_per_expert))
    num_minibatches = (total_blocks + minibatch_blocks - 1) // minibatch_blocks
    comparisons = []
    for global_minibatch in range(num_minibatches):
        for task in range(minibatch_blocks * col_blocks):
            decoded, count = _prefix_decode(
                tokens_per_expert,
                minibatch_blocks,
                global_minibatch,
                col_blocks,
                task,
            )
            assert decoded == _legacy_decode(
                tokens_per_expert,
                minibatch_blocks,
                global_minibatch,
                col_blocks,
                task,
            )
            if decoded is not None:
                comparisons.append(count)
    return comparisons


def _random_partition(rng: random.Random, total_blocks: int, experts: int) -> list[int]:
    counts = [0] * experts
    for _ in range(total_blocks):
        counts[rng.randrange(experts)] += MLP_MB
    return counts


def _task_widths(minibatch: int, hidden: int, intermediate: int) -> tuple[int, int]:
    row_blocks = minibatch // MLP_MB
    intermediate_blocks = intermediate // MLP_MB
    hidden_blocks = hidden // MLP_MB
    swiglu_tiles = (minibatch // 128) * (intermediate // 128)
    swiglu_tasks = (swiglu_tiles + 4 - 1) // 4
    minibatch_bwd_tasks = row_blocks * intermediate_blocks + swiglu_tasks + row_blocks * hidden_blocks
    segment_wgrad_tasks = 3 * intermediate_blocks * hidden_blocks
    return minibatch_bwd_tasks, segment_wgrad_tasks


def _replay_task_width(minibatch: int, intermediate: int, *, paired_gate_up: bool = False) -> int:
    row_blocks = minibatch // MLP_MB
    intermediate_blocks = intermediate // MLP_MB
    swiglu_tiles = (minibatch // 128) * (intermediate // 128)
    swiglu_tasks = (swiglu_tiles + 6 - 1) // 6
    gate_up_tasks = row_blocks * intermediate_blocks
    return (1 if paired_gate_up else 2) * gate_up_tasks + swiglu_tasks


def _paired_replay_enabled(
    *,
    minibatch_release: bool,
    num_devices: int,
    use_mxfp8: bool,
    full_context: bool,
    num_macrobatches: int,
) -> bool:
    return minibatch_release and num_devices == 8 and not use_mxfp8 and not full_context and num_macrobatches > 1


def _completed_segment_offsets(tokens_per_expert: list[int], macrobatch: int, minibatch: int) -> list[int]:
    assert macrobatch > 0 and macrobatch % minibatch == 0
    routed_rows = sum(tokens_per_expert)
    num_minibatches = (routed_rows + minibatch - 1) // minibatch
    offsets = [0]
    completed = 0
    for global_minibatch in range(num_minibatches):
        minibatch_begin = global_minibatch * minibatch
        macrobatch_begin = (minibatch_begin // macrobatch) * macrobatch
        macrobatch_end = min(macrobatch_begin + macrobatch, routed_rows)
        expert_begin = 0
        for expert_rows in tokens_per_expert:
            expert_end = expert_begin + expert_rows
            segment_begin = max(expert_begin, macrobatch_begin)
            segment_end = min(expert_end, macrobatch_end)
            if segment_begin < segment_end and (segment_end - 1) // minibatch == global_minibatch:
                completed += 1
            expert_begin = expert_end
        offsets.append(completed)
    return offsets


def _completed_segment_experts_by_owner(tokens_per_expert: list[int], macrobatch: int, minibatch: int) -> list[list[int]]:
    routed_rows = sum(tokens_per_expert)
    num_minibatches = (routed_rows + minibatch - 1) // minibatch
    experts_by_owner: list[list[int]] = [[] for _ in range(num_minibatches)]
    expert_begin = 0
    for expert, expert_rows in enumerate(tokens_per_expert):
        expert_end = expert_begin + expert_rows
        first_macrobatch = expert_begin // macrobatch
        last_macrobatch = (expert_end - 1) // macrobatch if expert_rows else -1
        for macrobatch_idx in range(first_macrobatch, last_macrobatch + 1):
            segment_begin = max(expert_begin, macrobatch_idx * macrobatch)
            segment_end = min(expert_end, (macrobatch_idx + 1) * macrobatch)
            if segment_begin < segment_end:
                owner = (segment_end - 1) // minibatch
                experts_by_owner[owner].append(expert)
        expert_begin = expert_end
    return experts_by_owner


def _task_prefixes(
    completed_segment_offsets: list[int],
    minibatch_bwd_tasks: int,
    segment_wgrad_tasks: int,
    saved_context_minibatches: int | None = None,
    minibatch_replay_tasks: int = 0,
) -> list[int]:
    num_minibatches = len(completed_segment_offsets) - 1
    if saved_context_minibatches is None:
        saved_context_minibatches = num_minibatches
    return [
        owner * minibatch_bwd_tasks + max(0, owner - saved_context_minibatches) * minibatch_replay_tasks + completed * segment_wgrad_tasks
        for owner, completed in enumerate(completed_segment_offsets)
    ]


def _build_task_owner_buckets(prefixes: list[int], bucket_width: int) -> list[int]:
    num_minibatches = len(prefixes) - 1
    num_buckets = (prefixes[-1] + bucket_width - 1) // bucket_width
    buckets = []
    owner = 0
    for bucket in range(num_buckets):
        bucket_begin = bucket * bucket_width
        while owner + 1 < num_minibatches and prefixes[owner + 1] <= bucket_begin:
            owner += 1
        buckets.append(owner)
    return buckets


def _bucket_decode(prefixes: list[int], buckets: list[int], bucket_shift: int, idx: int) -> tuple[int, int]:
    owner = buckets[idx >> bucket_shift]
    if idx >= prefixes[owner + 1]:
        owner += 1
    return owner, idx - prefixes[owner]


def _linear_task_owner(prefixes: list[int], idx: int) -> tuple[int, int]:
    for owner in range(len(prefixes) - 1):
        if idx < prefixes[owner + 1]:
            return owner, idx - prefixes[owner]
    raise AssertionError("valid task index has no owner")


def _binary_task_owner(prefixes: list[int], idx: int) -> tuple[int, int]:
    owner = bisect.bisect_right(prefixes, idx) - 1
    return owner, idx - prefixes[owner]


def _assert_task_owner_case(
    tokens_per_expert: list[int],
    macrobatch: int,
    minibatch: int,
    hidden: int,
    intermediate: int,
    replay: bool = False,
    paired_gate_up: bool = False,
) -> list[int]:
    completed = _completed_segment_offsets(tokens_per_expert, macrobatch, minibatch)
    minibatch_tasks, segment_tasks = _task_widths(minibatch, hidden, intermediate)
    saved_context_minibatches = macrobatch // minibatch
    replay_tasks = _replay_task_width(minibatch, intermediate, paired_gate_up=paired_gate_up) if replay else 0
    prefixes = _task_prefixes(
        completed,
        minibatch_tasks,
        segment_tasks,
        saved_context_minibatches,
        replay_tasks,
    )
    bucket_shift = minibatch_tasks.bit_length() - 1
    bucket_width = 1 << bucket_shift
    buckets = _build_task_owner_buckets(prefixes, bucket_width)
    assert bucket_width <= minibatch_tasks
    assert all(prefixes[owner + 1] - prefixes[owner] >= bucket_width for owner in range(len(prefixes) - 1))
    for idx in range(prefixes[-1]):
        decoded = _bucket_decode(prefixes, buckets, bucket_shift, idx)
        assert decoded == _linear_task_owner(prefixes, idx)
        assert decoded == _binary_task_owner(prefixes, idx)
    experts_by_owner = _completed_segment_experts_by_owner(tokens_per_expert, macrobatch, minibatch)
    completed_experts = [expert for owner_experts in experts_by_owner for expert in owner_experts]
    for owner, owner_experts in enumerate(experts_by_owner):
        begin = completed[owner]
        end = completed[owner + 1]
        assert completed_experts[begin:end] == owner_experts
    return completed


def test_seeded_decoder_matches_legacy_with_empty_experts() -> None:
    rng = random.Random(20260823)
    _assert_matches_legacy([0, 256, 0, 768, 0, 256], 2, 7)
    for _ in range(40):
        experts = rng.choice([1, 3, 16, 63, 64, 65, 128])
        tokens = _random_partition(rng, rng.randint(1, 80), experts)
        _assert_matches_legacy(
            tokens,
            rng.choice([1, 2, 4, 8, 16]),
            rng.choice([1, 4, 7]),
        )


def test_qwen_ep8_uses_exactly_six_unrolled_steps() -> None:
    tokens = [0, 512, 0, 256] * 16
    comparisons = _assert_matches_legacy(tokens, 8, 4)
    assert comparisons and set(comparisons) == {6}


def test_more_than_64_experts_uses_fallback() -> None:
    tokens = [0, 512, 0, 256] * 32
    comparisons = _assert_matches_legacy(tokens, 8, 4)
    assert comparisons and all(count > 6 for count in comparisons)


def test_scheduled_tail_tasks_remain_noops() -> None:
    tokens = [0, 256, 0, 512]
    minibatch_blocks = 8
    col_blocks = 4
    valid_tasks = sum(_expert_blocks(tokens)) * col_blocks
    for task in range(valid_tasks, minibatch_blocks * col_blocks):
        decoded, _ = _prefix_decode(tokens, minibatch_blocks, 0, col_blocks, task)
        assert decoded is None
        assert _legacy_decode(tokens, minibatch_blocks, 0, col_blocks, task) is None


def test_task_owner_buckets_cover_empty_split_and_partial_tail() -> None:
    empty = _assert_task_owner_case([0, 512, 0, 256], 512, 256, 1024, 512)
    assert empty == [0, 0, 1, 2]

    split = _assert_task_owner_case([768, 256], 512, 256, 1024, 512)
    assert split == [0, 0, 1, 2, 3]

    partial_tail = _assert_task_owner_case([256, 512], 1024, 512, 2048, 1024)
    assert partial_tail == [0, 1, 2]


def test_seeded_task_owner_buckets_match_both_references() -> None:
    rng = random.Random(20260824)
    for _ in range(40):
        minibatch = rng.choice([256, 512, 1024])
        macrobatch = minibatch * rng.choice([1, 2, 4, 8])
        experts = rng.choice([1, 4, 16, 64])
        tokens = _random_partition(rng, rng.randint(1, 48), experts)
        _assert_task_owner_case(
            tokens,
            macrobatch,
            minibatch,
            rng.choice([1024, 2048, 4096]),
            rng.choice([512, 1024, 2048]),
        )


def test_replay_aware_owner_buckets_cover_threshold_empty_split_and_tail() -> None:
    cases = [
        ([0, 512, 0, 768, 256], 512, 256, 1024, 512),
        ([768, 0, 256, 512], 512, 256, 2048, 1024),
        ([0, 768, 256, 0, 768], 1024, 512, 4096, 1024),
    ]
    for tokens, macrobatch, minibatch, hidden, intermediate in cases:
        completed = _assert_task_owner_case(
            tokens,
            macrobatch,
            minibatch,
            hidden,
            intermediate,
            replay=True,
        )
        bwd_tasks, segment_tasks = _task_widths(minibatch, hidden, intermediate)
        replay_tasks = _replay_task_width(minibatch, intermediate)
        saved = macrobatch // minibatch
        prefixes = _task_prefixes(completed, bwd_tasks, segment_tasks, saved, replay_tasks)
        if len(prefixes) - 1 > saved:
            assert prefixes[saved + 1] - prefixes[saved] >= (bwd_tasks + replay_tasks)
            assert _binary_task_owner(prefixes, prefixes[saved]) == (saved, 0)


def test_seeded_replay_aware_owner_buckets_match_both_references() -> None:
    rng = random.Random(20260824 ^ 0xB300)
    for _ in range(40):
        minibatch = rng.choice([256, 512, 1024])
        macrobatch = minibatch * rng.choice([1, 2, 4, 8])
        experts = rng.choice([1, 4, 16, 64])
        tokens = _random_partition(rng, rng.randint(1, 96), experts)
        _assert_task_owner_case(
            tokens,
            macrobatch,
            minibatch,
            rng.choice([1024, 2048, 4096]),
            rng.choice([512, 1024, 2048]),
            replay=True,
        )


def test_qwen_100k_b131072_replay_task_count_and_bucket_capacity() -> None:
    # 64 EP8-local expert spans use the three padded sizes observed in the
    # Qwen-shaped 100K benchmark.  This ordering avoids coincident expert and
    # macrobatch boundaries, giving the worst-case E + M - 1 segments.
    expert_blocks = [64] * 32 + [63] * 16 + [62] * 16
    random.Random(0).shuffle(expert_blocks)
    tokens = [blocks * MLP_MB for blocks in expert_blocks]
    macrobatch = 131072
    minibatch = 4096
    completed = _completed_segment_offsets(tokens, macrobatch, minibatch)
    bwd_tasks, segment_tasks = _task_widths(minibatch, 4096, 1024)
    replay_tasks = _replay_task_width(minibatch, 1024)
    saved = macrobatch // minibatch
    prefixes = _task_prefixes(completed, bwd_tasks, segment_tasks, saved, replay_tasks)
    bucket_shift = bwd_tasks.bit_length() - 1
    buckets = _build_task_owner_buckets(prefixes, 1 << bucket_shift)

    assert sum(tokens) == 1036288
    assert len(completed) - 1 == 253
    assert saved == 32
    assert bwd_tasks == 384
    assert replay_tasks == 171
    assert segment_tasks == 192
    assert completed[-1] == 71
    assert prefixes[-1] == 148575
    assert len(buckets) == 581

    boundary_indices = {index for prefix in prefixes for index in (prefix - 1, prefix, prefix + 1) if 0 <= index < prefixes[-1]}
    rng = random.Random(20260824)
    sampled_indices = boundary_indices | {rng.randrange(prefixes[-1]) for _ in range(10000)}
    for idx in sampled_indices:
        decoded = _bucket_decode(prefixes, buckets, bucket_shift, idx)
        assert decoded == _binary_task_owner(prefixes, idx)


def test_qwen_100k_paired_replay_counts_and_bucket_capacity() -> None:
    expert_blocks = [64] * 32 + [63] * 16 + [62] * 16
    random.Random(0).shuffle(expert_blocks)
    tokens = [blocks * MLP_MB for blocks in expert_blocks]
    macrobatch = 131072
    minibatch = 4096
    completed = _completed_segment_offsets(tokens, macrobatch, minibatch)
    bwd_tasks, segment_tasks = _task_widths(minibatch, 4096, 1024)
    old_replay_tasks = _replay_task_width(minibatch, 1024)
    paired_replay_tasks = _replay_task_width(minibatch, 1024, paired_gate_up=True)
    saved = macrobatch // minibatch
    old_prefixes = _task_prefixes(completed, bwd_tasks, segment_tasks, saved, old_replay_tasks)
    paired_prefixes = _task_prefixes(completed, bwd_tasks, segment_tasks, saved, paired_replay_tasks)
    bucket_shift = bwd_tasks.bit_length() - 1
    buckets = _build_task_owner_buckets(paired_prefixes, 1 << bucket_shift)

    assert bwd_tasks == 384
    assert segment_tasks == 192
    assert old_replay_tasks == 171
    assert paired_replay_tasks == 107
    assert len(completed) - 1 == 253
    assert saved == 32
    assert len(completed) - 1 - saved == 221
    assert completed[-1] == 71
    assert old_prefixes[-1] == 148575
    assert paired_prefixes[-1] == 134431
    assert old_prefixes[-1] - paired_prefixes[-1] == 221 * 64
    assert len(buckets) == 526
    assert 1 << bucket_shift == 256

    # The 100K harness reserves 4,096,000 routed rows even though this seeded
    # fixture executes 1,036,288.  Keep host launch capacity, device retirement,
    # and the compact Replay width on the same accounting contract.
    shared_tasks = 9792
    capacity_minibatches = 1000
    capacity_macrobatches = 32
    capacity_replay_minibatches = capacity_minibatches - saved
    max_completed_segments = capacity_macrobatches + len(tokens) - 1
    old_host_clusters = (
        shared_tasks + capacity_minibatches * bwd_tasks + capacity_replay_minibatches * old_replay_tasks + max_completed_segments * segment_tasks
    )
    paired_host_clusters = (
        shared_tasks + capacity_minibatches * bwd_tasks + capacity_replay_minibatches * paired_replay_tasks + max_completed_segments * segment_tasks
    )
    assert (old_host_clusters, paired_host_clusters) == (577560, 515608)
    assert (2 * old_host_clusters + 44, 2 * paired_host_clusters + 44) == (
        1155164,
        1031260,
    )

    old_device_clusters = shared_tasks + old_prefixes[-1]
    paired_device_clusters = shared_tasks + paired_prefixes[-1]
    assert (old_device_clusters, paired_device_clusters) == (158367, 144223)
    assert (2 * old_device_clusters + 44, 2 * paired_device_clusters + 44) == (
        316778,
        288490,
    )

    # the contract is pure Python; load it by path (no torch, no package import)
    comm_tile = load_cutedsl_module("persistent_bf16_bwd_contract").DEFAULT_COMM_TILE_SHAPE
    assert (comm_tile.rows, comm_tile.columns) == (32, 512)  # the default communication tile of the task stream
    replay_dispatch_arrivals = comm_tile.tiles(minibatch, 4096)  # one arrival per tile
    legacy_gate_up_ready = 2 * 2
    paired_gate_up_ready = 1 * 2 * 2
    assert replay_dispatch_arrivals == 1024
    assert legacy_gate_up_ready == paired_gate_up_ready == 4

    assert (old_prefixes[31], paired_prefixes[31]) == (13248, 13248)
    assert old_prefixes[32] == paired_prefixes[32] == 14016
    assert (old_prefixes[33], paired_prefixes[33]) == (14571, 14507)
    assert (old_prefixes[252], paired_prefixes[252]) == (147828, 133748)

    for idx in range(paired_prefixes[-1]):
        assert _bucket_decode(paired_prefixes, buckets, bucket_shift, idx) == _binary_task_owner(paired_prefixes, idx)


def test_qwen_100k_paired_replay_work_identity() -> None:
    expert_blocks = [64] * 32 + [63] * 16 + [62] * 16
    random.Random(0).shuffle(expert_blocks)
    tokens = [blocks * MLP_MB for blocks in expert_blocks]
    macrobatch = 131072
    minibatch = 4096
    completed = _completed_segment_offsets(tokens, macrobatch, minibatch)
    bwd_tasks, segment_tasks = _task_widths(minibatch, 4096, 1024)
    old_replay_tasks = _replay_task_width(minibatch, 1024)
    paired_replay_tasks = _replay_task_width(minibatch, 1024, paired_gate_up=True)
    saved = macrobatch // minibatch
    old_prefixes = _task_prefixes(completed, bwd_tasks, segment_tasks, saved, old_replay_tasks)
    paired_prefixes = _task_prefixes(completed, bwd_tasks, segment_tasks, saved, paired_replay_tasks)
    bucket_shift = bwd_tasks.bit_length() - 1
    buckets = _build_task_owner_buckets(paired_prefixes, 1 << bucket_shift)
    gate_up_tasks = (minibatch // MLP_MB) * (1024 // MLP_MB)

    expanded = Counter()
    paired_tasks = 0
    for physical_idx in range(paired_prefixes[-1]):
        owner, local = _bucket_decode(paired_prefixes, buckets, bucket_shift, physical_idx)
        if owner < saved:
            expanded[(owner, local)] += 1
        elif local < gate_up_tasks:
            expanded[(owner, local)] += 1
            expanded[(owner, gate_up_tasks + local)] += 1
            paired_tasks += 1
        else:
            expanded[(owner, local + gate_up_tasks)] += 1

    canonical = Counter((owner, local) for owner in range(len(old_prefixes) - 1) for local in range(old_prefixes[owner + 1] - old_prefixes[owner]))
    assert expanded == canonical
    assert paired_tasks == 221 * 64 == 14144
    assert sum(expanded.values()) == old_prefixes[-1] == 148575


def test_paired_replay_identity_guards_and_small_routing_cases() -> None:
    target = {
        "minibatch_release": True,
        "num_devices": 8,
        "use_mxfp8": False,
        "full_context": False,
        "num_macrobatches": 2,
    }
    assert _paired_replay_enabled(**target)
    for key, identity_value in (
        ("minibatch_release", False),
        ("num_devices", 4),
        ("use_mxfp8", True),
        ("full_context", True),
        ("num_macrobatches", 1),
    ):
        identity = dict(target)
        identity[key] = identity_value
        assert not _paired_replay_enabled(**identity)

    cases = [
        ([0, 512, 0, 768, 256], 512, 256, 1024, 512),
        ([768, 0, 256, 512], 512, 256, 2048, 1024),
        ([0, 768, 256, 0, 768], 1024, 512, 4096, 1024),
    ]
    for tokens, macrobatch, minibatch, hidden, intermediate in cases:
        assert _assert_task_owner_case(
            tokens,
            macrobatch,
            minibatch,
            hidden,
            intermediate,
            replay=True,
            paired_gate_up=True,
        ) == _completed_segment_offsets(tokens, macrobatch, minibatch)
