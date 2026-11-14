# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Host launch path of the CuTe DSL operators.

CPU: the launch-plan cache (``_host_common.LaunchPlanCache``) reuses a plan only for the same live
source tensors with unchanged address, shape, strides and dtype, and typed keys keep ``True``, ``1``
and ``1.0`` (also inside lists) apart; an executor miss during a CUDA graph capture raises.  GPU: the scheduler's call-signature cache validates a signature once and
in full on every new one."""

from __future__ import annotations

import re
import types
import unittest
import unittest.mock

import torch

from ._cutedsl_by_path import CUTEDSL_DIR, load_cutedsl_module


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


class HostEnvironment(unittest.TestCase):
    def test_kernel_modules_neither_read_nor_set_the_environment(self):
        # The target and TVM FFI are compile options (_arch.COMPILE_OPTIONS): no kernel or host module sets a
        # CuTe DSL variable or requires one before CUTLASS is imported.  _arch reads CUTE_DSL_ARCH only to decline
        # a value that names another part; _host_common reads only the optional executor cache directory.  The
        # backward host reads no test-only switch either (tests patch the host module).
        allowed = {"_arch.py": {"ARCH_ENV"}, "_host_common.py": {"EXECUTOR_CACHE_DIR_ENV"}}
        for path in sorted(CUTEDSL_DIR.glob("*.py")):
            text = path.read_text()
            reads = set(re.findall(r"os\.environ(?:\.get)?\s*[\[(]\s*([^\s,\)\]]+)", text))
            self.assertEqual(reads, allowed.get(path.name, set()), path.name)
            self.assertNotRegex(text, r"os\.environ(\[[^]]*\]\s*=[^=]|\.(setdefault|update|pop)\()", path.name)
        self.assertNotIn("MOK_", (CUTEDSL_DIR / "persistent_bf16_bwd.py").read_text())


class FwdEpilogueCombineVariant(unittest.TestCase):
    """CuteMoKConfig.epilogue_combine is a compiled variant of the BF16 forward mega: the two combine placements
    never share an executor.  (The launch-plan key is built inside ``forward_bf16``; that a call with the option does
    not reuse the default's plan is checked on the GPU, test_cutedsl_functional.py::_assert_variants_ran.)"""

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
    the GPU, test_cutedsl_comm_tile_shape.py.)"""

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
