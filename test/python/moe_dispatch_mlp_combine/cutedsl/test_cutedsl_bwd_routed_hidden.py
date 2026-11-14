# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CPU checks of the backward contract with separate shared and routed hiddens (no GPU, no torch, no
CuTe DSL import: the contract modules use the standard library only and are loaded by path).

The shared chain (shared Gate/Up dgrad, shared Wgrads, d_x_shared, shared weights) sizes from hidden_size,
the routed chain (rings, communication readiness, routed tasks and Wgrads, routed weights) from
routed_hidden_size; equal hiddens give exactly the ordinary geometry.
"""

import unittest
from dataclasses import replace

from ._cutedsl_by_path import load_cutedsl_module

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
