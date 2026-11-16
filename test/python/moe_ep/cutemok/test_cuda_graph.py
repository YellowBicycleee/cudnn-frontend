# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""CUDA graph capture and replay of build_schedule / forward / backward.

Graphs are captured after a warmup, either one graph for every layer's forward and backward (``joint``) or one
per layer and direction in one memory pool (``split``: forward graphs in layer order, backward graphs in
reverse, as per-layer CUDA graphs capture a model).  Each replay takes new inputs (activations, routing,
router weights, output gradient) through static tensors and must match an eager call on the same inputs bit
for bit: outputs, returned gradients and the main_grads the ``mcore`` layout accumulates into.

The backward without a kept forward context (``forward_context=None``, or ``recompute_forward_context``'s) runs
the same way: its shared FC1 launch and the backward variant that replays every macrobatch are captured and
replayed with per-expert weights.

Per-expert weights name their experts through TMA descriptors the megakernel builds on the first launch that
uses a set of storage tables (its ``build_descriptors`` argument).  A capture that follows an eager use of the
tables -- the warmup -- must record launches that do not build; only tables first used inside a capture are
built by the graph.  A capture before the warmup raises instead of compiling inside the recording.
"""

import contextlib
import sys

import pytest
import torch
import torch.distributed as dist

from cudnn.moe_ep.cutemok import functional
from cudnn.moe_ep.cutemok.cutedsl._arch import TARGET_CAPABILITY
from cudnn.moe_ep.cutemok.cutedsl.persistent_bf16_fwd import RoutedDescriptors
from .call_helpers import call_arguments, generate_inputs, shared_output_gate_weight

LOCAL_EXPERTS, HIDDEN, INTERMEDIATE, TOKENS, TOPK = 4, 1024, 512, 1024, 4
STEPS = 3


class _Model:
    """Layers of one MoE shape on one workspace, fed from static input tensors."""

    def __init__(self, context, precision, layout, *, gated, limits, forward_context="saved"):
        rank, world_size, device = context
        self.rank, self.world_size, self.device = rank, world_size, device
        self.precision, self.layout = precision, layout
        # What the backward is passed: the forward's context ("saved"), None ("none"), or
        # recompute_forward_context's ("recomputed").
        self.forward_context = forward_context
        self.config = functional.CuteMoKConfig(
            fwd_num_comm_sms=8, bwd_num_comm_sms=8, minibatch_size=1024, macrobatch_size=4096, schedule_capacity_multiplier=1.5
        )
        # A workspace of its own, so the backward buffer sets it keeps are this test's.
        self.workspace = functional.create_workspace(self.config, dist.group.WORLD, device=device, num_local_tokens=TOKENS, hidden_size=HIDDEN, topk=TOPK)
        self.layers = []
        for layer, limit in enumerate(limits):
            inputs = generate_inputs(
                rank,
                device,
                100 + layer,
                num_experts=LOCAL_EXPERTS * world_size,
                local_experts=LOCAL_EXPERTS,
                topk=TOPK,
                tokens=TOKENS,
                hidden=HIDDEN,
                intermediate=INTERMEDIATE,
            )
            gate = shared_output_gate_weight(rank, device, 100 + layer, HIDDEN) if gated else None
            self.layers.append(dict(source=(inputs[3:9], gate), limit=limit))
            self.make_views(self.layers[-1])
        self.static = dict(
            x=torch.empty(TOKENS, HIDDEN, dtype=torch.bfloat16, device=device),
            topk=torch.empty(TOKENS, TOPK, dtype=torch.int64, device=device),
            rw=torch.empty(TOKENS, TOPK, dtype=torch.float32, device=device),
            dout=torch.empty(TOKENS, HIDDEN, dtype=torch.bfloat16, device=device),
        )

    def make_views(self, layer):
        """The layer's call arguments: new weight views, storage tables and main_grads over its weights."""
        weights, gate = layer["source"]
        fw, bw, fk, bk, keepalive = call_arguments(self.layout, self.precision, weights[:3], weights[3:], output_gate=gate)
        main_grads = []
        if "main_grads" in bk:
            main_grads = [bk["main_grads"][i] for i in (0, 2, 4)] + [keepalive[-1]]
            if "shared_output_gate_main_grad" in bk:
                main_grads.append(bk["shared_output_gate_main_grad"])
        layer.update(fw=fw, bw=bw, fk=fk, bk=bk, keepalive=keepalive, main_grads=main_grads)

    def load(self, step):
        x, topk, rw, *_, dout = generate_inputs(
            self.rank,
            self.device,
            1000 + 31 * step,
            num_experts=LOCAL_EXPERTS * self.world_size,
            local_experts=LOCAL_EXPERTS,
            topk=TOPK,
            tokens=TOKENS,
            hidden=HIDDEN,
            intermediate=128,
        )
        for name, value in (("x", x), ("topk", topk), ("rw", rw), ("dout", dout)):
            self.static[name].copy_(value)
        for layer in self.layers:
            for grad in layer["main_grads"]:
                grad.zero_()

    def forward(self, layer):
        schedule = functional.build_schedule(self.workspace, self.config, self.static["topk"], num_local_experts=LOCAL_EXPERTS)
        output, forward_context = functional.forward(
            self.config, self.workspace, schedule, self.static["x"], self.static["rw"], *layer["fw"], layer["limit"], **layer["fk"]
        )
        return schedule, output, forward_context

    def backward(self, layer, schedule, forward_context):
        if self.forward_context == "none":
            forward_context = None
        elif self.forward_context == "recomputed":
            shared_gate, shared_up, _, routed_gate, routed_up, _ = layer["fw"]
            forward_context = functional.recompute_forward_context(
                self.config, self.workspace, schedule, self.static["x"], shared_gate, shared_up, routed_gate, routed_up, layer["limit"]
            )
        return functional.backward(
            self.config,
            self.workspace,
            schedule,
            forward_context,
            self.static["dout"],
            self.static["x"],
            self.static["rw"],
            *layer["bw"],
            layer["limit"],
            **layer["bk"],
        )

    def step(self):
        forwards = [self.forward(layer) for layer in self.layers]
        grads = [None] * len(self.layers)
        for index in reversed(range(len(self.layers))):
            grads[index] = self.backward(self.layers[index], forwards[index][0], forwards[index][2])
        return [(forward[1], grad) for forward, grad in zip(forwards, grads)]

    def snapshot(self, results):
        """Copies of each layer's output, returned gradients and main_grads."""
        torch.cuda.synchronize(self.device)
        return [
            [output.clone(), *(g.clone() for g in grads if g is not None), *(g.clone() for g in layer["main_grads"])]
            for (output, grads), layer in zip(results, self.layers)
        ]

    def capture(self, joint):
        """Warm up on a side stream, then capture: returns the graphs (in replay order) and their results."""
        side = torch.cuda.Stream(self.device)
        side.wait_stream(torch.cuda.current_stream(self.device))
        with torch.cuda.stream(side):
            for _ in range(2):
                self.load(0)
                self.step()
        torch.cuda.current_stream(self.device).wait_stream(side)
        torch.cuda.synchronize(self.device)
        dist.barrier()
        return self.capture_only(joint)

    def capture_only(self, joint):
        graphs, pool = [], torch.cuda.graph_pool_handle()
        if joint:
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph, pool=pool):
                results = self.step()
            graphs.append(graph)
        else:
            forwards = []
            for layer in self.layers:
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph, pool=pool):
                    forwards.append(self.forward(layer))
                graphs.append(graph)
            grads = [None] * len(self.layers)
            for index in reversed(range(len(self.layers))):
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph, pool=pool):
                    grads[index] = self.backward(self.layers[index], forwards[index][0], forwards[index][2])
                graphs.append(graph)
            results = [(forward[1], grad) for forward, grad in zip(forwards, grads)]
        torch.cuda.synchronize(self.device)
        dist.barrier()
        return graphs, results


def _assert_equal(label, reference, actual):
    for layer, (expected_tensors, actual_tensors) in enumerate(zip(reference, actual, strict=True)):
        for index, (expected, result) in enumerate(zip(expected_tensors, actual_tensors, strict=True)):
            if not torch.equal(expected, result):
                difference = (expected.float() - result.float()).abs().max().item()
                raise AssertionError(f"{label}: layer {layer} result {index} differs (max abs {difference:.3e})")


def _references(model):
    references = []
    for step in range(STEPS):
        model.load(step)
        references.append(model.snapshot(model.step()))
    model.load(0)
    _assert_equal("eager run-to-run (the bitwise comparisons below need it)", references[0], model.snapshot(model.step()))
    return references


def _replay_all(model, graphs, results, references, label):
    for step in list(range(STEPS)) + [0]:
        model.load(step)
        for graph in graphs:
            graph.replay()
        _assert_equal(f"{label}: replay of step {step}", references[step], model.snapshot(results))


def _skip_unless_supported(context):
    if torch.cuda.get_device_capability(context[2]) != TARGET_CAPABILITY:
        pytest.skip("CuTe DSL kernels")


@contextlib.contextmanager
def _descriptor_builds():
    """The ``build_descriptors`` argument of every megakernel launch made in the block, in launch order (the
    launches read it from ``RoutedDescriptors.build``; grouped weights always pass 0)."""
    builds, original = [], RoutedDescriptors.build

    def build(descriptors):
        builds.append(original.fget(descriptors))
        return builds[-1]

    RoutedDescriptors.build = property(build)
    try:
        yield builds
    finally:
        RoutedDescriptors.build = original


@pytest.mark.parametrize("precision,layout,joint", [("bf16", "mcore", False), ("mxfp8", "mcore", False), ("bf16", "plain", True), ("mxfp8", "plain", True)])
def test_replay_matches_eager(context, precision, layout, joint):
    """Replays with new inputs match eager calls, also after an eager call between replays."""
    _skip_unless_supported(context)
    model = _Model(context, precision, layout, gated=layout == "mcore", limits=(None, None))
    references = _references(model)
    with _descriptor_builds() as builds:
        graphs, results = model.capture(joint)
    # The eager steps above built the descriptors: neither the warmup nor the captured launches build again.
    assert builds and not any(builds), builds
    _replay_all(model, graphs, results, references, "before an eager step")
    model.load(1)
    _assert_equal("eager step between replays", references[1], model.snapshot(model.step()))
    _replay_all(model, graphs, results, references, "after an eager step")
    del graphs, results, model
    dist.barrier()


@pytest.mark.parametrize("precision", ["bf16", "mxfp8"])
def test_storage_tables_first_used_in_capture(context, precision):
    """Weight views and storage tables made after the warmup are first used inside the capture, which only
    records building their TMA descriptors: an eager call on them before any replay still builds them, and
    the replays match."""
    _skip_unless_supported(context)
    model = _Model(context, precision, "mcore", gated=True, limits=(None, None))
    references = _references(model)
    model.capture(joint=False)  # warms up (and compiles) on the first views; these graphs are dropped
    for layer in model.layers:
        model.make_views(layer)
    with _descriptor_builds() as builds:
        graphs, results = model.capture_only(joint=False)
    assert any(builds), "the captured launches on new tables must build their descriptors"
    model.load(1)
    _assert_equal("eager step on tables first used in the capture", references[1], model.snapshot(model.step()))
    _replay_all(model, graphs, results, references, "tables first used in the capture")
    del graphs, results, model
    dist.barrier()


@pytest.mark.parametrize("precision", ["bf16", "mxfp8"])
@pytest.mark.parametrize("forward_context", ["none", "recomputed"])
def test_backward_without_a_forward_context(context, precision, forward_context):
    """The backward of a layer that kept nothing of its forward, with per-expert weights and main_grads: the
    shared FC1 launch that recomputes the shared activations and the backward that replays every macrobatch are
    captured after a warmup (so no captured launch builds descriptors) and their replays match eager calls."""
    _skip_unless_supported(context)
    model = _Model(context, precision, "mcore", gated=False, limits=(None, None), forward_context=forward_context)
    references = _references(model)
    with _descriptor_builds() as builds:
        graphs, results = model.capture(joint=False)
    assert builds and not any(builds), builds
    _replay_all(model, graphs, results, references, f"forward context {forward_context}")
    model.load(1)
    _assert_equal("eager step between replays", references[1], model.snapshot(model.step()))
    _replay_all(model, graphs, results, references, f"forward context {forward_context}, after an eager step")
    del graphs, results, model
    dist.barrier()


def test_layers_with_different_backward_configurations(context):
    """Two layers whose BF16 backward differs (the SwiGLU clamp) on one workspace keep a buffer set each,
    so neither call reallocates the other's and each layer's graph keeps its own."""
    _skip_unless_supported(context)
    model = _Model(context, "bf16", "mcore", gated=False, limits=(None, 7.0))
    references = _references(model)
    assert len(model.workspace._cutedsl_bwd_preparations) == 2
    graphs, results = model.capture(joint=False)
    _replay_all(model, graphs, results, references, "layers with different configurations")
    assert len(model.workspace._cutedsl_bwd_preparations) == 2
    del graphs, results, model
    dist.barrier()


def test_capture_before_warmup_raises(context):
    """A public call captured before any eager call of its configuration raises instead of compiling inside the
    capture (the executor caches are emptied for the test and restored after it)."""
    _skip_unless_supported(context)
    rank, world_size, device = context
    config = functional.CuteMoKConfig(fwd_num_comm_sms=8, bwd_num_comm_sms=8, minibatch_size=1024, macrobatch_size=4096, schedule_capacity_multiplier=1.5)
    workspace = functional.create_workspace(config, dist.group.WORLD, device=device, num_local_tokens=TOKENS, hidden_size=HIDDEN, topk=TOPK)
    top_experts = torch.zeros(TOKENS, TOPK, dtype=torch.int32, device=device)
    caches = [
        cache
        for module in list(sys.modules.values())
        if module is not None and module.__name__.startswith("cudnn.moe_ep.cutemok.cutedsl.")
        for cache in (getattr(module, "_EXECUTOR_CACHE", None), getattr(module, "_EXECUTORS", None))
        if isinstance(cache, dict)
    ]
    saved = [dict(cache) for cache in caches]
    try:
        for cache in caches:
            cache.clear()
        with pytest.raises(RuntimeError, match="before CUDA graph capture"):
            with torch.cuda.graph(torch.cuda.CUDAGraph()):
                functional.build_schedule(workspace, config, top_experts, num_local_experts=LOCAL_EXPERTS)
    finally:
        for cache, entries in zip(caches, saved):
            cache.clear()
            cache.update(entries)
    del workspace
    dist.barrier()
