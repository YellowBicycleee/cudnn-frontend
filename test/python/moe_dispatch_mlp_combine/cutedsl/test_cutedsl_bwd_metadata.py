# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""The CuTe DSL BWD metadata builder writes exactly the tables of a plain full-scan reference.

One GPU (SM100 / SM103): ``python -m pytest -s moe_dispatch_mlp_combine/cutedsl/test_cutedsl_bwd_metadata.py``.  The builder visits, for
minibatch m >= 1, only the experts overlapping that minibatch's rows; the reference is a per-minibatch
scan of every expert (CUDA's ``build_minibatch_expert_offsets_kernel``
order, plus the CuTe DSL rule that an expert without rows is registered once, in minibatch 0, and the
task counts without SwiGLU tasks).  Every table slot is compared, including the capacity tails, on random
expert row counts with empty experts (first, last, runs of them), experts spanning several minibatches and
macrobatches, a partial last minibatch, minibatches holding more than 32 experts (several warp chunks),
one macrobatch (no expert / owner tables) and several (the O(1)-decode tables).  ``recompute_forward_context`` (the
megakernel replays the first macrobatch too): the expert / owner tables at any capacity, every minibatch with replay
tasks.
"""

import random

import pytest
import torch
from cudnn.moe_dispatch_mlp_combine.cutedsl._arch import SUPPORTED_CAPABILITIES

SENTINEL = -7


def _reference(counts, capacity, macro, mini, hidden, intermediate, k, segment_capacity, bucket_capacity, recompute_forward_context=False):
    nrows = sum(counts)
    n = -(-nrows // mini)
    o1 = recompute_forward_context or capacity > macro
    pairs, completed = [0] * (k + 1), [0] * (k + 1)
    experts = [SENTINEL] * segment_capacity
    buckets = [SENTINEL] * bucket_capacity
    pair_count = segment_count = 0
    for m in range(n):
        begin, end = m * mini, min(m * mini + mini, nrows)
        macro_begin = begin // macro * macro
        macro_end = min(macro_begin + macro, nrows)
        expert_begin = 0
        for e, rows in enumerate(counts):
            expert_end = expert_begin + rows
            pair_count += max(expert_begin, begin) < min(expert_end, end)
            segment_begin, segment_end = max(expert_begin, macro_begin), min(expert_end, macro_end)
            if (segment_begin < segment_end and begin < segment_end <= end) or (segment_begin >= segment_end and rows == 0 and m == 0):
                if o1:
                    experts[segment_count] = e
                segment_count += 1
            expert_begin = expert_end
        pairs[m + 1], completed[m + 1] = pair_count, segment_count
    for m in range(n + 1, k + 1):
        pairs[m], completed[m] = pair_count, segment_count
    if o1:
        q, i, h = mini // 256, intermediate // 256, hidden // 256
        down, replay, wgrad = q * i + q * h, q * i, 3 * i * h
        width = 1
        while width * 2 <= down:
            width *= 2
        saved = 0 if recompute_forward_context else min(n, macro // mini)
        prefix = lambda owner: owner * down + max(0, owner - saved) * replay + completed[owner] * wgrad
        count = -(-prefix(n) // width)
        owner = 0
        for bucket in range(count):
            while owner + 1 < n and prefix(owner + 1) <= bucket * width:
                owner += 1
            buckets[bucket] = owner
        for bucket in range(count, bucket_capacity):
            buckets[bucket] = max(0, n - 1)
        for segment in range(segment_count, segment_capacity):
            experts[segment] = -1
    return pairs, completed, experts, buckets


def _counts(rng, experts, rows):
    """Random multiples of 256 summing to rows, with empty experts at the ends and in runs."""
    blocks = rows // 256
    weights = [0 if rng.random() < 0.3 else rng.random() for _ in range(experts)]
    weights[0] = weights[-1] = 0.0
    if not any(weights):
        weights[experts // 2] = 1.0
    total = sum(weights)
    counts = [int(blocks * w / total) for w in weights]
    counts[max(range(experts), key=lambda e: weights[e])] += blocks - sum(counts)
    return [c * 256 for c in counts]


# (local experts, rows, capacity, macrobatch, minibatch, hidden, intermediate)
CASES = (
    (8, 16384, 16384, 16384, 4096, 2048, 1024),  # one macrobatch, few experts spanning minibatches
    (64, 65536, 65536, 65536, 4096, 4096, 1024),  # one macrobatch (the bench default)
    (512, 131072, 131072, 131072, 2048, 2048, 1280),  # one macrobatch, 512 experts
    (64, 61440, 65536, 16384, 4096, 4096, 1024),  # several macrobatches, partial last macrobatch
    (64, 62464, 65536, 16384, 4096, 4096, 1024),  # several macrobatches, partial last minibatch (1024 rows)
    (128, 32768, 65536, 32768, 16384, 2048, 1024),  # ~64 experts per minibatch: several warp chunks per minibatch
    (384, 98304, 131072, 32768, 4096, 7168, 3072),  # several macrobatches, 384 experts
    (16, 49152, 65536, 8192, 2048, 2048, 1024),  # experts span several macrobatches
)


@pytest.mark.skipif(not torch.cuda.is_available() or torch.cuda.get_device_capability() not in SUPPORTED_CAPABILITIES, reason="SM100/SM103 GPU required")
@pytest.mark.parametrize("recompute_forward_context", (False, True), ids=("saved", "recompute_forward_context"))
@pytest.mark.parametrize("seed", range(3))
@pytest.mark.parametrize("case", CASES, ids=lambda c: f"E{c[0]}-rows{c[1]}-macro{c[3]}-mini{c[4]}")
def test_metadata_tables_match_full_scan(case, seed, recompute_forward_context):
    from types import SimpleNamespace

    from cudnn.moe_dispatch_mlp_combine.cutedsl._host_common import current_stream_handle
    from cudnn.moe_dispatch_mlp_combine.cutedsl._persistent_bf16_bwd_metadata import prepare_bwd_metadata_launch

    experts, rows, capacity, macro, mini, hidden, intermediate = case
    counts = _counts(random.Random(seed * 1000 + experts), experts, rows)
    k = capacity // mini
    segment_capacity = experts * (capacity // macro + 1) + 1
    bucket_capacity = 4096
    i32 = dict(dtype=torch.int32, device=torch.device("cuda"))
    inputs = SimpleNamespace(
        schedule_peer_rank=torch.zeros(capacity, **i32),
        schedule_peer_token_idx=torch.zeros(capacity, **i32),
        num_tokens=torch.tensor([sum(counts)], **i32),
        tokens_per_expert=torch.tensor(counts, **i32),
    )
    tables = tuple(torch.full((size,), SENTINEL, **i32) for size in (k + 1, k + 1, segment_capacity, bucket_capacity))
    config = SimpleNamespace(
        macrobatch_size=macro,
        minibatch_size=mini,
        num_local_tokens=1,
        topk=1,
        routed_hidden_size=hidden,
        intermediate_size=intermediate,
        recompute_forward_context=recompute_forward_context,
    )
    stream = current_stream_handle(0)
    executor, args = prepare_bwd_metadata_launch(inputs, SimpleNamespace(metadata=tables), config=config, stream=stream)
    executor(*args, stream)
    torch.cuda.synchronize()
    expected = _reference(counts, capacity, macro, mini, hidden, intermediate, k, segment_capacity, bucket_capacity, recompute_forward_context)
    for name, got, want in zip(("pairs", "completed", "experts", "buckets"), tables, expected):
        assert got.tolist() == want, f"{name} differs"
