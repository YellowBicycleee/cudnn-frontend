# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Split routed weights (MCore's per-expert allocations behind address tables): the host contract of the
tables and the forward and backward against the grouped layout, bit for bit.

Run under torchrun on SM100 / SM103.
"""

from __future__ import annotations

import pytest
import torch
import torch.distributed as dist

from cudnn.moe_dispatch_mlp_combine import functional as f
from cudnn.moe_dispatch_mlp_combine import functional, ops
from cudnn.moe_dispatch_mlp_combine.cutedsl._arch import SUPPORTED_CAPABILITIES
from cudnn.moe_dispatch_mlp_combine.functional import get_workspace

from .utils import BF16_TOLERANCE, capacity_multiplier, check_correctness, generate_inputs, run_reference_bf16, ungated

# --------------------------------------------------------------------------------------------------------------------
# Storage tables
#
# The host contract for MCore's split routed weights.
#
# MCore holds the routed experts as independent per-expert allocations whenever the grouped
# single-parameter layout is unavailable -- which is the common case, since the LayerWise
# distributed optimizer refuses ``moe_single_grouped_weight`` and the DeepSeek recipes run Muon.
# It hands MOK a ``SplitRoutedWeight``: one representative expert plus a table of base addresses.
#
# What is checked here is the part that can be wrong silently. The table is only usable if every
# expert agrees on dtype, shape, stride and device, because the kernel keeps one TMA descriptor and
# only retargets its address; a table built from mismatched allocations would read the right address
# with the wrong schema. The validators therefore have to reject that, and ``_routed_weight_payload``
# has to hand the rest of the stack exactly the shape the non-split forms do, so that every
# downstream check stays the one it already was.
#
# GPU only (the tables hold device addresses):
# ``python -m pytest moe_dispatch_mlp_combine/cutedsl/test_split_weights.py``


EXPERTS, INTERMEDIATE, HIDDEN = 4, 256, 512


pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="storage tables hold device addresses")


def _experts(dtype=torch.bfloat16, shape=(INTERMEDIATE, HIDDEN), count=EXPERTS):
    return [torch.zeros(*shape, dtype=dtype, device="cuda") for _ in range(count)]


def _scales(count=EXPERTS):
    return [torch.zeros(INTERMEDIATE // 128, HIDDEN // 128, 32, 16, dtype=torch.uint8, device="cuda") for _ in range(count)]


def test_table_holds_each_expert_address():
    tensors = _experts()
    table = ops.make_routed_weight_storage_table_bf16(tensors)
    assert table.dtype is torch.int64 and table.device == tensors[0].device
    assert table.tolist() == [tensor.data_ptr() for tensor in tensors]
    # Distinct allocations: a table that collapsed them would read one expert everywhere.
    assert len(set(table.tolist())) == len(tensors)


def test_table_rejects_mismatched_experts():
    """The whole point of the table is that one descriptor schema fits every entry."""
    good = _experts()
    for label, bad in (
        ("dtype", torch.zeros(INTERMEDIATE, HIDDEN, dtype=torch.float16, device="cuda")),
        ("shape", torch.zeros(INTERMEDIATE, HIDDEN // 2, dtype=torch.bfloat16, device="cuda")),
    ):
        with pytest.raises((ValueError, TypeError)):
            ops.make_routed_weight_storage_table_bf16([*good[:-1], bad])
    with pytest.raises(ValueError, match="contiguous"):
        ops.make_routed_weight_storage_table_bf16([*good[:-1], torch.zeros(INTERMEDIATE, HIDDEN * 2, dtype=torch.bfloat16, device="cuda")[:, ::2]])
    with pytest.raises(TypeError):
        ops.make_routed_weight_storage_table_bf16([])


def test_scale_table_demands_the_tcgen05_shape():
    ops.make_routed_scale_storage_table(_scales())
    for bad in (torch.zeros(4, 4, 32, 8, dtype=torch.uint8, device="cuda"), torch.zeros(4, 4, 32, 16, dtype=torch.int8, device="cuda")):
        with pytest.raises((ValueError, TypeError)):
            ops.make_routed_scale_storage_table([bad])


def test_mxfp8_and_main_grad_tables_check_their_dtypes():
    ops.make_routed_weight_storage_table_mxfp8(_experts(dtype=torch.float8_e4m3fn))
    with pytest.raises(TypeError):
        ops.make_routed_weight_storage_table_mxfp8(_experts(dtype=torch.bfloat16))
    for dtype in (torch.float32, torch.bfloat16):
        ops.make_routed_d_weight_storage_table(_experts(dtype=dtype))
    with pytest.raises(TypeError):
        ops.make_routed_d_weight_storage_table(_experts(dtype=torch.float8_e4m3fn))


def _split_weight(dtype=torch.bfloat16, **extra):
    tensors = _experts(dtype=dtype)
    table = (ops.make_routed_weight_storage_table_mxfp8 if dtype is torch.float8_e4m3fn else ops.make_routed_weight_storage_table_bf16)(tensors)
    return f.SplitRoutedWeight(data=tensors[0], storage_table=table, **extra)


def test_payload_is_the_shape_the_rest_of_the_stack_expects():
    """BF16 gives a tensor, MXFP8 a pair, and the backward's MXFP8 the five-tuple with its flag."""
    bf16 = _split_weight()
    assert f._routed_weight_payload(bf16) is bf16.data

    scales = _scales()
    forward = _split_weight(
        dtype=torch.float8_e4m3fn, scale=scales[0], scale_storage_table=ops.make_routed_scale_storage_table(scales), scale_tensors=tuple(scales)
    )
    assert f._routed_weight_payload(forward) == (forward.data, forward.scale)

    transposed = _experts(dtype=torch.float8_e4m3fn)
    backward = _split_weight(
        dtype=torch.float8_e4m3fn,
        scale=scales[0],
        scale_storage_table=ops.make_routed_scale_storage_table(scales),
        scale_tensors=tuple(scales),
        transposed_data=transposed[0],
        transposed_scale=scales[0],
        native_columnwise=True,
    )
    assert f._routed_weight_payload(backward) == (backward.data, backward.scale, backward.transposed_data, backward.transposed_scale, True)

    # Anything that is not a SplitRoutedWeight travels untouched.
    plain = torch.zeros(EXPERTS, INTERMEDIATE, HIDDEN, dtype=torch.bfloat16, device="cuda")
    assert f._routed_weight_payload(plain) is plain


def test_splitting_is_all_or_nothing():
    """One descriptor per operand means one addressing scheme; a mix has no meaning."""
    split = _split_weight()
    plain = torch.zeros(EXPERTS, INTERMEDIATE, HIDDEN, dtype=torch.bfloat16, device="cuda")
    assert f._routed_storage_tables(plain, plain, plain) is None
    tables = f._routed_storage_tables(split, split, split)
    assert tables.gate is split.storage_table and tables.transposed is None
    with pytest.raises(ValueError, match="all be SplitRoutedWeight"):
        f._routed_storage_tables(split, plain, split)


# --------------------------------------------------------------------------------------------------------------------
# Split against grouped
#
# Split routed weights must read exactly what the grouped layout reads.
#
# MCore owns the routed experts as independent per-expert allocations whenever the grouped
# single-parameter layout is unavailable -- the usual case, since the LayerWise distributed
# optimizer refuses ``moe_single_grouped_weight`` and the DeepSeek recipes run Muon.  MOK is then
# handed one representative expert plus a table of base addresses, and the kernel retargets a TMA
# descriptor per expert instead of slicing an ``[E, N, K]`` parameter's L mode.
#
# The acceptance criterion is equality, not a tolerance.  Both arms are given *the same memory*:
# the split arm's table is built from the per-expert slices of the very tensor the grouped arm
# passes, so the arithmetic is identical by construction and the only thing that differs is how the
# kernel finds each expert.  Anything short of bit-identical means it read the wrong bytes.
#
# Three distinct failures are covered, and they are the ones that are otherwise silent:
#
# * a descriptor that kept the atom's own base address -- every expert would read expert 0, which
#   ``test_split_matches_grouped`` catches as soon as two experts hold different weights;
# * a descriptor array indexed by something other than the expert -- ``test_split_follows_the_table``
#   reverses the table and requires the result to follow it, so reading *some* other expert is not
#   enough to pass;
# * a combined FC1 whose Up half is addressed from the Gate half's base --
#   ``test_split_combined_fc1_matches_grouped``, where one ``[2I, H]`` parameter per expert serves
#   both and the Up rows start partway into every allocation.
#
# Each test first replays the grouped arm and requires the two runs to agree, so a failure says
# which of the two properties broke rather than leaving the comparison ambiguous.
#
# Run under torchrun on SM100 / SM103, e.g.
# ``torchrun --standalone --nproc-per-node=8 -m pytest -s moe_dispatch_mlp_combine/cutedsl/test_split_weights.py``.
# EP1 works too and is the quick way to run it on a single GPU.


SUPPORTED_WORLD_SIZES = (1, 4, 8, 16, 32, 64)


#: Small, and four local experts: the point of the table is that experts differ, so one of them
#: would make every test here vacuous.  Modelled on utils.shapes' "All minimums" otherwise.
LOCAL_EXPERTS, LAYER_HIDDEN, INTERMEDIATE, TOPK, LOCAL_TOKENS = 4, 256, 256, 4, 512


def _setup(context, macrobatch_size=None):
    """``macrobatch_size``: small enough for several macrobatches, so the backward's replay and
    its several Wgrad contributions per tile run too; the default is one macrobatch."""
    rank, world_size, device = context
    if world_size not in SUPPORTED_WORLD_SIZES:
        pytest.skip(f"a workspace supports EP sizes {SUPPORTED_WORLD_SIZES}")
    if torch.cuda.get_device_capability(device) not in SUPPORTED_CAPABILITIES:
        pytest.skip("CuTe DSL backends are SM100/SM103 only")

    num_experts = LOCAL_EXPERTS * world_size
    geometry = {} if macrobatch_size is None else dict(macrobatch_size=macrobatch_size, minibatch_size=256)
    config = functional.CuteMoKConfig(
        schedule_capacity_multiplier=capacity_multiplier(world_size, LOCAL_TOKENS, TOPK, LOCAL_EXPERTS),
        **geometry,
    )
    workspace = get_workspace(config, dist.group.WORLD, device=device, num_local_tokens=LOCAL_TOKENS, hidden_size=LAYER_HIDDEN, topk=TOPK)
    inputs = generate_inputs(rank, device, num_experts, LOCAL_EXPERTS, TOPK, LOCAL_TOKENS, LAYER_HIDDEN, INTERMEDIATE)
    schedule = functional.build_schedule(workspace, config, inputs[1], num_local_experts=LOCAL_EXPERTS)
    return config, workspace, schedule, inputs


def _split(experts):
    """A ``SplitRoutedWeight`` over ``experts``, in that order.

    The experts are slices of a grouped tensor, so the split and grouped arms read identical
    bytes: the addressing is what is under test, not the arithmetic.  They are nonetheless
    genuinely different addresses, which is what a no-op retarget gets wrong.
    """
    return functional.SplitRoutedWeight(data=experts[0], storage_table=ops.make_routed_weight_storage_table_bf16(experts))


def _run(config, workspace, schedule, inputs, routed):
    """One forward, reduced to the results that are defined everywhere.

    The forward context's routed buffers (``x_routed``, ``gate_routed``, ``up_routed``,
    ``hidden_routed``) are macrobatch-sized scratch -- 131072 rows here for ~2k dispatched ones --
    allocated with ``torch.empty`` and written only where there are tokens.  Comparing them whole
    compares uninitialized memory, which agrees between two runs exactly as often as the caching
    allocator hands back the same block, so it passes until it does not.  (It cost a debugging
    round here: "the grouped forward disagreed with its own replay", 16M of 33M elements, max
    delta 2.4e-38 -- denormal noise, i.e. never-written bytes.)  ``test_backward.py`` keeps
    the same four results for the same reason.

    Nothing is lost: ``output`` is the combine of every routed expert, so an expert read from the
    wrong address cannot fail to show up in it.
    """
    x, _, router_weights, shared_gate, shared_up, shared_down = inputs[:6]
    output, context = functional.forward(config, workspace, schedule, x, router_weights, shared_gate, shared_up, shared_down, *routed, None)
    return output, context.gate_shared, context.up_shared, context.hidden_shared


NAMES = ("output", "gate_shared", "up_shared", "hidden_shared")


def _assert_identical(expected, actual, what: str) -> None:
    for name, left, right in zip(NAMES, expected, actual):
        if torch.equal(left, right):
            continue
        differing = (left != right).sum().item()
        raise AssertionError(
            f"{what}: {name} differs in {differing} of {left.numel()} elements "
            f"(max |delta| {(left.float() - right.float()).abs().max().item():g}); the two arms "
            "read the same memory, so this is an addressing error, not a numerical one"
        )


def _grouped_twice(config, workspace, schedule, inputs, routed):
    """The grouped arm, run twice and checked against itself.

    A forward that is not deterministic would make every comparison below meaningless, so it is
    ruled out first and reported as itself.
    """
    first = _run(config, workspace, schedule, inputs, routed)
    _assert_identical(first, _run(config, workspace, schedule, inputs, routed), "the grouped forward disagreed with its own replay")
    return first


def _run_backward(config, workspace, schedule, inputs, routed, main_grads, main_grad_storage_tables=None):
    """A forward and the backward that consumes its context, as MCore calls them: the combined
    FC1 under both names and the weight gradients accumulated into ``main_grads``.  Returns d_x
    and d_router_weights; the main_grads are the caller's."""
    x, _, router_weights, shared_gate, shared_up, shared_down = inputs[:6]
    d_output = inputs[9]
    _, forward_context = functional.forward(config, workspace, schedule, x, router_weights, shared_gate, shared_up, shared_down, *routed, None)
    gradients = ungated(
        functional.backward(
            config,
            workspace,
            schedule,
            forward_context,
            d_output,
            x,
            router_weights,
            shared_gate,
            shared_up,
            shared_down,
            *routed,
            None,
            main_grads=main_grads,
            main_grad_storage_tables=main_grad_storage_tables,
        )
    )
    return gradients[0], gradients[1]


def _seeded(shape, generator, device):
    """A main_grad that already holds earlier microbatches: an epilogue that stored instead of
    adding, or wrote the wrong expert, cannot pass by writing into zeros."""
    return torch.rand(*shape, generator=generator, device=device) - 0.5


def _backward_both_layouts(context, order, combined_fc1=True):
    """MCore's two layouts of the same weights and main_grads, through the backward.

    Grouped: one combined ``[E, 2I, H]`` FC1 and ``[E, H, I]`` Down parameter, with main_grads of
    the same shapes.  Split: every expert's FC1, Down and their main_grads in an allocation of its
    own, handed over in ``order`` -- so the split arm's expert ``e`` is the grouped arm's expert
    ``order[e]``.  Both start from the same seeded main_grads.  Several macrobatches, so the
    replay reads the weights K-major through the descriptors as well as the dgrad's MN-major view,
    and each Wgrad tile takes several ordered contributions.

    ``combined_fc1=False``: Gate and Up are separate weights in both layouts -- two ``[E, I, H]`` parameters, or
    two allocations and two address tables per expert -- so the Up weights start at row 0 of their own
    allocations.  The main_grads stay the combined ``[2I, H]`` buffers (split main_grads are always combined).

    Returns, per layout, (d_x, d_router_weights, shared main_grads, FC1 main_grad by expert,
    Down main_grad by expert), the split arm's put back in the grouped arm's expert order.
    """
    config, workspace, schedule, inputs = _setup(context, macrobatch_size=512)
    gate, up, down = inputs[6:9]
    device = gate.device
    combined = torch.cat((gate, up), dim=1).contiguous()  # [E, 2I, H]
    generator = torch.Generator(device=device)
    generator.manual_seed(1234 + dist.get_rank())
    shared_seed = tuple(_seeded(weight.shape, generator, device) for weight in inputs[3:6])
    fc1_seed = _seeded(combined.shape, generator, device)
    down_seed = _seeded(down.shape, generator, device)

    def shared_main_grads():
        fc1 = torch.cat(shared_seed[:2]).contiguous()  # MCore's shared FC1 main_grad, narrowed
        intermediate = shared_seed[0].shape[0]
        return fc1, fc1[:intermediate], fc1[intermediate:], shared_seed[2].clone()

    # Grouped.  The expert order is the split arm's, so the two arms route identically.
    grouped_fc1 = fc1_seed[list(order)].clone()
    grouped_down = down_seed[list(order)].clone()
    shared_fc1, shared_gate, shared_up, shared_down = shared_main_grads()
    grouped_weights = (combined[list(order)].contiguous(),) * 2 + (down[list(order)].contiguous(),)
    if not combined_fc1:
        grouped_weights = tuple(weight[list(order)].contiguous() for weight in (gate, up, down))
    grouped = _run_backward(config, workspace, schedule, inputs, grouped_weights, (shared_gate, grouped_fc1, shared_up, grouped_fc1, shared_down, grouped_down))
    grouped_result = (*grouped, (shared_fc1, shared_down), list(grouped_fc1), list(grouped_down))

    # Split: each expert its own allocation -- weights and main_grads alike, as MCore's are --
    # allocated in the natural order and handed over in ``order``, so the table is not monotonic
    # in address and no stride from one expert to the next exists to be assumed.
    def own_allocations(tensor):
        allocations = [tensor[expert].clone() for expert in range(LOCAL_EXPERTS)]
        return [allocations[expert] for expert in order]

    fc1_experts, down_experts = own_allocations(combined), own_allocations(down)
    fc1_split = _split(fc1_experts)
    down_split = _split(down_experts)
    split_weights = (fc1_split, fc1_split, down_split)
    if not combined_fc1:
        # Held here until both calls have returned: a table names its experts by address only.
        gate_experts, up_experts = own_allocations(gate), own_allocations(up)
        split_weights = (_split(gate_experts), _split(up_experts), down_split)
    fc1_grads, down_grads = own_allocations(fc1_seed), own_allocations(down_seed)
    fc1_table = ops.make_routed_d_weight_storage_table(fc1_grads)
    tables = (fc1_table, fc1_table, ops.make_routed_d_weight_storage_table(down_grads))
    shared_fc1, shared_gate, shared_up, shared_down = shared_main_grads()
    split = _run_backward(
        config,
        workspace,
        schedule,
        inputs,
        split_weights,
        (shared_gate, fc1_grads[0], shared_up, fc1_grads[0], shared_down, down_grads[0]),
        main_grad_storage_tables=tables,
    )
    split_result = (*split, (shared_fc1, shared_down), fc1_grads, down_grads)
    return grouped_result, split_result


def _assert_same_backward(grouped, split, what: str) -> None:
    labels = ("d_x", "d_router_weights", "shared main_grads", "routed FC1 main_grad", "routed Down main_grad")
    for label, left, right in zip(labels, grouped, split, strict=True):
        lefts = left if isinstance(left, (list, tuple)) else (left,)
        rights = right if isinstance(right, (list, tuple)) else (right,)
        for index, (a, b) in enumerate(zip(lefts, rights, strict=True)):
            if torch.equal(a, b):
                continue
            raise AssertionError(
                f"{what}: {label}[{index}] differs in {(a != b).sum().item()} of {a.numel()} "
                "elements; the two arms read the same values, so this is an addressing error"
            )


def test_split_backward_matches_grouped(context):
    """MCore's split layout gives the grouped layout's results bit for bit.

    The backward reads every routed weight twice -- MN-major for the swap-AB dgrad, K-major for
    the replay -- through two descriptor schemas over one set of addresses, finds the combined
    FC1's Up half by a tile coordinate in both, and reduce-adds into per-expert main_grads by
    descriptor.  Any one of those wrong leaves the others right, which a tolerance would wave
    through; equality does not.
    """
    grouped, split = _backward_both_layouts(context, order=range(LOCAL_EXPERTS))
    _assert_same_backward(grouped, split, "split routed weights and main_grads")


def test_split_backward_follows_the_table(context):
    """Hand the experts over in a scrambled order; every view must follow the table."""
    order = tuple((3 * expert + 1) % LOCAL_EXPERTS for expert in range(LOCAL_EXPERTS))
    assert sorted(order) == list(range(LOCAL_EXPERTS)) and order != tuple(range(LOCAL_EXPERTS))
    grouped, split = _backward_both_layouts(context, order=order)
    _assert_same_backward(grouped, split, "a scrambled expert table in the backward")


def test_split_backward_separate_gate_and_up_matches_grouped(context):
    """Gate and Up as separate per-expert allocations, each with its own address table (no Up row offset), in
    the natural and in a scrambled order: the backward gives the grouped separate weights' results bit for bit.
    Each arm's d_x and d_router_weights are first held against the PyTorch reference, so a difference between
    the arms names the one that is off."""
    rank = context[0]
    scrambled = tuple((3 * expert + 1) % LOCAL_EXPERTS for expert in range(LOCAL_EXPERTS))
    for order in (tuple(range(LOCAL_EXPERTS)), scrambled):
        grouped, split = _backward_both_layouts(context, order=order, combined_fc1=False)
        inputs = _setup(context, macrobatch_size=512)[3]
        weights = (weight[list(order)].contiguous() for weight in inputs[6:9])
        reference = run_reference_bf16(*inputs[:6], *weights, inputs[9], None)[1:3]
        what = f"separate split Gate and Up weights in the backward, expert order {order}"
        for arm, result in (("grouped", grouped), ("split", split)):
            for name, want, got in zip(("d_x", "d_router_weights"), reference, result, strict=False):
                check_correctness(f"{what}: {arm} {name} vs reference", want, got, BF16_TOLERANCE, print_stats=rank == 0)
        if rank == 0:
            print(f"{what}: max |grouped - split| of d_x " f"{(grouped[0].float() - split[0].float()).abs().max().item():g}")
        _assert_same_backward(grouped, split, what)


def test_split_reading_one_expert_everywhere(context):
    """The split path, with every descriptor aimed at the same allocation.

    Deliberately first in the file, and deliberately a degenerate table: it separates "the
    retargeted-descriptor path works at all" from "each expert gets its own address".  The
    reference is a grouped run whose experts are all copies of expert 0, so the two arms still
    have to agree bit for bit -- but if this one fails while the layouts are identical, nothing
    about per-expert addressing is implicated and the fault is in the descriptor machinery.
    """
    config, workspace, schedule, inputs = _setup(context)
    grouped = inputs[6:9]
    repeated = tuple(weight[0].unsqueeze(0).expand_as(weight).contiguous() for weight in grouped)
    expected = _grouped_twice(config, workspace, schedule, inputs, repeated)
    split = tuple(_split([weight[0]] * LOCAL_EXPERTS) for weight in grouped)
    _assert_identical(expected, _run(config, workspace, schedule, inputs, split), "one expert behind every descriptor")


def test_split_matches_grouped(context):
    config, workspace, schedule, inputs = _setup(context)
    grouped = inputs[6:9]
    expected = _grouped_twice(config, workspace, schedule, inputs, grouped)
    split = tuple(_split([weight[index] for index in range(LOCAL_EXPERTS)]) for weight in grouped)
    _assert_identical(expected, _run(config, workspace, schedule, inputs, split), "split routed weights")


def test_split_descriptors_are_built_once(context):
    """The per-expert TMA descriptors are built by the first launch over a set of address tables
    and reused by every later one -- the C++ port builds them once, with its table, and MCore keeps
    its tables -- while a new table object is a new set, built by its own first launch, and a
    freed table takes its descriptors with it.  Reused descriptors must still read every expert
    from its own allocation: each run is checked against the grouped layout."""
    import gc

    from cudnn.moe_dispatch_mlp_combine.cutedsl import persistent_bf16_fwd as fwd_host

    config, workspace, schedule, inputs = _setup(context)
    grouped = inputs[6:9]
    expected = _grouped_twice(config, workspace, schedule, inputs, grouped)
    before = set(fwd_host._ROUTED_DESCRIPTORS)

    split = tuple(_split([weight[index] for index in range(LOCAL_EXPERTS)]) for weight in grouped)
    _assert_identical(expected, _run(config, workspace, schedule, inputs, split), "first split launch")
    first_keys = set(fwd_host._ROUTED_DESCRIPTORS) - before
    assert len(first_keys) == 1, first_keys
    entry = fwd_host._ROUTED_DESCRIPTORS[next(iter(first_keys))]
    assert entry.built and entry.build == 0, "the first launch did not mark its descriptors built"
    _assert_identical(expected, _run(config, workspace, schedule, inputs, split), "a launch reusing the built descriptors")
    assert set(fwd_host._ROUTED_DESCRIPTORS) - before == first_keys, "the second launch made a new set"

    # The same addresses in new table objects: a new set, built by its own first launch.
    rebuilt = tuple(_split([weight[index] for index in range(LOCAL_EXPERTS)]) for weight in grouped)
    _assert_identical(expected, _run(config, workspace, schedule, inputs, rebuilt), "new tables")
    both = set(fwd_host._ROUTED_DESCRIPTORS) - before
    assert len(both) == 2 and first_keys < both, both

    # Freeing the first tables drops their set; the second stays.
    del split
    gc.collect()
    assert set(fwd_host._ROUTED_DESCRIPTORS) - before == both - first_keys
    dist.barrier()


def test_split_follows_the_table(context):
    """Reverse the expert order in the table; the result must follow it.

    Reading expert 0 everywhere is not the only way to be wrong -- an index that is off by the
    descriptor stride, or a schedule whose expert and whose descriptor disagree, reads some other
    expert.  Requiring the split arm to reproduce a grouped run over the reversed parameter pins
    every expert to its own allocation, not merely to "not expert 0".
    """
    config, workspace, schedule, inputs = _setup(context)
    grouped = inputs[6:9]
    reversed_grouped = tuple(weight.flip(0).contiguous() for weight in grouped)
    expected = _grouped_twice(config, workspace, schedule, inputs, reversed_grouped)
    # The same allocations as `grouped`, named back to front: expert e of the call is expert
    # E-1-e of the parameter, which is exactly what `reversed_grouped` holds.
    split = tuple(_split([weight[index] for index in reversed(range(LOCAL_EXPERTS))]) for weight in grouped)
    _assert_identical(expected, _run(config, workspace, schedule, inputs, split), "a reversed expert table")


def test_split_combined_fc1_matches_grouped(context):
    """One ``[2I, H]`` parameter per expert for Gate and Up, which is what MCore actually owns.

    The two halves share a table, so the Up addresses are the Gate addresses plus the offset of
    the Up half -- the pointer counterpart of the scale block offsets a combined MXFP8 FC1 uses.
    Getting that offset wrong reads the Gate rows twice, which this catches because Gate and Up
    hold different weights.
    """
    config, workspace, schedule, inputs = _setup(context)
    gate, up, down = inputs[6:9]
    combined = torch.cat((gate, up), dim=1).contiguous()  # [E, 2I, H]
    expected = _grouped_twice(config, workspace, schedule, inputs, (combined, combined, down))

    combined_split = _split([combined[index] for index in range(LOCAL_EXPERTS)])
    down_split = _split([down[index] for index in range(LOCAL_EXPERTS)])
    _assert_identical(expected, _run(config, workspace, schedule, inputs, (combined_split, combined_split, down_split)), "a split combined FC1")
