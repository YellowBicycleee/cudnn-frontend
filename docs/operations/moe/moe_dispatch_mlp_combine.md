# MoE Dispatch, Expert MLP and Combine

**Experimental.** `cudnn.ops.moe_dispatch_mlp_combine_forward` and `cudnn.ops.moe_dispatch_mlp_combine_backward`
run one expert-parallel mixture-of-experts layer: each rank's tokens are dispatched to the ranks that own their
top-k experts, the routed experts' gated MLPs (and an optional shared expert's) are computed, and the results are
combined back on the tokens' rank. The forward and the backward are each one persistent megakernel that overlaps
the cross-rank token traffic with the expert GEMMs, plus a few small launches listed under
[Launches per call](#launches-per-call).

The operator is derived from [Mixture-of-Kittens](https://github.com/cursor/mixture-of-kittens) by Cursor
Research, whose megakernels are written in CUDA C++;
see its [README](https://github.com/cursor/mixture-of-kittens/blob/main/README.md) and
[blog post](https://cursor.com/blog/mixture-of-kittens) for the design, and [Citation](#citation).

A layer runs in two steps, with the same division of work as the MoeEP operation (`cudnn.MoeEp`):

- **Prepare**, once per layer configuration: `cudnn.ops.moe_dispatch_mlp_combine_prepare(...)` compiles every
  kernel the forward and backward launch and allocates the symmetric-memory buffers, counters and scratch they
  use. It returns a `cudnn.ops.MoeDispatchMlpCombinePlan`.
- **Execute**, every step: the forward and backward take the plan and the caller's output buffers. They compile
  nothing, allocate no device memory (one exception: [split routed weights](#options)) and never read device
  memory back to the host.

## Operation

Let rank $r$ hold $T$ tokens $x_t \in \mathbb{R}^{H}$. Token $t$ selects up to $K$ global experts $e_{t,k}$
with router weights $p_{t,k}$; $e_{t,k} = -1$ marks an unused slot. With $E$ global experts over $P$ ranks,
each rank owns $E_{\mathrm{local}} = E / P$ consecutive experts. Weights are stored `[out, in]`. For a routed
expert $e$ with $W^{g}_{e}, W^{u}_{e} \in \mathbb{R}^{I \times H}$ and $W^{d}_{e} \in \mathbb{R}^{H \times I}$:

$$
g_{t,k} = x_t {W^{g}_{e_{t,k}}}^{\top},\qquad
u_{t,k} = x_t {W^{u}_{e_{t,k}}}^{\top},\qquad
h_{t,k} = \mathrm{SiLU}(g_{t,k}) \odot u_{t,k},\qquad
z_{t,k} = h_{t,k} {W^{d}_{e_{t,k}}}^{\top}.
$$

The shared expert computes $s_t$ the same way from $W^{g}_{s}, W^{u}_{s}, W^{d}_{s}$ (no routing), and the
output is

$$
y_t = s_t + \sum_{k:\, e_{t,k} \ge 0} p_{t,k}\, z_{t,k}.
$$

The router weight is applied to the Down output, during the combine. The backward returns
$\partial L/\partial x$, $\partial L/\partial p_{t,k} = \langle \partial L/\partial y_t,\ z_{t,k}\rangle$
(zero at unused slots) and the gradients of all six weights. The equations describe the mathematics: GEMM
operands and the saved activations are BF16 (or MXFP8, below), accumulation is FP32.

Options change the MLP:

| Option | Effect |
|---|---|
| `swiglu_limit=C` | clamped SwiGLU: $\min(g, C)$ for the gate and $\mathrm{clip}(u, -C, C)$ for the up projection |
| `situ_beta=(\beta_1, \beta_2)` | SiTU-GLU instead of SwiGLU: $h = \beta_1 \tanh(g/\beta_1)\,\sigma(g) \cdot \beta_2 \tanh(u/\beta_2)$ |
| `shared_output_gate=G` (forward) | $y_t = G_t\, s_t + \sum_k p_{t,k} z_{t,k}$ with the caller's $G = \sigma(x w^{\top})$, BF16 `[T, 1]`; its backward is not available in this API |
| no shared expert | pass `None` for the three shared weights: $y_t$ is the routed sum |
| combined FC1 | pass one `[E_local, 2I, H]` tensor as both routed gate and up weights (gate rows first) |
| `routed_hidden_size=R` (prepare) | the routed experts work in width $R \ne H$ on `routed_input` `[T, R]`; the forward returns the pair $(s, \sum_k p z)$ and the backward takes `routed_grad_output` |

## Usage

```python
import torch
import torch.distributed as dist
import cudnn

config = cudnn.ops.MoeDispatchMlpCombineConfig(
    fwd_num_comm_sms=40, bwd_num_comm_sms=28,       # SMs each megakernel gives to communication
    fwd_minibatch_size=4096, bwd_minibatch_size=8192,
    macrobatch_size=131072,
)
plan = cudnn.ops.moe_dispatch_mlp_combine_prepare(   # collective over the group, outside graph capture
    config, dist.group.WORLD, device=torch.device("cuda", local_rank),
    max_tokens_per_rank=16384, hidden_size=H, intermediate_size=I, num_local_experts=E_local, topk=K,
)

def empty(shape, dtype=torch.bfloat16):            # buffers of the plan's *_shapes
    if shape is None:
        return None
    if isinstance(shape[0], int):
        return torch.empty(shape, dtype=dtype, device="cuda")
    if isinstance(shape[1], torch.dtype):           # MXFP8 context: (shape, dtype)
        return torch.empty(shape[0], dtype=shape[1], device="cuda")
    return tuple(empty(item, dtype) for item in shape)

context_shapes, out_shape = plan.forward_shapes(config, T)
d_x_shape, d_router_shape, d_weight_shapes = plan.backward_shapes(T)
schedule_out = empty(plan.schedule_shapes(T), torch.int32)
context_out, out = empty(context_shapes), empty(out_shape)
d_x, d_router = empty(d_x_shape), empty(d_router_shape, torch.float32)
d_weights = empty(d_weight_shapes)

weights = (w_shared_gate, w_shared_up, w_shared_down, w_routed_gate, w_routed_up, w_routed_down)
y, schedule, saved = cudnn.ops.moe_dispatch_mlp_combine_forward(
    x, top_experts, router_weights, *weights,
    plan=plan, config=config, out=out, context_out=context_out, schedule_out=schedule_out,
)
(d_x, d_router_weights, d_w_routed_gate, d_w_routed_up, d_w_routed_down,
 d_w_shared_gate, d_w_shared_up, d_w_shared_down, _) = cudnn.ops.moe_dispatch_mlp_combine_backward(
    d_y, x, router_weights, *weights, schedule, saved,
    plan=plan, config=config, out=d_x, d_router_weights_out=d_router, d_weights_out=d_weights,
)
```

Both functions return a `TupleDict`, which unpacks as above and is also read by name: the forward's keys are
`output`, `schedule` and `forward_context`; the backward's are `d_x`, `d_router_weights`, `d_w_routed_gate`,
`d_w_routed_up`, `d_w_routed_down`, `d_w_shared_gate`, `d_w_shared_up`, `d_w_shared_down` and `d_w_output_gate`
(always `None` in this API).

`y` is `out`, `schedule` lives in `schedule_out` and `saved` in `context_out`: keep those buffers untouched
from the forward to its backward. The backward writes into `d_x`, `d_router` and `d_weights`. With `main_grads`
instead of `d_weights_out` it accumulates the weight gradients into the caller's buffers (FP32 or BF16, the
dtype given to `prepare` as `main_grad_dtype`).

One plan serves every token count $T$ up to `max_tokens_per_rank`; `config` may change between calls in its
runtime fields (communication SMs, minibatch sizes, a macrobatch up to the plan's), and none of them compiles a
new kernel. Calls on one plan share its buffers and must not overlap: issue them in one stream order.

### Tensors

| Argument | Shape and dtype |
|---|---|
| `x`, `d_y` | BF16 `[T, H]`, contiguous |
| `top_experts` | int32 `[T, K]`, contiguous; global expert ids, `-1` for an unused slot |
| `router_weights` | FP32 `[T, K]`, contiguous |
| `w_shared_gate`, `w_shared_up` | BF16 `[I, H]` |
| `w_shared_down` | BF16 `[H, I]` |
| `w_routed_gate`, `w_routed_up` | BF16 `[E_local, I, H]` |
| `w_routed_down` | BF16 `[E_local, H, I]` |
| `out`, `d_x` | BF16 `[T, H]` |
| `d_router_weights_out` | FP32 `[T, K]` |
| `d_weights_out` | six BF16 buffers shaped like the weights, ordered shared / routed per matrix (gate, up, down) |

### MXFP8 routed weights

Prepare with `routed_precision="mxfp8"` and pass the routed weights prequantized to MXFP8 (E4M3 payload with
one E8M0 scale per 32 elements, `cudnn.moe_dispatch_mlp_combine.ops.mxfp8_quantize`): the forward takes
`(e4m3, scale)` per matrix; the backward takes `(e4m3, scale, e4m3_t, scale_t)` for gate and up and
`(e4m3_t, scale_t)` for down. Activations, outputs and gradients stay BF16; the kernels quantize x and the
hidden activations on the fly. `config.gate_up_context` selects the precision of the routed gate / up
activations kept for the backward: `"mxfp8"` (default) or `"bf16"` (more memory, no requantization).
`prepare(native_columnwise=True)` accepts TransformerEngine's columnwise payload layout for the transposed
operands.

### Options

- Split routed weights: per-expert allocations passed as `SplitRoutedWeight` (in
  `cudnn.moe_dispatch_mlp_combine.functional`) with a table of their addresses; prepare with
  `split_routed_weights=True` and accumulate into `main_grads` (with `main_grad_storage_tables`). The first call
  over a new set of address tables allocates and builds their TMA descriptors once (one small launch per routed
  operand); later calls over the same tables reuse them.
- Variants are compile-time choices made at prepare: `swiglu_clamped`, `situ`, `has_shared`,
  `shared_output_gate`, `split_routed_weights`, `main_grad_dtype`, `routed_precision`, `native_columnwise`,
  `config.gate_up_context` and `config.comm_tile_shape` (BF16 only: rows x columns of one communication tile,
  `(32, 512)`, `(16, 1024)`, or `"auto"`: 16 x 1024 when the routed hidden size is a multiple of 1024). Executing a variant the plan did
  not compile raises `RuntimeError` instead of compiling.
- `zero_unrouted_router_grads=False` skips the masking of unused slots in `d_router_weights`, whose values there
  are then unspecified.

## Launches per call

| Call | Launches, in order |
|---|---|
| forward, routing | route all-gather → EP barrier → scheduler (three kernels) |
| forward, BF16 | staging of x and the router weights into symmetric memory (one kernel copying the routed rows; two copies when x is not 256-byte aligned or the routed hidden differs) → EP barrier → counter reset (memset) → forward megakernel → EP barrier → combine epilogue |
| forward, MXFP8 | two staging copies → EP barrier → counter reset → forward megakernel → EP barrier → combine epilogue |
| backward, BF16 | staging of dY, x and the router weights (one kernel, or three copies as above) → EP barrier → counter reset → metadata builder → backward megakernel → EP barrier → dX epilogue → copy of the router-weight gradients → compare and masked fill of the unused slots (with `zero_unrouted_router_grads`) |
| backward, MXFP8 | three staging copies → EP barrier → counter reset → backward megakernel (one compiled call with its tensormap prologue and, without `main_grads`, the zeroing of empty experts' weight gradients) → EP barrier → dX epilogue → router-weight gradient copy → masking of the unused slots |

All launches go to the current stream. The forward and backward are collective over the plan's process group:
every rank calls them in the same order with the same $T$ (not checked: a check would read device memory).

## CUDA graphs

- Prepare outside capture. The first forward / backward after prepare can be captured without an eager warm-up:
  nothing is compiled inside the capture, and a variant the plan did not compile raises there.
- A graph replays its own $T$ and addresses: inputs, weights, `out`, `context_out`, `schedule_out` and the
  gradient buffers must keep their addresses until every graph that uses them is destroyed. Their contents,
  including the routing, may change between replays. The plan's counters, scratch and symmetric buffers have
  fixed addresses for the plan's lifetime.
- Capture and replay on every rank in the same order, on one stream per plan; graphs that share a plan must not
  replay concurrently.

## Support

| Question | Answer |
|---|---|
| Device | NVIDIA Blackwell, compute capability 10.0 or 10.3 (B200, B300, GB200, GB300) |
| EP group | 1, 4, 8, 16, 32 or 64 ranks whose GPUs share PyTorch symmetric memory |
| Tokens per rank | a multiple of 256, at least 512, at most `max_tokens_per_rank` |
| Sizes | hidden and intermediate sizes multiples of 128 and at least 256 (a different routed hidden size: a multiple of 256); top-k in `[1, 255]` |
| Config | minibatch sizes multiples of 256 that divide the macrobatch; an even number of communication SMs |
| Routing | the routed rows a rank receives, padded to 256 per local expert, must fit the schedule of $T \cdot K \cdot \max(2, \lceil P \cdot m \rceil)$ rows with $m$ = `config.schedule_capacity_multiplier`; a schedule that does not fit stops the kernel with a device-side trap |
| Dtypes | BF16 activations and weights, or MXFP8 routed weights; FP32 router weights; int32 routing |
| Layouts | contiguous `[T, ...]` activations and routing; anything else is declined with `NotImplementedError` naming the tensor |
| Not in this API | the backward of a layer with a shared output gate; a backward from a forward context without its routed activations (the replay of the first macrobatch) |
| Software | PyTorch 2.9 or later built for CUDA 13 (`pip install --group torch`); `nvidia-cutlass-dsl[cu13] >= 4.8.0` in its cu13 build (a lower version or the cu12 build is declined with an error naming it) |

## Source and tests

Source: `python/cudnn/moe_dispatch_mlp_combine/` (`api.py` the operator functions, `functional.py` the layer
interface they call, `cutedsl/` the kernels and their hosts, `cutedsl/_compat/` the helpers adapted from QuACK). The
layer interface (`get_workspace`, `build_schedule`, `forward`, `backward`) and the helpers named above are
imported from the package and are not exported from `cudnn.ops`.

Tests: `test/python/moe_dispatch_mlp_combine/cutedsl/`. The host-only tests run anywhere; the tests that need an EP
group run under `torchrun` on SM100 / SM103 GPUs that share symmetric memory, for example
`torchrun --standalone --nproc-per-node=8 -m pytest moe_dispatch_mlp_combine/cutedsl/test_api.py` from
`test/python`, and skip elsewhere.

## Citation

```
Stuart H. Sul, Nash Brown, Henry Wildermuth, William Lin, and Federico Cassano. "Mixture-of-Kittens: MoE Megakernel for NVL72s." Cursor Research, Aug 2026. https://github.com/cursor/mixture-of-kittens
```
