# Vendoring record: QuACK

This file records where the QuACK sources in this directory come from and how they differ from upstream.  The
license and attribution are in [`LICENSE`](LICENSE) (QuACK's, unchanged) and in the QuACK section of the repository's
[`THIRD_PARTY_LICENSES.txt`](../../../../../THIRD_PARTY_LICENSES.txt).

## Upstream

- **Project**: QuACK, https://github.com/Dao-AILab/quack (PyPI `quack-kernels`), Apache License 2.0.
- **Version**: `quack-kernels` 0.5.0, files taken from the released wheel (`quack/<file>.py`); the hashes below are
  the ones in that wheel's `RECORD`.
- **Vendored subset**: the modules the CuTe DSL kernels reach, closed over their own imports: the kernels use
  `gemm_default_epi.GemmDefaultSm100` (the base class of the BF16 / MXFP8 GEMM collectives), `gemm_base.NamedBarrierGemm`,
  `pipeline.PipelineTmaUmma` / `PipelineUmmaAsync`, `varlen_utils.VarlenArguments`, `copy_utils`,
  `layout_utils.tile_atom_to_shape_SF_strided` and `activation.swiglu` / `dswiglu`.  `gemm_default_epi` also defines
  the SM80 / SM90 / SM120 classes, so `gemm_sm80`, `gemm_sm90`, `gemm_sm120` and `sm80_utils` come along without
  being used; `trace` is imported only inside QuACK's own GEMM kernels, which this operator does not launch.
- **Not vendored**: QuACK's package `__init__.py` (it optionally patches the CuTe DSL compiler binaries when
  `QUACK_CUTE_DSL_SHIM=1` or `CUTE_DSL_PTXAS_PATH` is set, and imports the RMSNorm / softmax / cross-entropy
  kernels), and every other module.  `__init__.py` here is a package marker that imports and patches nothing.

## Files

23 Python files.

| File | Wheel `RECORD` hash of `quack/<file>` | Local changes |
|---|---|---|
| `activation.py` | `sha256=zXAG9T5yFWeGeljJZySf82tzlkN04P2jhaTnNo2zJ1g` | none |
| `copy_utils.py` | `sha256=gJRxYeIPG61shnEWT2KPqgyNKBitfcmCTakI4ZfGMlo` | relative imports; `cute.core.ThrCopy` -> `cute.ThrCopy` (2 annotations) |
| `cute_dsl_utils.py` | `sha256=4jBC37wrqgJU_2Y3HFVbi46rQBkade3-ymPcHIxAYQk` | import-time converter patch removed (see below) |
| `epi_composable.py` | `sha256=O2IemnbOrvh2AGUD5LwXr6dL-thikLKQgPmOSqTqjMQ` | relative imports |
| `epi_ops.py` | `sha256=6yW3lDYnKHypKOiZEB5P28QUr47rxZDVWUSEaYwn0EM` | relative imports |
| `epi_utils.py` | `sha256=bAVkc1uGD7K3sCz2PMD1PbORLs9aTyQuUywqbFifOJk` | relative imports |
| `fast_math.py` | `sha256=4vrW_HB65rBlCyg0d6h6Y8Q4Gvjti5g74yqCv7acsxA` | none |
| `gemm_base.py` | `sha256=k9PpwZfRtDLRXmaWQQUs9DZgHJ8lYVNtSVY4unmxWNY` | relative imports |
| `gemm_default_epi.py` | `sha256=jNoHY2_aFKCBqwU2F3-AkcSdbcFspvflI6J5VJxgfTQ` | relative imports |
| `gemm_sm100.py` | `sha256=VUdomgWeMQb_DxbZG2nRfJrmRTdCVO1NLkBq71-kHdc` | relative imports |
| `gemm_sm120.py` | `sha256=UswFAWPBFALcmnXVPxsGSd9EnQCpQBDw9PbNEa6Fvcs` | relative imports |
| `gemm_sm80.py` | `sha256=aqoicoLxc9uNF8zm2sK33_7crkApLFOqnWZeM-pmo-M` | relative imports |
| `gemm_sm90.py` | `sha256=MhCFuDYmjeuWGAyl-n2Qv1b1j0CBVDDsmVmiqVXrbPk` | relative imports |
| `layout_utils.py` | `sha256=B3L1kOR8DLADQaTff-ERqbRVjTzj60AQ25wHOJ_OlbU` | `cute.core.ThrMma` / `cute.core.ThrCopy` -> `cute.ThrMma` / `cute.ThrCopy` (4 annotations) |
| `pipeline.py` | `sha256=eW8rmeOEsim3jArWW6D4CYTi5nfk2K-kLym89ilUjD8` | none |
| `rounding.py` | `sha256=E810UAfi7C4AI72OMJnA-6pRoniwCty01RmpWKeN6EE` | none |
| `sm100_utils.py` | `sha256=-AkAPWzg-2cXZMwTijIVTvWWwZmk5m3bAsOJQTffzDo` | none |
| `sm80_utils.py` | `sha256=0yOgRslAIq8B7MS8tIkgZuilOMIkYnJDUl7c2gjdkzg` | none |
| `sm90_utils.py` | `sha256=QaLZ3MLlp9dKOD-VwdhxmqEstPdq8CVGaNwffbwspNw` | none |
| `tile_scheduler.py` | `sha256=n8anPwy6Wpcds3DLVZY_A0XXxhLNJzIhNNAxr79xB9g` | relative imports |
| `trace.py` | `sha256=0dz4Ehk-bacF9JK9hCwAN8tSLNRpz56lzCuiLbXeky4` | relative imports |
| `utils.py` | `sha256=NY8C4H5rByXqVgl5vdUDNqZWIYM4zgTeAEvvmtrpT5s` | none |
| `varlen_utils.py` | `sha256=DcaFPYhUmk8I2b_k37OutH4jJnqEe1iop4pOeUWEIkc` | relative imports |

## Local differences from upstream

Each file starts with a one-line vendoring note (source file and local changes) and an
`SPDX-License-Identifier: Apache-2.0` line; below them the original copyright header is kept verbatim.

- **Relative imports**: `from quack.X import Y`, `from quack import X` and `import quack.X as Z` are written
  `from .X import Y`, `from . import X` and `from . import X as Z` (also the function-local imports of `rounding` and
  `trace`), so the vendored modules never import an installed `quack`.
- **CuTe DSL 4.8**: `cute.core.ThrCopy` / `cute.core.ThrMma` annotations are spelled `cute.ThrCopy` /
  `cute.ThrMma` (2 in `copy_utils`, 4 in `layout_utils`); CuTe DSL 4.8 no longer has them under `cute.core`.  The
  code is otherwise the 0.5.0 code, the same that runs on CuTe DSL 4.8 with these six annotations changed.
- **No CuTe DSL patch at import**: upstream `cute_dsl_utils` replaces
  `cutlass.cute._tvm_ffi_args_spec_converter._convert_single_arg` when it is imported (so that TVM FFI arguments
  annotated `cutlass.Constexpr[...]` become compile-time constants and NamedTuple arguments use their own type
  hints).  The vendored copy drops that block (its `get_origin` and `spec` imports are kept, now unused): a process
  may also import another QuACK, which applies its own patches, and this package should not change CuTe DSL for other
  libraries.  The operator does not need it: every `cute.compile` entry of the package takes tensors, DSL scalars and the
  stream, none annotated `Constexpr` and none a NamedTuple, so the patched converter passed them through unchanged.

- **Formatting**: the files are formatted with this repository's black configuration (line length 160), which
  changes line breaks only.

No other differences.  `test/python/moe_dispatch_mlp_combine/cutedsl/test_cutedsl_no_quack.py` checks that nothing imports `quack`, that every import in
this directory resolves to a vendored file, and that no module here assigns an attribute of a CUTLASS module at
import.

## Updating

Copy the new release's files for the same closure (follow each file's `quack` imports), re-apply the changes above,
update the vendoring notes, this table and the QuACK section of `THIRD_PARTY_LICENSES.txt`, and check generated code
for an unchanged kernel before and after.
