# emmy-ptx

A compiler from [Emmy](https://github.com/cloudrift-ai/emmy)'s Kernel IR straight to NVIDIA PTX,
written for the [RiftStack Emmy Hackathon 2026](https://riftstack.ai/hackathon).

```bash
python compile.py path/to/kernel.ir.json > out.ptx
```

It takes one Kernel IR file, finds the kernel's `meta.json` next to it, and prints a PTX module
that `ptxas` assembles for `sm_89` and that runs unchanged on the runner's RTX 5090 (`sm_120`).
Emmy is used for one thing only: `Graph.from_dict` to deserialise the IR. Every instruction is
produced by the code in `ptxgen/`. No CUDA C++, no `nvcc`, no Emmy renderer.

## Results

| set | problems | pass |
|---|---:|---:|
| open problem set (all 5 categories) | 15 | **15** |
| `practice/extra`: hidden-set-like variations I generated with the task's own `tools/generate.py` | 25 | **25** |
| `practice/hard`: odd shapes, 1024x1024 matmul (rasterised CTAs), unpinned reductions, 512-token attention | 14 | **14** |

Every one is checked the way the hidden set is evaluated: `compile.py` runs on the IR, the
runner's `ptxas` assembles the PTX, it launches under `meta.json`'s geometry on the RTX 5090, and
the output is compared with `expected.npz` within the problem's tolerance.

The open set on the runner, median of 20 launches after 3 warm-ups (`tools/bench.py`; no clock
locking, so the times are indicative):

| problem | category | median us | registers | max abs error |
|---|---|---:|---:|---:|
| p00_scalar_add | pointwise | 3.1 | 16 | 0 |
| p01_vector_add | pointwise | 3.1 | 16 | 0 |
| p02_scale_shift | pointwise | 3.2 | 22 | 0 |
| p03_relu_mul | pointwise | 3.2 | 20 | 0 |
| r01_array_sum | reduction | 3.0 | 18 | 3.8e-06 |
| r02_big_sum | reduction | 21.1 | 18 | 1.5e-05 |
| r03_mean | reduction | 13.0 | 18 | 9.3e-10 |
| r04_row_sum | reduction | 3.1 | 20 | 5.7e-06 |
| m01_matmul_512 | matmul | 15.9 | 40 | 8.4e-05 |
| m02_matmul_rect | matmul | 5.0 | 35 | 1.1e-05 |
| f01_rmsnorm | fused | 4.8 | 20 | 1.9e-06 |
| f02_row_softmax | fused | 3.1 | 20 | 7.5e-09 |
| f03_l2norm | fused | 3.1 | 20 | 1.5e-08 |
| a01_attention | attention | 84.6 | 136 | 6.6e-07 |
| a02_attention_128 | attention | 35.5 | 80 | 4.5e-07 |

The pointwise kernels are bit-exact (`atol = rtol = 0` is required there), and compiling any
kernel takes about half a second, most of it importing Emmy. The full table for all 54 problems
is in [`docs/bench.md`](docs/bench.md).

## How it works

```
kernel.ir.json ──Graph.from_dict──▶ statement tree ──▶ ptxgen ──▶ PTX text
meta.json ───────────────────────▶ arg_order, block size, dynamic smem size
```

| file | role |
|---|---|
| `compile.py` | the CLI contract: IR path in, PTX on stdout, diagnostics on stderr |
| `ptxgen/kernel.py` | loads IR + `meta.json`, lays out shared memory, writes the prologue and the module |
| `ptxgen/lower.py` | one handler per IR statement (`s_Tile`, `s_Load`, `s_WarpShuffle` ...) plus index expressions |
| `ptxgen/values.py` | compile-time values: constants, registers and affine integer forms, with scoped CSE |
| `ptxgen/ops.py` | the elementwise math library: Emmy op name to a PTX sequence |
| `ptxgen/emitter.py` | virtual registers, labels, instruction lines |

### 1. Every IR name owns one register

Emmy's IR reads like C: `acc0 <- add(acc0, in0)` inside a loop, `acc1 = sa + sb` reassigning
the online-softmax state, `TreeHalve` folding back into the same names. I give every IR name
one virtual register for its whole life. A later assignment to the same name is a `mov` into
that register. So a loop body is emitted once, and every iteration reads and writes the same
registers. That is the semantics of the C the IR describes, and no phi nodes are needed.
`ptxas` turns it into SSA and does the real register allocation, which is what the task asks
for ("do not build liveness analysis").

The one subtlety: a register may never be shared by two names. If it were, reassigning one
would silently change the other the next time a loop came round. `CodeGen.define` enforces
this. A new name takes over a fresh register, and a name that would alias someone else's
register gets a copy.

### 2. Integer index math stays symbolic until it's needed

The IR's indices are expression trees like
`_b_smem[((_ks/64 % 3)*64 + _ki), (((a1_b*64) + (a1_u*2)) + 1) - (a1_b*64)]`.
Lowered naively, every load in the unrolled matmul loop re-computes all of it.
`values.py` keeps integers as **affine forms**, `c0 + c1*r1 + c2*r2 + ...` over registers:

* adds, subtracts and multiplies by constants are free; they just rewrite the form;
* `((a1_b*64) + (a1_u*2) + 1) - (a1_b*64)` cancels to `a1_u*2 + 1` at compile time;
* exact division folds, so `(x*64 + 128) / 64 = x + 2`; and with a non-negativity proof
  `(64x + 70) % 64 = 6`;
* when an address is needed, the form splits into a variable part (one register) and a
  constant part, which becomes the **immediate offset** of the load.

The unrolled matmul inner loop therefore becomes loads with immediate offsets from two base
registers, interleaved with `fma`s:

```
ld.shared.f32  %f5,  [%r57+24576];    // in1__c0_0 = _b_smem[...]
ld.shared.f32  %f6,  [%r60];          // in0__c0_0 = _a_smem[...]
fma.rn.f32     %f7,  %f5, %f6, %f1;
ld.shared.f32  %f8,  [%r57+24580];    // in1__c0_1
fma.rn.f32     %f9,  %f8, %f6, %f2;
ld.shared.f32  %f10, [%r60+256];      // in0__c1_0
```

Materialisations and opaque operations (register times register, division, remainder) are
**common-subexpression cached**. The cache is **scoped like the generated code**. A loop body
and each arm of an `if` open a scope that is dropped when the region ends. So a cached
register is only reused where its definition dominates the use, and a value computed inside
a branch is never read on a path that skipped it. If an integer name is ever reassigned, the
whole cache is flushed.

C semantics are kept exactly: `/` truncates and `%` follows the dividend's sign. So
`x / 8` becomes `shr.u32` (and `x % 8` becomes `and.b32`) only when `x` is proven
non-negative. Thread ids, axis indices and sums and products of them are tracked as
non-negative registers.

### 3. Numerics: one IR op, one rounding

The pointwise problems demand bit-exact output. Two decisions follow from that:

* `add`, `sub` and `mul` carry an explicit `.rn`. Without a rounding modifier `ptxas` may
  contract `mul` and then `add` into a single `fma`, which rounds once instead of twice. That
  is more accurate, but it is no longer what numpy computes.
* `divide`, `reciprocal` and `sqrt` use the IEEE forms (`div.rn`, `rcp.rn`, `sqrt.rn`),
  never `.approx`.

The one place I do contract is a **reduction**. When `t = a*b` feeds only
`acc <- add(acc, t)` in the next statement, the pair becomes `fma.rn acc, a, b, acc`.
Reductions are always graded with a tolerance, because summation order is free (the
runner's tree reduction and numpy disagree anyway), so `fma` is strictly a gain. That covers
the matmul inner loop, `sum(x*x)`, RMSNorm and attention's `q·k`.

Transcendentals use the hardware units through identities: `exp(x) = ex2(x·log2 e)` and
`ln(x) = lg2(x)·ln 2`. `tanh` is built from `exp` in an overflow-free form
(`(1-e^-2|x|)/(1+e^-2|x|)` with the input's sign copied back). I don't use `tanh.approx`,
whose roughly 2^-11 error is too coarse for these tolerances. `erf` (and so GELU) uses
Abramowitz–Stegun 7.1.26, whose error is below 1.5e-7. The max-reduction identity is
Emmy's own `-1e30`, not `-inf`, so `exp(acc - max)` never sees `inf - inf`.

### 4. Following the IR's own rules

These rules aren't printed anywhere obvious, but correctness depends on them:

* **Accumulator seeding.** A `Loop` / `StridedLoop` seeds every `Accum` in its *immediate*
  body to the op's identity right before the loop, unless `seed=False`. Matmul accumulators
  are `Init`-ed outside the `_ki` loop and must not be reset per tile. In attention, `acc0`
  restarts for every key column (the inner loop), while the softmax state carries across
  the outer loop.
* **`Accum.base`.** In the online softmax, `acc1 <- add(sa, sb)` means `acc1 = sa + sb`,
  not `acc1 += ...`. `sa` already contains the rescaled old `acc1`.
* **The tile guard.** `if (gid < N)` is not in the IR. It is needed exactly when the launch
  has threads past the last cell: `N % block_threads != 0`, or a warp-specialised producer
  band. Elsewhere it's left out, as Emmy does, because a guard around a `bar.sync` would be
  a hazard.
* **Tile decode.** `gid` decodes row-major into the tile axes. The outermost axis needs no
  modulo, and grouped CTA rasterisation (`raster_group`, used by large matmuls for L2 reuse)
  remaps the two block axes exactly as Emmy's launch expects.
* **Parameters** come from `meta.json`'s `arg_order`, not the node's input list.

### 5. Cross-thread machinery

* `WarpShuffle` becomes an XOR butterfly: `shfl.sync.bfly.b32` at 16, 8, 4, 2, 1 over the
  bits of each state component, with the IR's `combine_states` program applied after each
  step. That handles both a plain sum and the two-component online softmax `(max, sum)`. The
  mask is `activemask`, not `0xffffffff`, so a partial warp is still legal.
* `TreeHalve` is unrolled (its length is static): `if (warp < s)` folds slot `warp+s` into
  slot `warp`, then `bar.sync`, then every thread reads the root. It supports the
  segment-indexed (`inner`) layout and named barriers.
* `CpAsyncCopy` / `Commit` / `Wait` map to `cp.async.cg` (16 B) or `cp.async.ca` (4 and 8 B),
  plus `commit_group` and `wait_group N`. That gives the matmul's three-stage pipeline
  verbatim. Emmy's software swizzle (`e ^ ((e >> s) & m) << 3`) is applied when a mode is
  set.
* **Shared memory** is laid out once, at module scope. Static `.shared` arrays go up to
  48 KiB. Above that, a single `.extern .shared` pool is carved by offset, and the launch's
  `smem_bytes` sizes it. `m01` needs 72 KiB of `cp.async` stages. Vector loads and stores
  (`v2`, `v4`) come straight from the IR's grouped loads and writes. Read-only inputs use
  `ld.global.nc`, and atomic writes use `red.global.add.f32`.

### 6. Loops

A loop with static bounds and Emmy's `unroll` flag is unrolled in the compiler: the
induction variable becomes a constant, so every index in the body folds. That only happens
within a size budget, because ptxas gets 30 s per kernel. Everything else is a real loop
with a test at the top. A one-trip loop is always inlined, and a `Cond` whose predicate is
decided at compile time emits only the taken arm.

## What it doesn't do

Everything the task marks as appearing in no problem is out of scope: tensor cores (`mma`,
`wgmma`, `ldmatrix`), TMA and mbarriers, carried state (`Pre` / `Carry`), and symbolic shapes.
Buffers must be 32-bit floats. A kernel that uses any of these fails with a clear message on
stderr and exit code 1, rather than producing wrong PTX. I also added `Select` (only seen in
odd-shaped matmuls) and the integer and predicate variants of the elementwise ops.

## Tests

```bash
pip install emmy-ml==0.3.11 numpy requests pytest
python -m pytest -q                                   # offline: 125 tests
EMMY_TASK_PROBLEMS=../emmy-hackathon-2026/problems python -m pytest -q    # + the open set
EMMY_SUBMIT=../emmy-hackathon-2026/submit.py python -m pytest -q tests/test_runner.py   # on the GPU
python tools/run_suite.py --submit ../emmy-hackathon-2026/submit.py ../emmy-hackathon-2026/problems practice/extra practice/hard
python tools/bench.py ../emmy-hackathon-2026/problems practice/extra practice/hard
```

* `tests/test_values.py` covers C division semantics, affine cancellation, exact division,
  the non-negativity gates on shift and mask, and that CSE scopes never leak a register out
  of a branch.
* `tests/test_compile.py` compiles every problem it can find and checks the contract: pure
  ASCII, `.version` first, entry name, parameter order equal to `arg_order`, every register
  declared, every branch target defined.
* `tests/test_runner.py` is opt-in. It runs the real thing on the organisers' GPU.

`practice/` holds the 39 extra problems (IR, `meta.json`, `expected.npz`) generated with the
task repository's `tools/generate.py` from `tools/expressions_extra.py` and
`tools/expressions_hard.py`. Emmy can't trace `amin` or `logsumexp` yet, so those two were
dropped.

## AI assistance

The task allows AI coding assistants. This compiler, its tests and this README were written
with an AI coding agent (Anthropic's Claude) for Rui Codex. The design decisions above are
documented so that each one can be checked against the code and the PTX it produces.

## Licence

MIT, see `LICENSE`.
