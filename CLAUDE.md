# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

Research code comparing quantum error-correction decoders on the **BB72 `[[72,12,6]]` bivariate
bicycle code** (Bravyi et al. layout: 6x6 torus, `A = x^3 + y + y^2`, `B = y^3 + x + x^2`). Four
decoder backends are benchmarked:

1. **BP+OSD via `qldpc`** — pure-Python CPU, no GPU dependency.
2. **BP+OSD via CUDA-QX `nv-qldpc-decoder`** — GPU.
3. **Tensor-network decoder via CUDA-QX `tensor_network_decoder`** — GPU. This is the main
   research thread, plus a Walsh-Hadamard step that reconstructs the full 4096-sector logical
   posterior from 4095 per-parity marginals.
4. **Exact block branch-and-bound on the same TN decoder** — `v0.5_final_production/`, a frozen
   self-contained production baseline. Same 4096 sectors, reached by pruning rather than by
   the 4095-marginal Walsh-Hadamard route. See its own section below.

Git is initialised on branch `master` but has **no commits** — everything is untracked. There is
no linter config, packaging metadata, or CI. The only test suite is
`v0.5_final_production/test_exact_bnb.py` (CPU-only, pytest); everything else is established by
the cross-checks described under "Verifying changes".

## Environment: two very different machines

This is the single most important operational fact.

- **CUDA-QX does not run on the Windows Python host.** The `.qenv/` virtualenv at the repo root is
  essentially empty (only `pip`). Any script importing `cudaq_qec` aborts here by design.
- CUDA-QX work happens on **Linux with a conda env named `cudaqx-qec`**. Two ways in: PBS submission
  scripts (see below), or **WSL2** — the runtime-summary CSVs record runs from `/mnt/c/QEC` on an
  RTX 3050 laptop GPU, so WSL is a working local path.
- **Everything in `common/local_code_capacity_runtime.py` and the `qldpc` baseline is backend
  agnostic and runs on plain CPU Python.** Keep it that way: only the entry-point scripts under
  `examples/`, `tn_reconstruct/`, `bp_osd_baselines/`, and the decoder module in
  `v0.5_final_production/` may touch `cudaq_qec`. The first three guard the import with a pointed
  message; `v0.5_final_production/multi_logical_tensor_network_decoder.py` imports `cudaq_qec` and
  `quimb` at module top level **unguarded**, so on the Windows host it dies with a bare
  `ImportError` rather than an explanatory one. `v0.5_final_production/exact_block_bnb.py` is pure
  NumPy and stays host-runnable; only the decoder half is GPU-bound.

## Commands

Scripts are normally run **from their own directory** (the PBS scripts `cd` to the job's working
dir first). Each entry point inserts the repo root onto `sys.path` itself, so imports of `common.*`
and `examples.*` work regardless of cwd — but `default_results_dir()` resolves `results/` relative
to the *calling script's* file, so output lands per-directory either way.

```bash
# GPU/PBS runs — one submit script per directory
qsub examples/submit_tn_decoder.txt
qsub tn_reconstruct/submit_tn_decoder.txt
qsub bp_osd_baselines/submit_tn_decoder.txt
qsub v0.5_final_production/submit_v0.5_final_production.pbs

# Direct runs (Windows host / WSL / Linux). Single-syndrome, single-parity TN example:
python bb72_cudaqx_nv_tn_flip_prob_example.py --error-type Z --p 0.02 --seed 20260827

# Full 4096-sector posterior, cached contraction paths (the main experiment):
python bb72_cudaqx_nv_tn_flip_to_coset_reconstruct_cached.py \
  --device cuda --error-type Z --syndrome-mode sample --p 0.02 --seed 20260827 \
  --path-cache path_cache/bb72_cudaqx_nv_tn_path_cache_Z_weight6-x_cutensornet.pkl \
  --output-stem results/run

# Un-cached variant, and the CPU-only qldpc baseline (~0.5 ms/decode, no GPU needed):
python bb72_cudaqx_nv_tn_flip_to_coset_reconstruct.py --error-type Z --p 0.02
python ../bp_osd_baselines/bb72_qldpc_bp_osd_baseline.py --osd-orders 0,2,4

# Sweep the BP+OSD baselines over OSD order; --osd-orders is the knobs' main axis
python bb72_cudaqx_nv_qldpc_bp_osd_baseline.py --error-type X --p-grid 0.005,0.01,0.02 --shots 5000

# v0.5 production baseline — MUST run from v0.5_final_production/ (bare same-dir imports)
cd v0.5_final_production
pytest -q test_exact_bnb.py                 # CPU-only BnB logic, no CUDA-QX/qldpc needed
python run_bb72_easy_hard.py --warmup 1 --repeats 5 --output-dir production_smoke_results
```

The PBS scripts purge modules, activate `cudaqx-qec`, print a torch/CUDA banner, then run the
script (queue `normal`, 1 GPU, 24h, project `13004445`). The v0.5 script is the exception on
walltime: it asks for 1h, since the frozen baseline decodes in under a second per case.

**Known bug:** `bp_osd_baselines/submit_tn_decoder.txt` sets `#PBS -o` to
`output-bb72_qldpc_bp_osd_baseline.txt` but actually runs the *CUDA-QX* baseline. That is why the
file named after the qldpc run contains a CUDA-QX run, and why the qldpc baseline has no captured
stdout. Fix the `-o` to match the script if both need capturing.

## Architecture

Four layers, each importing only downward, plus a self-contained production package that sits
outside them (section 5).

### 1. `common/local_code_capacity_runtime.py` — decoder-agnostic Monte-Carlo harness

The engine both BP+OSD baselines plug into. `DecoderFactory` (protocol) builds a `SyndromeDecoder`
per physical error rate `p`; `CodeCapacityRunner.sweep` samples errors with a `NoiseModel`, decodes
each, and scores. Failure accounting has two distinct modes: **`syndrome_valid`** (correction
reproduces the syndrome) and **`logical_failure`** (valid correction, but the residual flips a dual
logical); `total_failure` is the OR of the two. Both counters reach the CSV separately, which is how
you tell a decoder that is wrong from one that is merely inconsistent.

`DecodingProblem` freezes the matrices for one error channel, and the **X/Z naming is deliberately
crossed**: an `"X"`-error problem decodes against `code.matrix_z` with the Z logicals as dual
observables; a `"Z"`-error problem uses `matrix_x` and X logicals. This crossing recurs everywhere
(e.g. `logical_basis_type = "Z" if error_type == "X" else "X"`), so read the pairing carefully
rather than assuming the label matches the matrix.

Decoders that happen to expose `converge` / `iter` attributes get extra CSV columns; those that
don't get `None`. `SweepResult.write_csv` emits one fixed schema, and both BP+OSD baselines call it,
which is the whole reason the two backends' CSVs are directly comparable.

### 2. `common/bb72_baseline_common.py` — code construction and shared CLI

`build_bb72()` constructs the code and asserts `[[72,12,6]]`. `build_problem` / `build_noise` /
`parse_p_grid` / `add_common_bb72_args` / `default_results_dir` are the shared spine. Defaults:
`p`-grid `0.005..0.05`, 5000 shots, seed `20260826` for the BP+OSD baselines and `20260827` for the
TN scripts — keep those two defaults distinct, they are intentional.

### 3. `common/bb72_tn_common.py` — tensor-network machinery (the non-obvious part)

**The TN decoder answers exactly one question per call**: `P(a·λ = 1 | s)` for a *single selected*
logical parity `a ∈ F_2^12`, aggregating all 4096 sectors into two halves. A full posterior
therefore needs all `2^12 - 1 = 4095` nonzero masks — that fan-out is what `tn_reconstruct/` does.

**`reconstruct_coset_probs` is the bridge.** Each queried marginal yields a ±1 moment
`m_a = P(a·λ=0) - P(a·λ=1)`. Those moments *are* the Walsh-Hadamard spectrum of the sector
distribution, so an unnormalized FWHT (`fwht_inplace`) inverts the marginalization and recovers
`P(λ|s)` for all 4096 sectors at once. Numerical negatives are clipped only within tolerance
(relative `1e-9` / absolute `1e-12`); a genuinely broken contraction raises instead of silently
becoming a small probability.

**Two access paths to the same decoder, and the fallback between them is load-bearing.**
`decoder.decode(...)` returns a normalized public probability.
`raw_coset_masses_from_decoder` instead reaches into `decoder.full_tn` / `logical_obs_inds` /
`contractor_config.contractor` and contracts directly to the raw masses `(z0, z1)`, which are
unnormalized and can be slightly negative. `query_flip_probability_with_fallback` tries public
first and falls back to raw on exception, non-convergence, or out-of-range, recording which path
won in `flip_prob_source` (`public` or `raw_fallback`) plus the rejection reason. When reading TN
results, treat `raw_fallback_count`, `public_failure_count`, `raw_clip_count`, and
`max_raw_relative_negative_mass` in `_summary.txt` as the run's health indicators — a run is only
clean if they are all ~0.

**Path caching.** Each distinct logical mask needs its own optimized contraction path, and
optimization costs far more than contraction, so paths are cached per mask. `build_cache_meta` pins
everything that would invalidate a cached path (error type, logical basis *and its SHA-256*,
contractor, dtype, check-matrix hash). `load_path_cache` treats **every** failure as non-fatal —
missing, unreadable pickle, wrong shape, stale metadata — returning an empty cache so the run just
re-optimizes. Entries are written via `atomic_pickle_dump` (temp file + replace) and are
round-tripped through `pickle.dumps` before acceptance, so an incompatible contractor path object
fails loudly at cache time rather than at load time.

The cached reconstruct script additionally repairs *invalid* cached paths: if a restored path
raises during decode it clears `path_single`/`slicing_single`, re-optimizes, rewrites the entry, and
counts `cache_invalid_rebuild_count`. This exists because a cached path can go stale between runs
(different CUDA-QX build, different GPU) without any metadata changing.

### 4. Entry-point scripts

- `examples/bb72_cudaqx_nv_tn_flip_prob_example.py` — single-syndrome, single-parity example, **and
  the shared function library imported by both `tn_reconstruct/` scripts** (`build_decoder`,
  `build_logical_observable`, `build_syndrome`, `resolve_logical_basis`, `pick_optimize_arg`, …).
  Editing a function here changes all three scripts. It raises `SystemExit` at import time when
  `cudaq_qec` is missing, so it cannot be imported at all on the Windows host.
- `tn_reconstruct/bb72_..._flip_to_coset_reconstruct.py` — uncached; one path optimization per mask.
- `tn_reconstruct/bb72_..._flip_to_coset_reconstruct_cached.py` — the main experiment. Writes three
  artifacts per run: `<stem>_flip_probs.csv` (one row per nonzero mask, with per-mask path source,
  timings, and fallback diagnostics), `<stem>_coset_probs.csv` (4096 sectors with posterior, rank,
  `is_map`, `is_true_sector`), and `<stem>_summary.txt`. Default path cache is
  `tn_reconstruct/path_cache/bb72_cudaqx_nv_tn_path_cache_<error_type>_<basis>_<contractor>.pkl`.
- `bp_osd_baselines/bb72_{qldpc,cudaqx_nv_qldpc}_bp_osd_baseline.py` — the two BP+OSD baselines.
  Same runner, same CSV schema, different backend. See `bp_osd_baseline_comparison.md` for the full
  diff; the practical traps are that the two use **incompatible parameter vocabularies** (qldpc uses
  names and `--bp-max-iter`, CUDA-QX uses integer enums and `--max-iterations`), and that CUDA-QX's
  `--no-osd` silently collapses the sweep to a single `[0]` order while qldpc always iterates the
  orders given.

### 5. `v0.5_final_production/` — frozen exact-BnB production baseline

**Self-contained, and not one of the four layers.** It imports *nothing* from `common/` or
`examples/`; its own modules import each other by bare name (`from exact_block_bnb import ...`), so
it must be run **from its own directory**. It also does not use the `tn_reconstruct/`
Walsh-Hadamard route — it reaches the same 4096 sectors by exact branch-and-bound, which is why it
needs neither 4095 marginals nor a path cache per mask.

**The method.** `exact_block_bnb.py` (pure NumPy, CPU) is `ExactBlockBnB`: a depth-first
branch-and-bound over the 12 logical bits with a *block* branching factor. One
`oracle(fixed_logicals, open_logicals)` call fixes a whole block of bits and returns an exact
`(2,)*len(open_logicals)` array of **subtree masses**, each the total over every sector beneath it.
Children are sorted by descending mass and pruned on `bound + local_guard <= incumbent` — so a
single root call is usually enough (the production `easy` case is **2 oracle calls, 4095 leaves
pruned**; `hard` is 4). Because pruning leans on children being sorted, the first prunable sibling
prunes all the rest.

Three numerical behaviours to preserve: the oracle must be exact and non-negative; tiny negative
cancellation residuals are clipped, but their *magnitude* is converted into a per-call
`local_guard_abs` used in that decision (`negative_guard_factor`, default 16x) rather than
competing with one hand-tuned global threshold; and a residual exceeding
`negative_abort_rtol * root_mass` raises `FloatingPointError` instead of being absorbed.

**The decoder.** `multi_logical_tensor_network_decoder.py` (1841 lines) subclasses CUDA-QX's
`TensorNetworkDecoder` rather than rewriting its TN machinery, and its docstring pins upstream
commit `3d75f56c469343d1a7e9ecb133dccb8ad11aeb99` (2026-09-16) — a different CUDA-QX build may
break it. The V4 detail still load-bearing: Quimb's index-collision mangling must stay **disabled**
when assembling the full network, or a physical-error index shared by two logical rows is silently
renamed and disconnected from the real error variable. `contract_logical_mass_fast` returns
*unnormalized* masses, so only ratios and parent/child sums are meaningful.

**Frozen config (P0-P5).** b9 identity layout `[0..8] | [9..11]`, pinned to
`profiles/canonical_paths_b9_0-1-2-3-4-5-6-7-8-9-10-11.pkl`; P2 constant cache ON, P3 GPU-resident
operands ON, P4 scratch OFF (measured as no benefit). P6 batching, P7 reuse, and P8 multi-GPU are
*deliberately absent* — do not merge them back in from the research archive; P0/P1/P5 carry the
measured wins. `--profile`'s `metadata.memory_limit` is compared against `--memory-limit` and only
**warns** on mismatch, so a mis-set memory limit shows up as a WARNING line, not an error.

**Reference gate.** `run_bb72_easy_hard.py` decodes `easy` (X error `[28]`) and `hard` (X error
`[0,27,30]`) at p=0.005 and **exits non-zero** unless both lambda and probability match. The hard
syndrome is genuinely three-fold ML-degenerate, so any of `001110011011`, `111011000000`,
`100110011001` is accepted. Wall times are reported but never gated (system- and GPU-dependent).

## The weight-6 logical basis

`WEIGHT6_LOGICAL_X_SUPPORTS` (in the example script) is a **hand-assembled** table of 12
minimum-weight (weight-6 = the code distance) X-type logical operators. `resolve_logical_basis`
selects it for Z-error decoding (`auto` → `weight6-x` for `Z`, `qldpc-dual` for `X`).

Motivation: `problem.dual_logicals` returns whatever the generic nullspace routine produces —
correct, but arbitrary in weight and often much heavier. Feeding minimum-weight representatives into
the TN contraction keeps each observable a sparse 6-qubit operator instead of a dense one.

Two things to know before touching it:

- **Nothing in the code validates the table.** `build_weight6_logical_x_basis()` will happily
  flatten an invalid table into a 12x72 matrix, and the Walsh-Hadamard reconstruction downstream
  will then silently produce a wrong posterior. Any edit must preserve linear independence and the
  commutation relation `B·u + A·v = 0`.
- It is X-type only, so `--error-type X` rejects it — which is exactly why `auto` falls back to
  `qldpc-dual` there.

`weight6_logical_supports.md` derives the three algebraic families, verifies the table numerically,
and shows it is a non-canonical choice (12 of 84 possible minimum-weight logicals).

## Conventions and gotchas

- **Repo-root `sys.path` bootstrap.** Every entry-point script inserts
  `Path(__file__).resolve().parents[1]` so that `from common... import` and `from examples... import`
  resolve. Moving a script between directories requires updating that and any relative imports.
- **BB72 qubit indexing**: `bb72_qubit_index(kind, a, b)` = (`0` for `"L"`, `36` for `"R"`) `+ 6*a + b`.
  L block first, then R. Used both to build the weight-6 basis and to render `L_{a,b}` / `R_{a,b}`
  support labels.
- **Contractor choice drives the optimizer argument**: `pick_optimize_arg` returns `None` for the
  `cutensornet` contractor (cuQuantum does its own pathfinding) and `"auto"` otherwise (quimb /
  opt_einsum). Don't hardcode either.
- **Reproducibility comes from `SeedSequence`.** The runner spawns one child seed per sweep point
  from the master seed, so results are independent of ordering and reproduce from `--seed` alone.
- CSV column order is fixed by explicit `fieldnames` lists in each writer. Adding a field means
  updating the writer *and* every row dict, or `DictWriter` will raise at write time.

## Verifying changes

There is no test suite for the four-layer code above. Use these instead, cheapest first:

```bash
python gpu_test.py 2              # CUDA-Q basics (needs `cudaq`, not `cudaq_qec`)
python repetition_code.py         # 3-qubit hand-checked example; prints decoder vs brute force
python qrc311_cudaqx_nv_tn_flip_prob_example.py   # [3,1] repetition code, TN decoder
```

The one real test suite is `v0.5_final_production/test_exact_bnb.py` — 3 pytest cases over a
synthetic oracle, covering global-ML recovery, degenerate-ML acceptance, and the tiny-negative-
residual guard. It is pure NumPy and needs no GPU, so it is the cheapest check available; run it
from that directory with `pytest -q test_exact_bnb.py`. None of it exercises the TN decoder, so
for v0.5 the real end-to-end gate is `run_bb72_easy_hard.py`, which self-checks against the frozen
reference probabilities and exits non-zero on mismatch.

`qrc311_cudaqx_nv_tn_flip_prob_example.py` (in `examples/`) is the best first run after touching the
TN path: it is tiny, and it cross-checks the tensor-network masses, the public decode API, the batch
API, and an exhaustive brute-force enumeration against each other, printing absolute errors. If a
change breaks the TN plumbing, this fails loudly.

For `common/` changes, the `qldpc` baseline is the CPU-only end-to-end check — it exercises
`DecodingProblem`, `CodeCapacityRunner`, the noise models, and the CSV schema without needing a GPU.

## Reference documents

- `bp_osd_baseline_comparison.md` — detailed diff of the two BP+OSD baselines, including the
  behavioural edge cases, the OSD-0 guard asymmetry, and the `-o` filename bug.
- `weight6_logical_supports.md` — the algebra behind the weight-6 table, its three families, the
  meet-in-the-middle count of minimum-weight logicals, and the editing caveat.
- `v0.5_final_production/README.md` — the frozen production config, the P0-P5 definitions, what was
  deliberately stripped out, and the reference smoke values.
