# BB72 small-p ML sampling

This is a data-collection tool, not a new decoder. It imports the unchanged
`../v0.5_p0` production runtime and defaults to its `profiles/best_93.pkl`.
Copy both sibling directories to the cluster. All default dependency paths
are resolved from this script, not from a Windows or cluster absolute path.

## Problem and label definition

- BB72 `[[72,12,6]]`, iid physical X errors, perfect syndrome measurements.
- `s = H_Z e mod 2`; the sampled true sector is `L_Z e mod 2`.
- Teacher: float64 cuTensorNet, identity logical order, b9 root + b3 tail,
  P2/P3 enabled, retained scratch disabled, unchanged production BnB.
- Training input: **syndrome and target prior p**. Training label: teacher MAP
  logical sector, NOT the sector of the error used to generate that syndrome.
- Vector element zero is lambda_0; integer sector index is
  `sum(lambda_i * 2**i)`. Do not silently use a big-endian binary conversion.
- These are code-capacity labels, not circuit-level Quantinuum noise labels.

## Default dataset

| Source | Generation | Records before deduplication |
| --- | --- | ---: |
| enumerate | All physical errors of weights 0, 1, 2 | 2629 |
| stress | 5000 draws each at weights 3, 4, 5 | 15000 |
| natural | 5000 iid draws at each target p | 30000 |

Target grid: `0.001,0.002,0.003,0.005,0.007,0.01`.

Total generated records: **47629**, not independent unique syndromes.
Natural draws mostly overlap the enumerated low-weight coverage. The tool
first generates the complete configured input set, deduplicates syndromes,
and prints the actual unique count M. Full labeling requires 6M tasks.
Generating a pressure example does not imply labeling it at a high p.
All generating errors, possibly different true sectors, and natural frequencies
are retained. Equal-weight pressure sampling is not a natural-noise LER test.

Source order is fixed: enumerate, stress, natural. Enumeration uses weight then
lexicographic support order. RNG streams are independent for each pressure
weight and each natural p; labels are visited p-major then syndrome-ID order.

## Dependencies and devices

Use the working Linux `cudaqx-qec` environment. Dataset preparation needs
NumPy, SymPy, and standalone **qldpc==0.3.3**. Other qldpc versions are rejected
because their logical basis may not match the frozen production profile.
GPU labeling additionally needs CUDA-enabled Torch, CUDA-QX QEC, CuPy and
cuQuantum/cuTensorNet, plus the production decoder's tensor-network dependencies.
No package installation or production code modification is performed here.

Generation/deduplication/export and unit tests run on CPU. Labeling requires
CUDA; CPU fallback and projector fallback are forbidden. Every prepared
topology must report `path_source=compiled_profile`. A profile that fails and
triggers the production loader's runtime path search is rejected, not accepted
as a benchmark of the requested path. Only load trusted local pickle profiles.

## First GPU smoke run

From `ml_sampling/` on an allocated A100, after activating the environment:

```bash
python sample_bb72_dataset.py \
  --max-syndromes 10 \
  --output-dir results/bb72_ml_small_p
```

This prepares all configured CPU inputs but labels only the **first 10 unique
syndromes at SIX p values: 60 tasks**. It is a functional smoke test, not a
representative random benchmark. To test pressure examples specifically:

```bash
python sample_bb72_dataset.py --sources stress --max-syndromes 10 \
  --output-dir results/pressure_smoke
```

For a one-p smoke run, use a separate experiment:

```bash
python sample_bb72_dataset.py --p-grid 0.005 --max-syndromes 10 \
  --output-dir results/single_p_smoke
```

## CPU preparation and full scan

```bash
python sample_bb72_dataset.py --prepare-only \
  --output-dir results/bb72_ml_small_p

# Grow the same checkpoint without changing any data-generation settings:
python sample_bb72_dataset.py --max-syndromes 100 \
  --output-dir results/bb72_ml_small_p

# No limit means all generated unique syndromes:
python sample_bb72_dataset.py --output-dir results/bb72_ml_small_p
```

Options include `--enum-max-weight 0..3`, `--stress-weights 3,4,5`,
`--stress-shots-per-weight`, `--natural-shots-per-p`, `--sources`, `--seed`,
`--split-seed`, `--profile`, and `--production-dir`.

`--max-syndromes` limits labeling, not CPU generation. Increasing the limit is
safe on resume. Changing the p grid, source selection, generation counts, seeds,
matrices, numerical thresholds, profile hash or core code requires a NEW output
directory. Choosing a different profile in the same database is deliberately
rejected, to prevent mixing incompatible teacher labels.

## PBS submission (ASPIRE 2A/A100)

The PBS template follows the existing A100 production job, not the ASPIRE 2A+
Enroot/H100 setup. Check project/queue resources for your account. Submit from
this directory; stdout/stderr goes directly to `output-bb72-ml-sampling.txt`.
The current diagnostic template temporarily requests the previously successful
`ncpus=32:ngpus=1:mpiprocs=1:ompthreads=32` allocation. Sampling still sets
`OMP_NUM_THREADS=1`. It checks `nvidia-smi` and Torch CUDA before generating
inputs; this allocation does not itself guarantee CUDA driver initialization.
After the GPU smoke succeeds, a smaller CPU allocation can be tested separately.

```bash
qsub submit_sampling.txt                          # first 10 syndromes, 60 labels
qsub -v MAX_SYNDROMES=100 submit_sampling.txt      # same output, expand to 100
qsub -v MAX_SYNDROMES=all submit_sampling.txt      # full scan / resume
```

Default `OUT_DIR=results/bb72_ml_small_p` is stable across submissions.
`--max-seconds 84600` finishes the current task and checkpoints before the 24 h
limit when possible. It is a soft stop, not a guarantee if a contraction itself
takes longer than the remaining allocation. A partial labeling run exits 2;
the database is still resumable. SIGINT/SIGTERM request the same graceful stop.

## Preflight and numerical quality

Each GPU session tests the original easy `[28]` and hard `[0,27,30]` errors at
**p=0.005**, even when the requested grid is different. The hard case accepts
any of the three known degenerate ML sectors. Known probabilities are never
incorrectly applied to other priors. Each target-p decoder also warms its
root/tail network before online labeling.

Startup/reference checks failing stops the scan. Per-task records contain
compiled-path source, masses, oracle counts/timings, negative clipping,
guard/root ratio, and fallback metadata. Defaults:

- relative negative mass <= `1e-10`;
- relative local pruning guard <= `1e-8`;
- probabilities finite and inside [0,1];
- no recorded memory-pressure fallback.

Quality policy v2 permits a probability outside [0,1] by at most `1e-12`
ONLY when all other checks pass. Its training/export value is clipped to the
boundary; `best_probability_raw`, correction flag/delta and a quality note
preserve the actual float64 output. Larger excursions and other diagnostics
remain `review`. This does not change BnB, masses, sectors, or contraction paths.

Small benign negative clips can remain `done`, with diagnostics retained.
Threshold violations/fallbacks are `review`, excluded from the default ML
export. Exceptions are `failed`, never made into fabricated labels. These
checks are NOT an interval certificate, a universal correctness proof, or a
near-tie detector. Complete 4096-sector posterior/top-2 gap is NOT saved;
ambiguous cases need a separate full-posterior reference replay later.

## Resume and recovery

SQLite is authoritative. Every completed label commits separately; generation
commits every 250 records. A hard crash can lose only the uncommitted generation
batch or current label, which is deterministically recomputed. `running` labels
become pending on restart. `done` labels are skipped. `review` and `failed`
labels are retained and skipped unless explicitly selected for retry:

```bash
python sample_bb72_dataset.py --output-dir results/bb72_ml_small_p \
  --retry-failed
```

Use `--retry-review` for review records. By default a task exception stops the
job after checkpointing when invoking Python directly. The PBS template enables
`--continue-on-error` by default: it records failures, excludes them from ML
exports, and rebuilds the decoder before proceeding. Set `CONTINUE_ON_ERROR=0`
at submission to restore fail-fast behavior. Profile-path violations always stop
the job; GPU initialization errors also remain fatal. A scan with failed labels
reports `success=false` even if all other tasks were attempted.

For example, a negative root-prefix mass exceeding BnB's `1e-6 * root_mass`
abort threshold is NOT the tiny probability-boundary issue repaired by policy
v2. Do not loosen this threshold or replace the failed target with a guessed
label. Resume the remaining tasks first:

```bash
qsub -v MAX_SYNDROMES=all submit_sampling.txt
```

Previously failed tasks are skipped unless `RETRY_FAILED=1` is requested.
Their errors and generating physical patterns remain in SQLite for focused
numerical replay. Missing difficult labels can bias a training set, so audit
failure/review rates by prior and source before using the final export. Never
interpret the valid-label subset as unbiased low-p decoding accuracy.

Only ONE writer/job may use a result directory. An active job is never removed
by `--recover-lock`. After SIGKILL, node failure, or wall-time killing, confirm
the old job is stopped before:

```bash
python sample_bb72_dataset.py --output-dir results/bb72_ml_small_p \
  --recover-lock
# PBS equivalent, only after confirming no old job is using this directory:
qsub -v MAX_SYNDROMES=all,RECOVER_LOCK=1 submit_sampling.txt
```

The first labeling environment is frozen; a changed GPU model/Python/package
stack cannot silently mix new numerical labels into the same experiment.
The profile p=0.005 metadata describes its tuning workload, not a restriction
on target priors: path/slicing topology is reused, noise operands rebuilt per p.

### One-time upgrade of the original 59-done / 1-review checkpoint

Upload the updated `sample_bb72_dataset.py`, `sampling_utils.py` and
`submit_sampling.txt`. Do NOT remove the cluster result directory. With no job
writing the directory, run this on the login node (NumPy is sufficient):

```bash
python sample_bb72_dataset.py --export-only --upgrade-roundoff-checkpoint \
  --output-dir results/bb72_ml_small_p
```

The upgrade accepts only the known original sampler hashes, with unchanged
matrices, generation settings, target grid, teacher code/profile and other
quality tolerances. It first creates `samples.pre_roundoff_v1.sqlite` and
`run_manifest.pre_roundoff_v1.json`, then transactionally rechecks stored labels,
records their before/after contents in `label_quality_audit`, upgrades the
identity/manifest, and exports the data. It never runs TN or increments attempts.
The probability `1.0000000000000018` becomes training probability 1, while the
raw value remains available. Repeating the upgrade is idempotent. Unknown code
versions or unrelated configuration changes are refused, not blindly trusted.

After the upgrade, continue normally:

```bash
qsub -v MAX_SYNDROMES=all submit_sampling.txt
```

Alternatively, `qsub -v MAX_SYNDROMES=all,UPGRADE_CHECKPOINT=1 submit_sampling.txt`
performs the same migration before continuing GPU labeling. Prefer the CPU-only
upgrade first so its report can be inspected. Existing valid labels are reused.

## Output files

- `run_manifest.json`: immutable configuration/hashes, matrix fingerprint,
  source paths and trusted profile metadata. Paths are provenance, not runtime
  dependencies on the original host.
- `matrices.npz`: H_Z/H_X/L_Z/L_X, loadable with `allow_pickle=False`.
- `samples.sqlite`: unique syndromes; every source occurrence and its actual
  true sector; label tasks, status/attempts/results/errors; GPU sessions and
  preflight results; generation cursor/config identity.
- `sampling_summary.json`: source/unique counts, selected scope, expected
  labels, pending/failed/review counts, timing, export count. `success` means
  selected tasks finished without failed/pending tasks; `validation_passed`
  additionally requires every selected label to have `done` status.
- `ml_dataset.npz`: default export contains ONLY `done` labels. Export occurs
  on normal/graceful/error exit. `--export-every 1000` also refreshes during a run.

## ML loading

```python
import json
import numpy as np
import torch

with np.load("results/bb72_ml_small_p/ml_dataset.npz", allow_pickle=False) as d:
    X = np.column_stack([d["syndrome"], d["p_target"]]).astype(np.float32)
    y = d["map_sector_index"].astype(np.int64)  # 4096-way joint-sector target
    train = d["split_id"] == 0
    X_train = torch.from_numpy(X[train])
    y_train = torch.from_numpy(y[train])
    metadata = json.loads(str(d["metadata_json"]))
```

NPZ arrays:

| Key | Shape | Meaning |
| --- | --- | --- |
| syndrome | N x 36 | model input, uint8 |
| p_target / p_index | N | target prior / grid index |
| syndrome_id | N | stable grouping key |
| map_sector_bits | N x 12 | lambda vector |
| map_sector_index | N | documented little-endian class index |
| map_probability | N | teacher's probability of its selected sector |
| map_probability_raw | N | original float64 probability, possibly tiny overshoot |
| probability_boundary_corrected / probability_boundary_delta | N | explicit boundary-correction provenance |
| split_id | N | 0=train, 1=validation, 2=unseen-syndrome test |
| quality_flags | N | JSON text, no pickle/object arrays |
| natural_frequency | N x 6 | input frequency at each GENERATING natural p |
| source_frequency | N x 3 | enumerate/stress/natural occurrence counts |
| metadata_json | scalar | grid, indexing, split and label-quality conventions |

The natural-frequency second dimension follows the configured grid, not always
six. Source frequencies repeat across the six labels of the same syndrome.
They are not automatically normalized training weights. Error weight/support
is diagnostic truth, NOT an available decoder input. Estimated MAP sector
probability is NOT a full soft target over 4096 classes.

Splits use a stable hash of the syndrome bits (80/10/10 expected proportions).
All p labels for the same syndrome stay together. Very small smoke exports may
have empty splits. The grouped test is a generalization test, NOT an unbiased
natural-noise LER estimate. For LER, generate independent iid test errors at
the target p, retaining their real multiplicities and true sectors. Do not use
pressure-set accuracy or the MAP teacher labels alone as an LER measurement.

To re-export completed labels on CPU without CUDA-QX/qldpc:

```bash
python sample_bb72_dataset.py --export-only \
  --output-dir results/bb72_ml_small_p
```

Export-only reads the saved experiment configuration. `--max-syndromes` can
restrict the export. `--include-review` is explicit opt-in and preserves flags.

## CPU tests

```bash
python -m unittest discover -s . -p 'test_*.py' -v
```

Tests use small synthetic matrices and a clearly fake GPU teacher, including
generation resume, duplicate/ambiguous true labels, expanding task limits,
failure retry, config rejection, grouped exports, and GPU-free prepare/export.
They do not validate real BB72 contractions. Run the cluster GPU smoke first.
