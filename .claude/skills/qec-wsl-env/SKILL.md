---
name: qec-wsl-env
description: Activate the q_env311 virtualenv in WSL before running any QEC script. Use this whenever running a script from this repo — examples/, tn_reconstruct/, bp_osd_baselines/, gpu_test.py, repetition_code.py, or anything importing cudaq_qec, cudaq, or qldpc. The venv lives at ~/q_env311 and scripts fail on import without it. Also use when the user pastes a WSL prompt or reports a missing-module error.
---

# Running QEC scripts under WSL + q_env311

Every script in this repo runs in **WSL2**, inside the virtualenv **`q_env311`**
(`~/q_env311`). Activate it in the *same* shell invocation as the script.

## The command shape

This session's shell is Windows Git Bash, so go through `wsl.exe`:

```bash
wsl.exe bash -lc 'cd /mnt/c/QEC && source ~/q_env311/bin/activate && python examples/bb72_cudaqx_nv_tn_flip_prob_example.py --error-type Z --p 0.02 --seed 20260827'
```

The `source ... && python ...` must be **one command string**. Running activation and the
script as two separate Bash calls loses the environment — activation is per-invocation and
does not persist between tool calls.

For a quick sanity check inside the venv:

```bash
wsl.exe bash -lc 'source ~/q_env311/bin/activate && python -c "import sys; print(sys.executable)"'
```

Expected output (verified 2026-10-01): `/home/gohyuxian/q_env311/bin/python`, Python 3.11.16.
If it points at `/usr/bin/python3`, activation silently failed — stop and fix that before
reading any script output.

## What needs this

All in-repo entry points, including the CPU-only ones (they still need `qldpc`/`numpy`
from the venv):

- `examples/bb72_cudaqx_nv_tn_flip_prob_example.py`
- `examples/qrc311_cudaqx_nv_tn_flip_prob_example.py` — the cheapest TN cross-check
- `tn_reconstruct/bb72_cudaqx_nv_tn_flip_to_coset_reconstruct.py` (and `_cached.py`)
- `bp_osd_baselines/bb72_cudaqx_nv_qldpc_bp_osd_baseline.py`
- `bp_osd_baselines/bb72_qldpc_bp_osd_baseline.py`
- `gpu_test.py`, `repetition_code.py`

## What does *not* use it

The **Windows host**. Per `CLAUDE.md`, the repo-root `.qenv/` is essentially empty and
`cudaq_qec` aborts on native Windows by design, so never retry a failed script with
Windows Python — that failure is expected and tells you nothing. If a run "fails to import
cudaq_qec", the fix is the `wsl.exe` form above, not a different interpreter.

## Working directory

Run from `/mnt/c/QEC` (the repo root). Every entry point inserts its own repo root onto
`sys.path`, so imports of `common.*` / `examples.*` work from anywhere — but
`default_results_dir()` resolves `results/` relative to the *calling script's* file, so
output lands per-directory regardless. Run from the script's own directory when you want to
match a PBS run's layout.

## Note on CLAUDE.md

`CLAUDE.md`'s "Environment" section describes a **conda env named `cudaqx-qec`** as the
Linux path. That describes the PBS/cluster workflow; the working *local WSL* path the user
actually uses is this `q_env311` venv. Prefer the venv for local WSL runs, and don't
assume `conda activate cudaqx-qec` works on this machine.

## Long runs

The full 4096-mask reconstruction is a GPU job measured in hours. Pass
`run_in_background: true` rather than blocking on it, and reuse `--path-cache` across runs
so the contraction-path optimization is paid once.
