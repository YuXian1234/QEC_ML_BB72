# Repository Guidelines

## Project Structure & Module Organization

`common/` contains BB72 construction, shared decoder utilities, and the CPU-capable Monte Carlo runner. `examples/` holds CUDA-QX examples and shared tensor-network helpers. `tn_reconstruct/` reconstructs the full logical-sector posterior; `bp_osd_baselines/` contains the `qldpc` CPU and CUDA-QX BP+OSD implementations. `v0.5_final_production/` contains the frozen BB72 exact logical-sector branch-and-bound baseline, its multi-logical tensor-network decoder and canonical b9 profile, the production smoke runner and PBS submission, plus CPU-only BnB tests and smoke results. Root scripts (`repetition_code.py`, `gpu_test.py`) are small environment checks. Experiment outputs and submission scripts live beside their entry points.

Keep backend-independent code in `common/`; only entry points should import `cudaq_qec`. The CUDA-QX scripts require Linux with the `cudaqx-qec` environment. The root `.qenv` is not a usable CUDA-QX environment.

## Build, Test, and Development Commands

There is no package build or project-wide test command. Run scripts from their directory when possible; outputs default to that directory’s `results/` folder.

- `python repetition_code.py` — check the small hand-worked decoder example.
- `python bp_osd_baselines/bb72_qldpc_bp_osd_baseline.py --osd-orders 0,2,4` — exercise the CPU baseline and shared runner.
- `python gpu_test.py 2` — check basic CUDA-Q availability (requires `cudaq`).
- From `v0.5_final_production/`, `pytest -q test_exact_bnb.py` — run the CPU-only exact BnB logic tests.
- From `v0.5_final_production/` in the Linux `cudaqx-qec` environment, `python run_bb72_easy_hard.py --warmup 1 --repeats 5 --output-dir production_smoke_results` — run the frozen production BB72 smoke cases.
- `qsub examples/submit_tn_decoder.txt` — submit the CUDA-QX example on the configured PBS system; corresponding submission files are in `tn_reconstruct/` and `bp_osd_baselines/`.

For tensor-network changes, run `examples/qrc311_cudaqx_nv_tn_flip_prob_example.py` in the Linux CUDA-QX environment; it compares contraction results with brute force.

## Coding Style & Naming Conventions

Follow neighboring Python code: four-space indentation, type annotations, `snake_case` functions and variables, and `CapWords` classes. There is no configured formatter or linter. Keep CSV schemas explicit and update both field lists and row dictionaries when adding output columns. Preserve the repo-root `sys.path` bootstrap in entry-point scripts that import `common`.

## Testing Guidelines

No project-wide automated test suite or coverage requirement is configured. Use the relevant checks above; GPU-dependent checks cannot run on the Windows host. For changes to shared CPU code, the `qldpc` baseline is the end-to-end check. For tensor-network code, use the small QRC example before a longer BB72 experiment. The production folder's BnB logic test is CPU-only; its smoke benchmark requires the Linux `cudaqx-qec` environment.

## Commit & Pull Request Guidelines

Git metadata and commit history are absent, so no repository-specific commit convention can be confirmed. Use a short imperative commit subject (for example, `Clarify baseline output schema`). In a pull request, explain the affected decoder or experiment, list commands and environment used for validation, and attach or link result summaries when output behavior changes.
