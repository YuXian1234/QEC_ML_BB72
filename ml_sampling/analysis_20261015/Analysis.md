# BB72 ML Sampling Audit

Analysis date: 2026-10-15, Asia/Singapore.

## Outcome

The scan covered all 105,972 planned `(syndrome, target p)` tasks. The database
contains 104,080 `done`, 1,882 `review`, and 10 `failed` records. The ML export
contains exactly the 104,080 `done` records, not the numerical exceptions.
The quality-pass rate is 98.2146%, NOT logical decoding accuracy.

No production code, checkpoint, matrices, or export was changed. SHA-256 hashes
of all five source artifacts were checked before and after the audit.

## Integrity and Definitions

- SQLite integrity and foreign-key checks passed; no missing or duplicated
  `(syndrome_id, p_index)` tasks were found.
- All 47,629 source error vectors were independently multiplied by the saved
  H_Z and L_Z matrices. Every stored syndrome, true sector, and error weight matched.
- GF(2) ranks: H_Z=30, H_X=30, L_Z=12, L_X=12. CSS commutation and logical
  duality checks passed. The 36-bit syndrome representation has redundant checks.
- Manifest, SQLite identity and summary fingerprints agree.
- Every exported row was checked against SQLite: input, prior, sector bits,
  little-endian class index, probability/raw probability/correction, split,
  natural frequencies and source frequencies all agree.
- Each syndrome has the same split across priors. The grouped split has no
  syndrome leakage. There is no complete 4096-sector posterior in these files.
- Every completed contraction reports a compiled path and no decoder fallback.
  Matching saved records does not independently certify their mathematical MAP answer.

## Quality by Target Prior

Each row below has 17,662 tasks.

| p | Done | Review | Failed | Nonvalid fraction | Median online seconds |
| --- | ---: | ---: | ---: | ---: | ---: |
| 0.001 | 16,719 | 933 | 10 | 5.3391% | 0.14887 |
| 0.002 | 17,153 | 509 | 0 | 2.8819% | 0.14917 |
| 0.003 | 17,359 | 303 | 0 | 1.7155% | 0.14917 |
| 0.005 | 17,562 | 100 | 0 | 0.5662% | 0.14916 |
| 0.007 | 17,628 | 34 | 0 | 0.1925% | 0.14909 |
| 0.010 | 17,659 | 3 | 0 | 0.0170% | 0.14912 |

All 1,882 review records exceed the relative-negative-mass threshold `1e-10`.
Of these, 1,053 also exceed the relative-guard threshold `1e-8`; these counts
overlap. No current review is solely a probability-boundary violation.
The one corrected zero-syndrome p=0.001 probability is preserved as
`1.0000000000000018`, with training value 1 and an explicit correction flag.

At p=0.001, review relative negative masses range from 1.007e-10 to 9.937e-7,
with median 1.933e-9. Thus not every review is equally severe, but negative mass
alone is not a proven bound on the positive sector errors or on MAP decisions.
Do not indiscriminately loosen thresholds to include these labels.

## Which Samples Are Affected?

All weight-0/1/2 enumerated syndrome labels pass at all six priors.
For p=0.001, grouping each syndrome by the weight of its FIRST generating
error gives:

| First generating weight | Syndromes | Review | Failed | Nonvalid fraction |
| --- | ---: | ---: | ---: | ---: |
| 0-2 | 2,629 | 0 | 0 | 0% |
| 3 | 5,034 | 22 | 0 | 0.4370% |
| 4 | 5,003 | 189 | 3 | 3.8377% |
| 5 | 4,995 | 722 | 7 | 14.5946% |
| 6 | 1 | 0 | 0 | 0% |

This is the first OBSERVED generating weight, not a proof of minimum physical
error weight. The ten hard failures are pressure examples: three generated at
weight 4 and seven at weight 5, all at p=0.001.

Syndrome IDs 8866, 11426 and 12004 have NO valid label at any of the six priors;
they are also exactly the three review cases at p=0.01. All three come from
weight-4 pressure errors and are useful first replay targets.

Every natural-noise record passes when decoded at its OWN generating prior:
six groups of 5,000, all 30,000 numerically valid. One naturally generated
syndrome is review only when relabeled at p=0.001. The actual natural-draw
teacher/true-sector disagreements at the matching prior are 0,0,0,2,1,2 for
p=0.001,0.002,0.003,0.005,0.007,0.01. These are not numerical task failures,
not ML-model test results, and not independent large-sample accuracy guarantees.

The pattern is consistent with cancellation/conditioning problems when very
unlikely syndromes are queried under small priors. This is an inference from
diagnostics, not an identified cuTensorNet implementation bug.

## ML Readiness and Limitations

- NPZ rows: 104,080, representing 17,659 different syndromes.
- 16,706 syndromes have all six valid priors; 956 lack at least one valid label.
  Three have no valid labels. Never fill missing targets with a higher-p MAP label.
- Unique-source counts (overlapping, not additive): enumeration 2,629;
  pressure 14,781; natural 1,424. Most of the expanded coverage comes from pressure.
- Grouped export splits: training 83,315 rows; validation 10,577; unseen-syndrome
  test 10,188.
- Valid labels cover 2,897 of 4,096 possible sectors; training covers 2,685.
  Validation has 126 classes absent from training (728 rows), and test has 106
  absent classes (676 rows). A flat 4,096-class classifier has a genuine
  unseen-class coverage issue; do not remove syndrome grouping just to improve scores.
- 673 syndromes have different selected sectors across their available valid
  priors. Without common-reference posteriors, this does not distinguish true
  prior-dependent MAP changes from degenerate/near-tie selection.
- `done` means the current quality checks passed, not a rigorous interval
  certificate or independent verification of every teacher label.

The current export is usable as a documented preliminary training dataset,
but the small-p pressure subset has systematic selection bias from exclusions.
Use a separate natural-noise evaluation and report pressure generalization
and label-coverage metrics separately.

## Timing

Completed contractions have mean online latency 0.1534-0.1537 s by p, with
medians near 0.149 s. Cumulative recorded online time is 4.5185 h across sessions.

The resumed session ran about 11 h 14 min 30 s, with 4.1495 h of newly recorded
online decoding. About 7.09 h is outside that timing scope. Task-finish timestamps
show a median inter-task interval increasing from 0.1594 s in session 1 to
0.4061 s in session 2, while decoder time remains near 0.149 s. The median
interval minus the next decode duration increases from 0.0115 to 0.2550 s.

This localizes much of the slowdown outside online BnB contraction. Synchronous
SQLite checkpoints/shared-filesystem I/O are priority suspects, not proven
causes without separate commit/summary/initialization timing. Do not call the
entire job wall time GPU contraction time or change durability settings blindly.

## Next Steps

1. Freeze and retain this raw dataset and the quality-filtered NPZ as version 1.
2. Replay the three all-prior problematic syndromes and ten hard failures using
   old/new profiles plus a common full-posterior reference. Keep replay isolated
   from the existing database. The union is only 11 syndromes.
3. Sample additional borderline and severe review cases; compare selected
   sectors under the SAME reference posterior. Near-ties need explicit treatment.
4. For unresolved cases, investigate more stable tensor formulations or suitable
   higher-precision references rather than merely increasing negative tolerances.
5. Add targeted timing instrumentation in a separate branch if throughput matters.
   The current checkpoint and production implementation were not changed by this audit.

## Audit Artifacts

- `audit_ml_sampling.py`: reproducible, CPU-only, read-only audit.
- `audit_metrics.json`: exact counts, per-prior timing/quality, class coverage,
  session statistics and original-file SHA-256 fingerprints.
- `problem_labels.jsonl`: all 1,892 review/failed tasks, diagnostics, syndromes
  and their generating error occurrences for focused replay.
