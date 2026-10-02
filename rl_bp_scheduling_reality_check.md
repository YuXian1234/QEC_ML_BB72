# RL-scheduled BP for QLDPC — benchmark reality-check

Target: **arXiv:2609.20236**, Moradi / Kim / Chou (ASU, UT Arlington), *Conflict-Free Color-Clustered
Sequential Belief-Propagation Decoding of Quantum LDPC Codes via Reinforcement Learning*, v1
2 Sep 2026, and the RL-S lineage it builds on.

Produced by a deep-research run on 2026-10-02 (5 search angles, 15 sources fetched, 75 claims
extracted, 25 adversarially verified 3-vote, 24 confirmed, 1 killed, 7 after synthesis).
Companion to `ml_decoder_literature_survey.md` (2026-10-01).

## Bottom line

The method is real and the claims are internally consistent, but the performance story is thin and
the headline number is not what it sounds like.

- The **26.2×** is a *scheduling-depth count*, not wall-clock. It says one BP pass needs ≤11
  sequential color-layer updates instead of 288 VN updates. No runtime, throughput, or latency
  measurement appears anywhere in the paper.
- Everything is **code-capacity depolarizing**, in the **p = 0.03–0.1** regime. No circuit-level
  model.
- Every quantitative comparison is **read off a figure**. There is no BLER table, no confidence
  interval, and no absolute number in the text.
- The **Q-table is not trained here** — it is reused verbatim from the predecessor
  `arXiv:2603.10192` (Moradi, Nourozi, Habib, Mitchell), a different author group.
- **No implementation exists publicly**, and no independent replication of any result in the line
  was found.
- **No BB72, no small BB code.** Smallest tested is `[[144,12,12]]`.

## What the method does

Build a VN conflict graph — VN `i`, `i'` adjacent iff they share an X-type or Z-type check, i.e.
from `1{H_XᵀH_X + H_ZᵀH_Z > 0}` minus self-loops. Properly color it. Per BP iteration, the trained
VN-level Q-table picks a **seed** VN `v* = argmax Q(σ_v, v)`; the whole remaining same-color class
is then updated in parallel using the same pre-batch messages, and removed from the candidate set.

- vs **flooded BP**: flooded updates everything at once with no learned ordering; this keeps the
  learned ordering at color-class granularity.
- vs **VN-by-VN RL-S**: identical policy, but a batch per step instead of one VN per step.

The coloring guarantee is stated for Tanner **4- and 6-cycles** only. The paper concedes an 8-cycle
can place two of its VNs in the same batch.

## Claim status

| Claim | Verdict | Note |
|---|---|---|
| Method description (conflict graph, coloring, seed selection, pre-batch messages) | **high confidence** | Verbatim from the paper; 3-0 |
| 288 VNs → 11 colors, 26.2× depth reduction on `[[288,12,18]]` | **high confidence** | Depth count only; no wall-clock claim exists |
| Code-capacity depolarizing, no circuit-level model | **high confidence** | 3-0 |
| Gains are relative order-of-magnitude, figure-read, no table | **high confidence** | 3-0; grep of full text found no BLER table |
| Q-table reused from `2603.10192`, not trained here | **high confidence** | Paper says "the same trained VN-level RL-S Q-table as in [14]"; 3-0 |
| No `[[72,12,6]]`, no small BB code anywhere in the line | **high confidence** | Zero occurrences of any 72-qubit code in any source |
| No public code for any paper in the line | **medium** | Absence of evidence, not proof — search budget was exhausted |
| "At T=100, comparable to BP-OSD-10" | **figure-read** | No gap, no error bar, no compute-matching |

Killed (1-2): a claim that `arXiv:2607.20130` (the cluster paper) trains its own Q-table offline.
Its training/reuse status is genuinely contested — unlike the color-clustered paper's reuse, which
is verbatim and unanimous.

## Lineage and peer-review status

```
2602.13420  SCNS/SVNS — fixed-order (non-RL) sequential schedules   IEEE ICC 2026   credible
2603.10192  RL-SVNS / RL-QSVNS — offline tabular Q scheduling      preprint        unrefereed
2607.20130  cluster RL-S over a FIXED RANDOM VN partition          ITW 2026        credible
2607.24891  RL-S2LU — inference-time second-order local updates    preprint        unrefereed
2609.20236  conflict-free color-clustered (this paper)             QCE 2026        weakly witnessed
```

The clustering/coloring step **is** a genuine novelty: `2607.20130` states its partitions are
random and not conflict-free, and explicitly defers graph-aware partitioning "to future work" —
exactly what `2609.20236` supplies.

QCE'26 is **weakly witnessed**: it appears on co-author Rémi Chou's own conference page, with no
IEEE Xplore DOI or program entry independently located by the run.

A MERL 2026 presentation critiques this line's baseline methodology — curves re-plotted from prior
work without common error frames or matched computational budgets. Not a replication, but it is a
published objection to exactly the comparison style used here. Related: iteration budgets are
asymmetric throughout (baselines at T=1000 vs RL-S2LU at T=10/100/1000), so no "X-fold
improvement" in this line is compute-matched.

## Code availability

No repository located for `2609.20236`, `2607.20130`, `2607.24891`, or `2603.10192`. The target
paper's full text has **zero** hits for github/repository/availability/implementation — no
code-availability statement at all, so this is "no repository found," not an explicit refusal. An
author-held implementation cannot be ruled out.

(The run also surfaced a Zenodo artifact, `10.5281/zenodo.21466148`, Hung N. Dang, 21 Jul 2026 —
**not** this paper. It is a DQN that *selects among* four decoders on a `[[625,25]]` code, and its
result is a cautionary negative: the DQN did not beat a hindsight-tuned threshold rule.)

## Applicability to BB72 `[[72,12,6]]`

**Pure extrapolation downward.** Codes tested across the whole line: `[[288,12,18]]`,
`[[144,12,12]]`, `[[180,10,15≤d≤18]]` A5, `[[882,24,18≤d≤24]]`, `[[882,48,16]]`,
`[[1922,50,16]]` C2. The smallest BB code anywhere is `[[144,12,12]]` — 2× this repo's blocklength,
and no 72-qubit code appears in any source.

There is a plausible mechanistic hook: the coloring targets exactly the short-Tanner-cycle and
shared-check conflicts that are BB72's motivating failure modes. But shorter cycles, higher
degeneracy, smaller blocklength and only 12 logicals are all untested. **Do not scale
`[[288,12,18]]` numbers down to `[[72,12,6]]`.**

## The missing experiment — runnable here

**No source in the corpus compares RL-scheduled BP against exact ML or tensor-network ML decoding.**
The comparison in the literature is confined to RL-BP vs flooded BP, BP-OSD-0/10, and SVNS.

Since BP is approximate message passing, exact ML upper-bounds what any schedule can reach — and
this repo already has both sides: a GPU tensor-network decoder and an exact branch-and-bound MLE
over the same 4096 logical sectors. The fair test (same syndromes, matched noise, matched compute,
measured BLER against the exact posterior) has not been run by anyone, and can be run here. That
would settle whether scheduling buys anything an exact decoder does not already give at this size.

## Adjacent sources surfaced (BB72-relevant, not part of the verified core)

Fetched by the BB72-applicability angle; their claims did not survive into the final synthesis, so
treat as leads, not findings:

- `arXiv:2604.01040` — circuit-level Monte Carlo on a BB72 `[[72,12,6]]` baseline, 101 operating
  points; a geometry-derived "reference-support weighted exposure" correlates with logical error
  rate (Spearman ρ = 0.893). **Directly on this repo's code and noise regime** — the most
  immediately relevant thing the run touched.
- PRResearch `10.1103/ydb8-vd3x` — HAMLD, hypergraph approximate ML decoding; the authors claim
  it handles BB codes with hundreds of qubits that MLD and TN-MLD cannot.
- `arXiv:2609.03928` — AMLD post-processor, 13% LER reduction vs minimum-weight decoding over the
  same BP-OSD candidate pool at p=0.05.

## Caveats on this report

- Every source is a 2026 paper (Feb–Sep 2026). This is a months-old, fast-moving literature.
- Underlying reads were largely AI-mediated HTML/PDF summaries. The paper itself was also extracted
  locally to `_paper_2609.txt`, and the method/Q-table/no-BB72 claims were re-checked verbatim
  against it.
- The code-availability conclusion is absence-of-evidence; the search budget ran out before a
  broad GitHub sweep.
- No number in this line is compute-matched, and none is independently replicated.
