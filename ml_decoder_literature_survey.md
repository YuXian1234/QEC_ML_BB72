# Machine-learning decoders for bivariate bicycle codes — literature survey

Target: **BB72 `[[72,12,6]]`** (Bravyi et al. layout, `A = x³ + y + y²`, `B = y³ + x + x²`).
Produced by a deep-research run on 2026-10-01: 5 search angles, 16 sources fetched, 79 claims
extracted, 25 adversarially verified (3-vote), 22 confirmed, 3 killed, 13 after synthesis.

The question driving this: the repo already benchmarks BP+OSD (qldpc, CUDA-QX) and a TN decoder
plus exact branch-and-bound on BB72. What ML decoding options exist, and has anyone actually run
them on *this* code?

## Bottom line

Coverage of BB72 specifically is **thin**. Only three decoders touch it:

| Decoder | Kind | BB72? | Noise | Headline vs BP+OSD | Code |
|---|---|---|---|---|---|
| **Blue et al.** `2504.13043` | supervised recurrent transformer | **yes, direct** | circuit-level | ~4.5× lower LER @ p=0.1% | no |
| **GND** `2503.21374` | unsupervised autoregressive | **yes, direct** | code-capacity | ~10× better accuracy, p<10⁻² | `CHY-i/GND` |
| **BF-OSD** `2605.25777` | best-first OSD search (classical) | **yes, Table I row** | circuit-level | 6.58e-5 vs 7e-5 @ p=1e-3 | no |
| **Tesseract** `2503.10988` | A\* search MLE (classical) | appendix figure only | circuit-level | ~100× lower LER | `quantumlib/tesseract-decoder` |

The single most important negative result: **no neural-network decoder other than Blue et al. has
been demonstrated on BB72 at all**, and Blue et al.'s is the only one at circuit-level noise. Every
other neural approach (GNN, neural-BP, RL, diffusion, AlphaQubit) is surface-code-only or
BB-family-by-distance without naming `[[72,12,6]]`.

The single most promising near-term option for this repo: **GND** — it names BB72 explicitly, needs
no labels (sidestepping the `4^k` label-scaling problem for k=12), and claims ~10× over BP+OSD. Its
weakness is that the comparison is code-capacity, not circuit-level.

## Deep neural decoders

### Blue et al. — supervised recurrent transformer (the only DNN on BB72)

`arXiv:2504.13043`, Blue, Avlani, He, Ziyin, Chuang; published **Quantum 10, 2149 (2026)**.

- On BB72 at p=0.1% circuit-level: **~4.5× lower LER than BP-OSD-3** (body §1.3 says "roughly 4.5
  times lower"; the abstract and the IAIFI writeup say "almost 5 times"). It remains **~5.4× larger
  than an exact MLE decoder** — so it beats BP-OSD without reaching maximum-likelihood.
- The advantage **grows as p decreases** (measured across all tested p, not one point).
- Training is **fully supervised** binary cross-entropy, ~3.8e8 examples (16,384/epoch × ~23,000
  epochs). This is the label-scaling cost that GND avoids.
- **It does not scale.** On `[[144,12,12]]` circuit-level it only matches BP-OSD-3 for p=0.3–0.7%,
  and is **~7× worse at p=0.1%**. The authors' own conclusion: ML decoders beat conventional ones
  only on *small* qLDPC codes so far. Order-of-magnitude runtime advantage retained though.
- No code repository found.

### GND — unsupervised generative (the best BB72 + label-free fit)

`arXiv:2503.21374`, Cao, Pan, Feng, Wang, Zhang. Code: `github.com/CHY-i/GND`.

- Autoregressive model of the joint logical/syndrome distribution. **No labeled training data** —
  this is the key structural advantage, since supervised label count scales as `4^k` and k=12 here.
- **Evaluated directly on BB72**, alongside `[[60,8,4]]` and `[[18,4,4]]` BB codes and a
  `[[30,6,4]]` qLDPC code.
- Reports **~10× better accuracy than BP+OSD for p < 10⁻²** (authors' self-reported).
- Claimed `O(2k)` complexity for k logical qubits — favourable at k=12.
- **Scope limit:** code-capacity depolarizing only, not circuit-level. Not independently replicated.

### Astra — GNN message-passing (BB-family, but *not* BB72)

`arXiv:2408.07038`, Maan & Paler; **npj Quantum Information 11, 78 (2025)**.

- Trained on BB codes reported **only by distance** (6, 12, 18; extrapolated to 34) — no explicit
  `[[n,k,d]]` instance named except the large extrapolation targets. The distance-6 row is plausibly
  BB72 but is labeled only by distance. **Do not treat Astra as BB72 evidence.**
- Code-capacity only; circuit-level explicitly deferred to future work.

### L-NBP — neural belief propagation (surface codes only)

`arXiv:2608.27682`. NBP module produces posteriors; a logical classifier converts them to a
"soft syndrome" for logical-operator prediction; trained end-to-end.

- **Evaluation is rotated surface codes only** (d=3,9,11,13). No BB or qLDPC simulation at all —
  qLDPC work appears in the bibliography, never in results. **Zero BB72 evidence.**

### Others, for completeness

- **AlphaQubit** (`Nature` 2024, s41586-024-08148-8) — recurrent transformer, surface codes only
  (d=3..11, Google Sycamore data). No BB codes. BB72 applicability unestablished.
- **Diffusion decoder** (`arXiv:2509.22347`) — BB codes under circuit-level noise, but BB72 is never
  named; `[[72,12,6]]`-specific performance not established.
- **QuBA / SAGU** (`arXiv:2510.06257`) — BB and coprime-BB family (`[[90,8,10]]`, `[[144,12,12]]`,
  `[[288,12,18]]`, `[[756,16,≤34]]`, coprime `[[30,4,6]]`, `[[154,6,16]]`). BB72 is not a headline
  result.

## Classical learned / heuristic-search decoders

These currently *dominate* the circuit-level BB benchmarks — and none of them needs neural training.

### Tesseract — A\* MLE search

`arXiv:2503.10988`. Admissible heuristic + pruning, no training data at all.
On BB codes at circuit-level p=0.001: **~100× lower LER than BP+OSD** (one to two orders of
magnitude). Demonstrated on BB72 **only in Appendix D, Figure 8** — a peripheral comparison, not a
headline result (this sub-claim passed 2–1; the other two passed 3–0). Open implementation:
`quantumlib/tesseract-decoder` (GitHub/PyPI).

Directly relevant to this repo: Tesseract is the same *class* of thing as the frozen
`v0.5_final_production/` exact branch-and-bound — exact search over the logical space — but with an
A\* heuristic and pruning instead of block branching.

### BF-OSD — best-first ordered-statistics search

`arXiv:2605.25777`, Banfi et al., May 2026 (preprint, not peer-reviewed).

- **The most BB72-specific classical-search result in the set.** On `[[72,12,6]]` under full
  circuit-level noise at p=1e-3: per-round LER **6.58e-5 vs 7e-5** for the BP+OSD-CS baseline of
  Bravyi et al. — a slight improvement.
- Matches BP+OSD overall while exploring with **~1/100th of the query budget**.
- Tested across the BB family: `[[72,12,6]]`, `[[90,8,10]]`, `[[108,8,10]]`, `[[144,12,12]]`,
  `[[288,12,18]]`.

### Beam search — BP-directed

`arXiv:2512.07057`, Ye, Wecker, Delfosse; **PRX Quantum 7, 033002 (2026)**. Open code:
`github.com/ionq-publications/BeamSearchDecoder`.

- On `[[144,12,12]]` at p=1e-3 circuit-level: **17× LER reduction vs BP-OSD at beam width 64**;
  beam width 8 **matches BP-OSD accuracy with a 26.2× cut in 99.9-percentile runtime**.
- Beam width is an explicit speed/accuracy knob. Single-core software, no FPGA/ASIC.
- **Not a neural network** (BP + beam search). Figures are for `[[144,12,12]]`, not BB72 — this is
  the `2512.07057v2` PDF already in `materials/`.

### Ambiguity Clustering (AC)

`arXiv:2406.14527`, Wolanski & Barber. Run BP, then split the measurement data into independently
decodable clusters. On BB qLDPC (incl. the `[[144,12,12]]` Gross code) at 0.3% circuit-level:
matches BP-OSD accuracy, **up to 27× faster**.

**Caveat worth internalising:** the authors' own v2 (Jan 2025) **corrected a BP-OSD
misconfiguration that had inflated BP-OSD runtimes**, narrowing the gap. "Matched accuracy +
substantial speedup" survives; the exact 27× is not a stable constant. This is a live hazard for the
whole literature — BP+OSD baselines are easy to configure badly.

### Adaptive BP+OSD

`arXiv:2609.37629`, Wang & Coveney, Sep 2026 (preprint). Uses the disagreement between the BP hard
decision and the syndrome-consistent OSD-0 solution as an internal risk indicator, escalating only
high-risk instances. On `[[144,12,12]]` circuit-level: escalating the top 20% by risk recovers
**85–92% of the full-sweep LER improvement at 3.6× lower mean cost**. Beats routing by BP residual,
syndrome weight, or random selection. Directly relevant if the repo's BP+OSD sweep cost is a
bottleneck.

### RL-S — colour-clustered reinforcement-learning sequential BP

`arXiv:2609.20236`, Sep 2026 (preprint, **medium confidence**). Trained-but-label-free VN-level
Q-table picks a seed variable node; same-colour nodes update in parallel. On `[[288,12,18]]`, BLER
comparable to BP-OSD-10 (T=1000) already at **T=100**, avoiding OSD entirely. "Comparable" is the
authors' own qualitative phrasing over a figure, not a quoted LER; no code link; the Q-table is
reused from prior work rather than trained here.

## Caveats and internal conflicts

- **Self-reported numbers.** Nearly every quantitative comparison is the method authors' own. Only
  Blue et al. and Astra are peer-reviewed *and* report exact numbers.
- **BP+OSD baseline mismatch.** Several comparisons may use different BP+OSD configurations — a
  known distortion source (AC's 27× was revised down for exactly this reason). Any head-to-head
  against this repo's own BP+OSD should re-run the baseline locally.
- **Version-dependent scoping (unresolved).** Astra's "code-capacity only / BB72 never tested"
  scoping was **confirmed 3–0 against the arXiv version** (`2408.07038`) and its GitHub README, but
  the *same scoping* was **refuted 0–3 against the published npj version**
  (s41534-025-01033-w), and a related Astra code-coverage claim was refuted 0–3 the same way. The
  likely explanation is that the arXiv and published versions differ in what they report. Treat
  Astra's scope as version-dependent and check the specific version before relying on it. The same
  pattern killed a scoping claim about `2609.20236` (RL-S).
- **Half the sources are 2026 preprints** — adaptive BP+OSD (`2609.37629`), RL-S (`2609.20236`),
  BF-OSD (`2605.25777`), L-NBP (`2608.27682`). Their numbers are provisional.
- **Latency numbers are mostly not BB72.** Timing claims are for `[[144,12,12]]` or larger.
- **Code availability is uneven.** Open: beam search, Tesseract, Astra, GND. No repository found
  for RL-S, L-NBP, Blue et al., BF-OSD, AC, or adaptive BP+OSD.
- **Nothing here reproduces on the repo's own BB72 setup.** These are literature benchmarks.

## Open questions this survey could not settle

1. Can *any* deep-learning decoder reach MLE-level accuracy on BB72 at circuit-level noise, and at
   what budget — given Blue et al. need ~3.8e8 examples and still sit ~5.4× from MLE?
2. Do GND and the classical search decoders (Tesseract, BF-OSD, beam search, AC) hold their BB72
   advantage under **full circuit-level** noise rather than code-capacity, and how do they compare
   head-to-head with this repo's TN and exact-BnB decoders?
3. What is actual inference latency **at BB72 scale** (most reported numbers are `[[144,12,12]]` or
   bigger), and can beam-width, GND's `O(2k)`, or BF-OSD's query budget give real-time decoding for
   72 qubits / 12 logicals?
4. Which of these have reproducible code with matching noise models and BP+OSD baselines that this
   repo could benchmark directly?

## Sources

Primary papers:

- **Blue et al.**, *Machine Learning Decoding of Circuit-Level Noise for Bivariate Bicycle Codes* —
  https://arxiv.org/abs/2504.13043 (Quantum 10, 2149, 2026)
- **Cao et al. (GND)**, *Generative Decoding for Quantum Error-correcting Codes* —
  https://arxiv.org/abs/2503.21374 (also in `materials/`)
- **Ye, Wecker, Delfosse**, *Beam search decoder for quantum LDPC codes* —
  https://arxiv.org/abs/2512.07057 (PRX Quantum 7, 033002, 2026; also in `materials/`)
- **Tesseract** — https://arxiv.org/abs/2503.10988
- **BF-OSD**, Banfi et al. — arXiv:2605.25777 (via
  https://www.semanticscholar.org/reader/b36543cdbce0210c0fb70ce4beb500d025745e8a)
- **Wolanski & Barber (AC)**, *Ambiguity Clustering* — https://arxiv.org/abs/2406.14527
- **Adaptive BP+OSD**, Wang & Coveney — https://arxiv.org/abs/2609.37629
- **RL-S colour-clustered** — https://arxiv.org/abs/2609.20236 (PDF:
  https://export.arxiv.org/pdf/2609.20236)
- **Maan & Paler (Astra)**, *Machine Learning Message-Passing for the Scalable Decoding of QLDPC
  Codes* — https://arxiv.org/abs/2408.07038 / https://www.nature.com/articles/s41534-025-01033-w
- **L-NBP** — https://arxiv.org/abs/2608.27682
- **QuBA / SAGU** — https://arxiv.org/abs/2510.06257
- **Diffusion decoder** — https://arxiv.org/abs/2509.22347
- **Bausch et al. (AlphaQubit)** — https://www.nature.com/articles/s41586-024-08148-8

Secondary:

- IAIFI writeup of Blue et al. —
  https://research.iaifi.org/posts/machine-learning-decoding-of-circuit-level-noise-for-bivariate-bicycle-codes/
