# BB72 Transformer-Based Quantum Error Correction Decoder — Chat Export

> Export scope: the user and assistant messages visible in this conversation, including the assistant's complete technical response. Project documents and uploaded PDFs are referenced rather than embedded in full. UI-specific badges, diagrams, and file-citation widgets are represented in ordinary Markdown.

---

## User

In this current stage of the project, I am trying to explore the possibilities of using Machine learning methods to assist as a decoder. The main goal will be to bypass the expensive tensor contractions. (reference the RL optimised tensor contraction paper, transformer decoder paper and the MADE paper). Presently, the Transformer architecture as a decoder is what i am trying to do. Verify the validity of using a Transformer that takes in input of the syndrome and maybe the BP-OSD values and outputs the corrected logical state.

---

## Assistant — Research plan

I’ll check the three papers against the BB72 results you’ve already obtained, then pin down the Transformer’s input and output, how to generate correct training labels, and whether it can realistically replace the tensor-network decoder rather than just improve BP-OSD.

## Assistant — Technical assessment (part 1)

# Feasibility of a Transformer-Based Decoder for BB72 Quantum Error Correction

**Verdict: Yes — using a Transformer to decode BB72 directly from the syndrome is mathematically valid, supported by published research, and a promising approach to bypass expensive tensor-network (TN) contractions.**

However, the precise formulation matters. I recommend treating the Transformer as a **learned logical maximum-likelihood decoder**, rather than a neural network attempting to reconstruct all 72 physical errors.

The proposed mapping is

$$
\boxed{(s,\ \text{optional BP information},\ p)\xrightarrow{\text{Transformer}}\hat{\lambda}\in\mathbb F_2^{12}}
$$

where $\hat{\lambda}$ identifies a predicted logical error sector and $p$ is the physical error rate.

There is one important distinction: the output should be a **logical correction or logical Pauli-frame update**, not the corrected quantum state itself. The syndrome does not contain enough information to reconstruct an arbitrary encoded quantum state.

## 1. Why this approach is worth investigating now

Your project already has three relevant milestones:

| BB72 decoder | Recorded result | Main limitation |
|---|---|---|
| BP+OSD | Established fast baseline across six physical error rates | Does not explicitly maximize total logical-coset probability |
| Cached TN + Walsh–Hadamard | 182.25 s per shot on average | Requires 4,095 TN parity queries |
| Exact TN branch-and-bound v0.3 | 1.878 s (easy), 3.414 s (hard) | Still performs expensive contractions |

The branch-and-bound timings are from two validation syndromes, not a full logical-error-rate sweep, so they should not yet be treated as representative average latency. [Project source: 2026-09-18 BB72 Exact BnB v0.3 Report, lines 162–192]

The natural next step is to ask:

> Can a neural network learn the mapping from observed syndromes to likely logical error sectors without performing tensor contractions at inference time?

This is a meaningful research direction because TN decoding and neural decoding can target the same underlying probability distribution, but compute it in fundamentally different ways.

The evidence in the uploaded literature is particularly relevant:

- **Blue et al., *Machine Learning Decoding of Circuit-Level Noise for Bivariate Bicycle Codes* (2026):** demonstrates a recurrent Transformer-based decoder on the $[[72,12,6]]$ BB code, reporting approximately 4.5-fold lower logical error rate than its BP-OSD-3 benchmark at physical error rate $p=0.001$, under circuit-level noise. [Source: uploaded Transformer paper, lines 142–169]
- **Cao et al., *Generative Decoding for Quantum Error-correcting Codes* (2025):** explicitly considers MADE and causal Transformers for autoregressive logical decoding, including BB72. [Source: uploaded MADE/generative decoding paper, lines 61–94]
- **Merom et al., *Optimizing Tensor Network Contraction Using Reinforcement Learning* (2022):** uses an RL agent with a graph neural network to optimize contraction ordering, but does not eliminate contraction itself. [Source: uploaded RL contraction paper, lines 8–34]

These support two different ML approaches: **accelerating TN computation** and **replacing TN computation with an approximate learned decoder**. Your proposed Transformer belongs to the second category.

## Assistant — Interim research finding

One important caveat has emerged from the papers: Transformers are viable, but they are not automatically the best neural architecture here. The MADE paper reports better results with a smaller MADE model in its comparisons. I’m also checking how the papers define the 12 logical output bits, because that determines whether the model can correct BP-OSD’s mistakes without introducing a reference-frame error.

## Assistant — Technical assessment (continued)

## 2. Mathematical validity: Can a Transformer learn what the TN computes?

Yes. The reason is that logical maximum-likelihood decoding is ultimately a conditional probability estimation problem.

For the BB72 code under independent $X$-error noise, let

$$
e\in\mathbb F_2^{72}, \qquad s=H_Ze^T
$$

where $s$ is the measured syndrome and $e$ is the unknown physical error.

Your TN decoder computes the probability associated with each logical sector:

$$
Z_\lambda(s)
=
\sum_{g\in\mathcal S_X}
P(e_0(s)+\lambda L_X+g).
$$

Here, $e_0(s)$ is any syndrome-consistent reference error, $L_X$ contains the 12 independent logical $X$ operators, and $g$ runs over the $X$-stabilizer group.

The logical maximum-likelihood solution is

$$
\lambda_{\mathrm{ML}}(s)
=
\arg\max_{\lambda\in\mathbb F_2^{12}}P(\lambda\mid s).
$$

The Transformer does not have to explicitly perform the summation over stabilizers. Instead, it can learn an approximation:

$$
q_\theta(\lambda\mid s,p)
\approx P(\lambda\mid s,p).
$$

The learned decoder then produces

$$
\boxed{
\hat{\lambda}
=
\arg\max_\lambda q_\theta(\lambda\mid s,p)
}
$$

This is the fundamental justification for using machine learning to bypass TN contractions.

**One critical qualification:** learning an approximation to the posterior is possible, but it is not guaranteed that a practical neural network will approximate it sufficiently well. That must be established experimentally.

### Why the 12-bit output is sufficient

For an $X$-only error model, BB72 has 12 independent logical $X$ degrees of freedom.

Let $\bar L_Z$ be a matrix containing the 12 dual logical $Z$ operators, chosen so that

$$
\bar L_ZL_X^T=I_{12},
\qquad
\bar L_ZH_X^T=0.
$$

For every sampled physical error $e$, define the correct logical label relative to a reference $e_0(s)$:

$$
\boxed{
\lambda_{\mathrm{true}}
=
\bar L_Z(e\oplus e_0(s))^T.
}
$$

This produces a 12-bit training target.

After inference, a valid physical correction can be constructed:

$$
\boxed{
\hat e
=
e_0(s)\oplus\hat\lambda L_X.
}
$$

By construction, this correction has the measured syndrome, and it succeeds whenever the predicted logical sector is correct.

This also clarifies an important constraint: the reference error $e_0(s)$, logical basis, and labeling conventions must be consistent between training, inference, and any TN reference calculation.

For full depolarizing Pauli noise, rather than a single $X$- or $Z$-error component, the output generally expands to **24 binary logical variables**, representing $X$- and $Z$-type logical information.

---

## 3. The architecture I would recommend

I would start with an encoder–autoregressive-decoder Transformer, with a code-aware attention mechanism adapted to BB72.

**Architecture flow:**

```text
Input:
  36 Z-check syndrome bits + physical error rate p
  Optional: 72 BP+OSD correction bits and soft reliabilities
                    |
                    v
       Code-aware Transformer Encoder
  (self-attention constrained or biased by
        BB72 Tanner-graph connectivity)
                    |
                    v
       Autoregressive Logical Decoder
  predict q_theta(lambda_i | lambda_<i, s, p)
                for i = 1,...,12
                    |
                    v
         12-bit logical correction label
  (physical correction or logical Pauli-frame update)
```

### 3.1 Input representation

For the first implementation, use the 36 Z-check syndrome bits. Although $H_Z$ has rank 30, keeping all 36 check outcomes preserves a natural correspondence with the code's check graph.

Each input token can contain a syndrome-bit embedding, a check-index or toroidal-coordinate embedding, and information about the noise rate.

Unlike natural language, the positions of BB72 checks are not arbitrary. Their connections are determined by the parity-check matrix. This suggests incorporating structural information rather than relying entirely on unconstrained attention.

### 3.2 Code-aware self-attention

Blue et al. introduce a particularly relevant mechanism: instead of allowing every detector token to interact equally with every other, the network uses structural information from the detector error model.

Their attention logits include

$$
A_{ij}
=
\frac{Q_iK_j^T}{\sqrt{d_k}}+M_{ij},
$$

with a mask based on

$$
M=\log(DD^T),
$$

where $D$ describes which elementary errors flip which detectors. Entries with no shared elementary error receive zero attention weight after softmax. [Source: uploaded Transformer paper, lines 779–867]

For your one-round independent-$X$ code-capacity setting, an analogous incidence matrix is $D=H_Z$.

This gives a natural starting point:

$$
M_{ij}=\log((H_ZH_Z^T)_{ij}),
$$

with appropriate handling of zero entries and self-attention.

The resulting model is more closely connected to the BB72 Tanner graph than an ordinary Transformer.

### 3.3 Why the output should be autoregressive

A tempting approach is to predict all 12 logical bits independently using 12 sigmoid outputs.

That represents a product approximation:

$$
q(\lambda\mid s)
\approx
\prod_{i=1}^{12}q(\lambda_i\mid s).
$$

However, this ignores logical correlations. Those correlations matter because the correct decoder should compare the probability masses of complete logical sectors.

The autoregressive formulation is more expressive:

$$
\boxed{
q_\theta(\lambda\mid s,p)
=
\prod_{i=1}^{12}
q_\theta(\lambda_i\mid\lambda_{<i},s,p).
}
$$

This factorization can represent any joint distribution over the 12 bits, provided the conditional models have sufficient capacity.

During training, the decoder uses teacher forcing and binary cross-entropy:

$$
\mathcal L
=
-\sum_{i=1}^{12}
\log q_\theta(\lambda_i^{\mathrm{true}}
\mid\lambda_{<i}^{\mathrm{true}},s,p).
$$

There is an important computational subtlety here. Predicting each successive bit greedily does **not** always recover the globally most probable 12-bit sector. For example, if the probabilities of $00,01,10,11$ are $0.35,0.25,0.40,0$, greedy decoding selects $00$, whereas the true MAP result is $10$.

I recommend testing greedy decoding against beam search with widths such as 8 and 16. This gives a controlled inference-time accuracy trade-off without any TN contraction.

---

## 4. Should you include BP+OSD information?

**Yes, as a separate experimental variant — but not as a requirement for the first model.**

There are two useful formulations.

**A. Syndrome-only Transformer**

$$
(s,p)\rightarrow\hat\lambda
$$

Attempts to learn logical decoding entirely without BP+OSD or TN inference. It is the cleanest test of your original hypothesis.

**B. BP-assisted Transformer**

$$
(s,\hat e_{\rm BP},r,p)\rightarrow\hat\delta
$$

Uses the BP+OSD candidate correction $\hat e_{\rm BP}$, optionally with soft reliability information $r$, to predict a logical residual correction.

The second formulation is particularly interesting.

Suppose BP+OSD returns a syndrome-consistent estimate $e_{\mathrm{BP}}$, satisfying

$$
H_Ze_{\mathrm{BP}}^T=s.
$$

We can define the remaining logical error after applying that candidate correction:

$$
\boxed{
\delta_{\mathrm{true}}
=
\bar L_Z(e\oplus e_{\mathrm{BP}})^T.
}
$$

The neural network learns

$$
q_\theta(\delta\mid s,e_{\mathrm{BP}},r,p).
$$

At inference, its residual correction can be applied as

$$
\boxed{
e_{\mathrm{final}}
=
e_{\mathrm{BP}}\oplus\hat\delta L_X.
}
$$

This construction is particularly convenient because the extra logical operator does not alter the syndrome. It therefore changes the logical equivalence class of the BP candidate while preserving the measured parity checks.

### The limitation of BP+OSD features

If BP+OSD is deterministic given $s,p$, then its output contains no additional independent information about the true error beyond $s,p$. In information-theoretic terms,

$$
I(\lambda;e_{\mathrm{BP}}\mid s,p)=0.
$$

Nevertheless, its output may provide a useful inductive bias: BP+OSD has already performed structured inference that the Transformer would otherwise need to learn from data.

There is a practical trade-off. A hybrid model must pay the BP+OSD inference cost first, while the syndrome-only model avoids it. If the purpose is maximum speed, use the syndrome-only architecture as the primary decoder and the BP-assisted architecture as a competing variant.

---

## 5. What the three papers imply for your specific project

| Paper | Most transferable lesson | Important limitation |
|---|---|---|
| **Blue et al. — Transformer for BB codes** | Autoregressive logical prediction plus code-aware attention is a demonstrated approach on BB72. | Uses multiple noisy measurement rounds and circuit-level noise, unlike your initial code-capacity setting. |
| **Cao et al. — MADE / generative decoding** | Learning conditional logical probabilities directly from sampled errors avoids explicitly summing stabilizer cosets during inference. | In their comparisons, MADE performs better than their Transformer implementation. Training can be data-intensive. |
| **Merom et al. — RL TN contraction ordering** | ML can accelerate the choice of tensor contraction paths. | It does not directly replace probability-mass contractions. |

The MADE result deserves special attention. Cao et al. report that their causal Transformer underperformed MADE, with greater computational and memory demands. Their BB72 code-capacity experiments also used depolarizing noise and training at a much higher physical error rate than your present low-$p$ benchmarks. Their reported BB72 training involved $10^{10}$ sampled examples. [Source: uploaded MADE paper, lines 230–270 and 510–528]

Consequently, I would not select Transformer solely because of its popularity. Its potential advantage for BB72 is the combination of structured attention, flexible conditioning on decoder features, and extension to multi-round circuit-level data. Those are hypotheses worth testing against MADE.

The published Blue et al. study also makes the feasibility argument stronger: its implementation is available in the [authors' GitHub repository](https://github.com/jblue1/ml_bb_decoding), and the paper is published in *Quantum* (2026).

---

## 6. You do not need TN outputs to train the Transformer

This is probably the most useful implementation insight.

You already have a simulator capable of sampling BB72 physical errors. For each training shot, the physical error is known because you generated it.

Therefore, you can calculate its syndrome and logical label directly with binary linear algebra.

**Training-data construction — no TN contraction:**

```python
# e: sampled 72-bit X-error vector
# H_Z: Z-check matrix
# L_Z: 12 dual logical-Z observables
# L_X: 12 logical-X generators

s = (H_Z @ e) % 2

# Choose a deterministic syndrome-consistent reference
e0 = solve_syndrome(s)

lambda_true = (L_Z @ (e ^ e0)) % 2

# Train on (s, p) -> lambda_true
```

Here `solve_syndrome` can be implemented using a precomputed GF(2) linear right-inverse on the valid syndrome space, rather than running a general solver for every sample.

The data-generation process is then:

$$
e\sim P_p(e)
\quad\longrightarrow\quad
(s,\lambda_{\mathrm{true}},p).
$$

Training on these samples with conditional negative log-likelihood learns the underlying posterior distribution statistically. It does not require a TN-computed MAP label or a full posterior for every example.

This is closely related to the generative-decoding paper's central methodology. Its authors describe training from samples drawn directly from an error model, using an error–logical–syndrome decomposition. [Source: uploaded MADE paper, lines 174–224]

A crucial distinction: each sampled error gives one valid logical label, not the most probable logical label for that syndrome. This is entirely acceptable for probabilistic training. Different labels for the same syndrome represent the genuine conditional uncertainty the model is supposed to learn.

For a BP-assisted variant, run BP+OSD on the sampled syndrome and replace the target with $\delta_{\mathrm{true}}$.

---

## 7. A practical experiment plan

I would implement and compare the following models under one identical BB72 error model, with consistent logical-operator conventions.

| Model | Input | Output | Purpose |
|---|---|---|---|
| BP+OSD | Syndrome | Physical candidate correction | Existing fast baseline |
| MADE | Syndrome + $p$ | Autoregressive 12-bit logical label | Lightweight generative baseline |
| Plain Transformer | Syndrome + $p$ | Autoregressive 12-bit logical label | Test architectural feasibility |
| Code-aware Transformer | Syndrome + $p$, structural mask | Autoregressive 12-bit logical label | Test whether BB72 graph structure helps |
| BP-assisted Transformer | Syndrome + BP features + $p$ | 12-bit residual logical label | Test hybrid decoding |
| Exact TN BnB | Syndrome + $p$ | ML logical sector | High-quality reference on matched test cases |

For a first Transformer prototype, a model dimension of 128 or 256 with 2–4 Transformer layers and four attention heads would be a reasonable starting range. These are suggested hyperparameters, not verified optimal settings.

### Training and evaluation

Use the independent-$X$ error model first, matching your established BP+OSD baseline. The reported TN sweep used a $Z$-error channel and a `weight6-x` basis, so those results should not be compared directly with the $X$-error BP+OSD baseline without a matching experiment or verified symmetry argument. [Project source: 2026-09-10 BB72 cached TN sweep report, lines 259–284]

The main test should reproduce the existing physical error-rate sweep:

$$
p\in\{0.005,0.01,0.02,0.03,0.04,0.05\}.
$$

The most important accuracy metric is logical error rate:

$$
\boxed{
\mathrm{LER}(p)
=
\Pr\left[
\bar L_Z(e\oplus\hat e)^T\ne0
\right].
}
$$

Measure this over independent held-out simulated errors, and check syndrome consistency separately. For confidence in low-LER results, substantially more than 5,000 shots per point may be necessary. At $p=0.005$, the recorded CPU BP+OSD order-10 LER was only 0.0012, corresponding to about six failures in 5,000 shots. [Project source: 2026-08-28 BB72 BP+OSD baseline report, lines 260–275 and comparison section]

I would prioritize three experiment stages.

**Stage 1 — Feasibility:** Train the syndrome-only Transformer and MADE from noise-generated labels, then compare their LERs with BP+OSD. This establishes whether the network learns the logical decoding problem at all.

**Stage 2 — Architectural improvement:** Introduce code-aware attention and test an autoregressive beam search. Compare against the plain Transformer on the same train/test split, to identify whether the added complexity improves accuracy.

**Stage 3 — TN replacement test:** Compare the best neural model with exact TN branch-and-bound on a matched subset of syndromes, including ambiguous or highly degenerate cases. Record both logical-sector agreement and inference latency.

For speed, measure median, 99th-percentile, and worst-observed decoding time at batch size one, as well as throughput at larger batch sizes. A low average runtime alone is insufficient for real-time QEC.

### The most challenging part: rare logical failures

A Transformer can obtain an impressive overall decoding accuracy simply because most low-noise syndromes have a strongly dominant logical sector.

That does not establish that it handles difficult decoding cases.

You should separately measure performance conditioned on BP+OSD success and failure. The latter cases are particularly valuable for determining whether neural decoding learns something that BP+OSD misses.

However, training only on BP failures would distort the natural posterior. A better approach is to begin with correctly distributed samples, then investigate hard-example oversampling with appropriate weighting or subsequent fine-tuning.

---

## 8. What would count as success?

Your strongest comparison should be against **both** BP+OSD and the improved TN branch-and-bound decoder.

The current TN v0.3 result significantly changes the engineering target. Beating the earlier 182-second exhaustive-posterior method is not enough, because your project already demonstrates much faster exact ML search on selected cases.

A useful success criterion would be a neural decoder that:

- Achieves LER significantly better than BP+OSD, ideally approaching the exact TN reference.
- Performs no TN contraction at inference time.
- Has materially lower matched single-shot latency than exact TN BnB.
- Maintains performance across physical error rates and on held-out difficult syndromes.

The model is not expected to systematically outperform an accurate matched maximum-likelihood decoder: ML decoding already minimizes expected logical error probability under the specified noise model. The goal is to approximate that solution at a much lower online computational cost.

## 9. Final assessment

**Mathematical validity — Established.** An autoregressive Transformer can represent the conditional logical posterior $P(\lambda\mid s,p)$.

**Research precedent — Established.** Published Transformer and MADE-based logical decoders have been evaluated on BB72.

**Replacing TN inference — Technically feasible.** The trained neural decoder can operate without tensor contractions; its approximation quality remains an empirical question.

**Matching TN decoding accuracy — Not yet demonstrated here.** Requires benchmarking against your exact BnB implementation on matched syndromes.

**My recommendation is to pursue the Transformer, but start with a syndrome-only, autoregressive logical decoder and use MADE as a mandatory baseline.** Add BP+OSD features only after establishing the performance of the standalone neural model.

This keeps the central research question clear: *Can a code-aware neural model approximate BB72 logical maximum-likelihood decoding sufficiently well to eliminate the online tensor-contraction cost?*

A successful result would not merely accelerate the existing TN pipeline. It would replace its expensive probability-summation stage with learned statistical inference — while preserving the logical-sector objective that makes TN decoding valuable.

---

## User — Export request

export this chat as a .md file, keep all the contents

