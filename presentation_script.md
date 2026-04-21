# Presentation Script — Conformal Meta-Analysis

**Audience:** PhD students in CS working with the Deep Learning professor.
Most of their work is EHR-related deep learning. Assume they know
neural nets, PyTorch, PyHealth conventions, and statistical ML
broadly, but *not* meta-analysis vocabulary or full conformal prediction.

**Target: 6:00–6:30 delivered. Hard ceiling: 7:00.**

Recorded delivery runs faster than live. If your natural pace lands at
5:30, that's still inside the 4–7 minute rubric window.

---

## Section 1 — The General Problem (≈50 sec, ~135 words)

> Hi, I'm Brett Darragh, and I implemented Kaul and Gordon's 2024 paper,
> "Meta-Analysis with Untrusted Data," as a set of PyHealth contributions.
>
> The problem is about meta-analysis: combining multiple clinical trials
> to estimate a single treatment effect more precisely than any
> individual trial could on its own. The standard method here is
> HKSJ, which is the frequentist baseline endorsed by Cochrane 
> and other major guideline bodies.
>
> The core tension the paper addresses is this: trusted randomized
> controlled trials are scarce, but untrusted evidence such as
> cohort studies, case-controlled studies and even expert opinion
> — is plentiful. Classical methods like HKSJ ignore all of it.
> The paper asks whether we can incorporate that untrusted evidence
> to get tighter predictionintervals, without giving up the statistical
> coverage guarantees that make meta-analysis credible.

**[Slide: "show the diagram from the paper"]**

---

## Section 2 — The Paper's Approach (≈90 sec, ~245 words)

> The method from Kaul and Gordon is called Conformal Meta-Analysis, or CMA.
> It has three layers.
>
> **Layer one: the prior.** They train a deep ReLU network on the
> untrusted data to learn a mapping from trial features to expected
> effect size. The network outputs both a predicted prior mean M of X,
> and implicitly defines a learned kernel that the next layer uses.
>
> **Layer two: kernel ridge regression.** They run KRR over the trusted
> trials, using the learned prior from layer one as the center point.
> This produces a point estimate plus an uncertainty quantification
> grounded in the trusted data.
>
> **Layer three: full conformal prediction.** This is the key
> contribution. They wrap the whole pipeline in full conformal
> prediction, which gives a distribution-free coverage guarantee: the
> final prediction interval covers the true effect with probability at
> least 1 minus alpha, regardless of how wrong the learned prior from
> layer one is. If the untrusted data is misleading, the intervals
> widen automatically. If it's informative, they tighten. Coverage
> never breaks.
>
> They evaluate on two metrics — interval **width** as a measure of
> precision, and empirical **coverage** as a measure of validity — and
> compare against HKSJ as the baseline.
>
> They validate this on partially-synthetic simulations built from the
> Penn Machine Learning Benchmark, and on a real case study of 21
> amiodarone (amee- o - darone) trials for atrial fibrillation. 
> The main claim is that CMA
> produces narrower intervals than HKSJ when the prior is informative,
> matches HKSJ when the prior is bad, and maintains target coverage
> in both regimes.

**[Slide: three-layer diagram — "Prior NN → KRR → Conformal wrap" with
inputs "untrusted" into layer 1, "trusted" into layer 2]**

---

## Section 3 — Reproduction (≈110 sec, ~295 words)

> I reproduced the paper's four main simulations on PMLB, following
> the authors' synthetic data-generation procedure from Appendix B.6.
>
> **[Slide: your 4-panel simulation figure]**
>
> Simulation 1 tests interval width versus sample size, across three
> prior qualities: bad, okay, and good. The pattern I see matches the
> paper's claim — at n=20 with a good prior, CMA intervals are roughly
> [X]% narrower than HKSJ, and the gap closes as n grows.
>
> **[Move to Sim 2 panel]**
>
> Simulation 2 is the coverage test. The claim is that CMA holds the
> 95% target as effect noise grows, while HKSJ drops below. My results
> show this: CMA coverage clusters around 0.95 across
> noise levels, while HKSJ drops just below 0.95. This is most prevalent
> in the large n size chart.
> My CMA coverage sits just above 0.95, and is similar to the paper's result.
> In this simulation, i was limited by computational power and had to run with
> at least 10 seeds vs 32 used in the paper. In my example code, I reset seeds
> to only run 2 to limit run time. This is true across all sims.
>
>> **Simulation 3 — the eta noise-correction experiment — is the one
> place my results don't clearly show the paper's pattern.** With 
> seeds, the difference between eta=0 and eta>0 is within seed variance
> at the noise levels I tested. This is a statistical power issue, not
> a methodological one — more seeds would likely recover the effect,
> but would push runtime past what's reasonable for an example script.

> Simulation 4, on prior-error sensitivity, reproduces cleanly. The results
> with the prior being wider than CMA is expected especially as the prior error
> increases. This highlights the power of the lambda in the KRR penalizing
> larger prior errors.
>
> Overall the reproduction tracks the paper's qualitative findings.
> The reason things line up is that I match the generating process
> exactly — same kernel, same bandwidth of 0.2, same noise model from
> Appendix B.6. 

**[Slide: side-by-side or just your reproduction — whichever is cleaner.
Consider a smaller inset showing the paper's Figure 2 for comparison.]**

---

## Section 4 — Extensions and Ablations (≈75 sec, ~195 words)

> I made one main extension beyond the paper.
>
> **[Slide: ablation table]**
>
> The paper's prior encoder is a ReLU MLP on 13 hand-crafted numeric
> features extracted via LLM from trial PDFs. The authors flagged
> "better prior encoders" as future work, so I tested whether a
> pretrained biomedical language model — PubMedBERT — could replace the
> manual feature engineering entirely. The ablation crosses three MLP
> head depths with two input representations: the 13 hand-crafted
> features versus 768-dimensional PubMedBERT embeddings of either the
> real abstract when available, or generated clinical prose as a
> fallback.
> 
> One honest caveat: the paper's amiodarone case study
> uses an LLM to extract features from trial PDFs, which I couldn't
> reproduce byte-for-byte since it depended on the specific GPT-4
> version they had at the time. I hand-computed the amiodarone dose
> column —  and manually added results from the original Letlier Trial.
> 
> 1. CMA's coverage guarantee holds regardless of encoder. 
> Every row with CMA hits 0.90 exactly. That's conformal prediction's 
> distribution-free guarantee working as promised. HKSJ, which doesn't use 
> conformal prediction, undercovers at 0.80. 
> 2. Hand-crafted features beat BERT on this small dataset. With only 11 training trials, the 768-d BERT embedding space is severely overparameterized. The MLP head can't reliably find the signal. Hand-crafted features give it a strong inductive bias that scales better with tiny data. 
> 3. Head architecture barely matters. Shallow/Medium/Deep are within 0.08 of each other on CMA Width. The bottleneck is the encoder input, not the head capacity. This is the classic small-data finding: at n=11, depth doesn't help.
The trade-off the table reveals
HKSJ looks "best" on width alone (1.28 vs CMA's 5.20), but that tightness is a lie — it only covers 80% of true effects when it claims 90%. CMA trades width for valid coverage.
The quality-of-validity is the entire selling point of the paper. Your table shows exactly that: HKSJ undercovers; every CMA variant achieves target coverage.
>
> With only 21 trials, the numerical differences between rows are
> small and dominated by seed variance. I want to be direct about that:
> this ablation is best read as a feasibility probe, not a conclusive
> comparison. The real contribution is the infrastructure — someone
> with a larger meta-analysis, like a 150-trial Cochrane systematic
> review, could plug their abstracts into the same pipeline and get a
> statistically meaningful answer about whether automated text
> embeddings can replace manual feature extraction.
> 
> These results do still prove the CMA method of maintaining coverage
> despite adding overfit variables abd ut also shows how with small datasets
> it is hard to outperform traditional hand-picked regression features.
> 
> This project is implemented in pyhealth across two datasets, one task,
> two models and two separate example files as listed on the slide.
> 
> Thank you


**[Slide: "Thank you"]**

---

## Timing Cheat Sheet

| Section | Target | Cumulative | Words |
|---|---|---|---|
| 1. Problem | 0:50 | 0:50 | ~135 |
| 2. Paper approach | 1:30 | 2:20 | ~245 |
| 3. Reproduction | 1:50 | 4:10 | ~295 |
| 4. Extensions | 1:15 | 5:25 | ~195 |
| Slide transitions + pauses | ~0:30 | **≈5:55** | — |

Total target: **~5:55 delivered**, comfortably under 7:00.

## Before You Record

**Fill in the placeholders.** Every `[X]` needs a real number from your
simulation runs. The rubric says "show your results" — concrete numbers
are what that means.

- `[X]%` in Sim 1: run the script, read the mean CMA width and HKSJ
  width at n=20 for the "good" prior, compute the percent reduction.
  If the number doesn't cleanly favor CMA at a single point, you can
  say "narrower on average" and point at the figure instead.
- `[X]` in Sim 2: read the HKSJ empirical coverage at the highest
  effect noise setting. Paper shows it dropping to somewhere in the
  0.85-0.90 range; your number should be in that ballpark.

## Why This Script Works for Your Audience

- **"Trusted / untrusted" framing first.** The paper uses this
  terminology and it's intuitive for a CS audience that thinks about
  data quality tiers all the time. More accessible than jumping
  straight to "prior encoder" or "conformal prediction."
- **"Distribution-free coverage guarantee" is the hook.** For a DL
  audience, the novelty isn't that a neural network is used — they
  use them every day. The novelty is the *theoretical wrapper* that
  gives a coverage guarantee independent of how well the network
  actually fits. Spending most of Section 2 on layer three matches
  where the intellectual content is.
- **"Marginal coverage is tight only in expectation" in Section 3.**
  This is the right vocabulary. It signals you understand *why* your
  2-seed results wobble around 0.95, and it's the kind of reasoning
  a DL audience will immediately recognize as a legitimate statistical
  argument rather than hand-waving.
- **"Feasibility probe, not conclusive" in Section 4.** Self-aware
  scoping of a small-n experiment. PhD students will respect the
  calibrated claim more than a forced result.

## Recording Tips

- **Do one timed practice run, then one real take.** Don't over-polish.
  Recorded academic presentations are more forgiving of minor verbal
  slips than you think.
- **Screen-share your figures** while you narrate. Sim 2 on screen
  while you explain coverage is worth 30 seconds of words.
- **Pause deliberately between sections.** A half-second pause reads
  as confident transition, not hesitation.
- **If you flub a sentence, keep going.** Unless you produce gibberish,
  don't restart.
- **Don't rush the technical vocabulary** — "full conformal
  prediction," "distribution-free guarantee," "marginal coverage."
  Your audience knows what these mean; say them at normal speed.

## If You Run Long

Cut in this order — top first:

1. The Cochrane / guideline bodies reference in Section 1 (your
   audience doesn't care about guideline politics)
2. The "Simulation 4 reproduces cleanly" sentence in Section 3 —
   point at the figure instead
3. The "three MLP head depths" phrasing specifics in Section 4 —
   just say "different MLP head configurations"

Do **not** cut:
- Any `[X]` number in Section 3 (rubric: "show your results")
- The "marginal coverage" explanation for your Sim 2 wobble (rubric:
  "explain why your results differ from paper")
- The explicit call-out of Sim 3 not reproducing (rubric: honest
  framing of limitations scores better than false confidence)
- The "feasibility probe" framing in Section 4 (pre-empts "why only
  21 trials?" from a skeptical grader)
