# Person C narration and completion plan

Target: 5 minutes 40 seconds. This is rehearsal text for the available portions,
not a claim that a final video or clinical-results presentation exists.

## 0:00–0:20 — Problem
Our project studies hourly prediction of remaining ICU length of stay. This
can help with bed planning, but measurements are irregular, incomplete and
heterogeneous, and the target distribution is strongly skewed.

## 0:20–1:10 — Paper approach
Rocheteau and colleagues propose temporal pointwise convolutions. Temporal
convolutions learn how individual clinical features evolve, and pointwise
convolutions combine features. Skip connections retain information, while
missingness indicators represent observation recency. The paper also studies
a logarithmic loss and auxiliary mortality prediction. Our contribution focuses
on the hourly prediction task within PyHealth, using an existing LSTM baseline.

## 1:10–1:45 — Published findings
The paper reports improvements over LSTM and Transformer baselines, with
results depending on dataset and metric. Show the source table alongside the
relevant baseline when discussing a number. These published results are not
our experimental results. We have not implemented the TPC architecture.
Source: https://arxiv.org/html/2007.09483v4, Tables 2–4.

## 1:45–2:30 — Our contribution
We reuse PyHealth's eICU loader and YAIB feature extraction, keeping adults
with at least five hours in ICU. Each stay has static admission features,
hourly measurements and remaining-stay labels. We cap observation history at
14 days, not the remaining-stay target. We document changes from the original
paper, including the features, aggregation and baseline model.

## 2:30–3:30 — Implementation and tests
Show the pipeline diagram, a short CSV excerpt and the numerical oracle.
Five invented patients exercise all measurement tables, including missing and
invalid observations. Small tests cover cohort boundaries, static values,
labels, observation removal, forward fill and decay. We found and corrected
a timing bug: an hourly bin must be timestamped when complete so that a
prediction cannot read measurements from later in that hour. The integration
check loads the CSVs, fits processors only on the training patient, and runs
an LSTM update with finite gradients. Synthetic results validate the pipeline;
they do not demonstrate useful medical prediction.

## 3:30–5:00 — Results section to finish with A
Required inputs: actual dataset/subset size, patient split, seed(s), epochs,
loss, feature list, units and zero-target policy; 2-by-3 missingness table;
MAD and paper-compatible MSLE; corresponding paper LSTM baseline with exact
source; explanation of scope differences. State whether results are better,
worse or incomparable and why. Do not insert fabricated values. If real-data
experiments cannot be completed, honestly report the attempt and limitations;
synthetic wiring evidence cannot be presented as replication performance.

## 5:00–5:40 — Extension and limitations
Explain that the extension trains with extra missing observations and tests
robustness at three removal rates using shared masks. Decay removal alone
would repeat an existing paper ablation. Describe the findings supplied by A
without extrapolating beyond the tested cohort. End with limitations: reduced
feature set, different model/preprocessing, and limited compute/data scope.

## Recording and submission checklist
- Fill the results section and update slides; every numerical claim has a source.
- Rehearse once; finish between four and seven minutes.
- Record your own narration; verify audio, screen readability and runtime.
- Upload and test the public video URL in a signed-out browser.
- Include the requested Gradescope link and all group members.
- Each member completes the final Google Form using the Illinois account.
