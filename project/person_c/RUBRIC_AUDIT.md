# Work-distribution compatibility audit — November 17, 2026

## Verdict
The distribution is compatible with **Option 3: standalone task (50 points)**.
The rubric expressly permits partial reproduction using an existing dataset.
Reusing eICUDataset and an existing LSTM is legitimate. The proposal's original
TPC-model scope is reduced; explain that change rather than claiming TPC was
implemented. Existing dataset + task + existing LSTM is not a new model and
does not establish the full-pipeline extra credit.

This audit uses the supplied rubric (final requirements on PDF pages 7–14),
work distribution (Stages 0–4), project draft, repository code and the sources
below. It is a readiness assessment, not a guaranteed grade.

## Checklist
| Rubric area | Evidence/status | Owner / next action |
|---|---|---|
| Core task (20) | BaseTask inherited via YAIB; cohort, static fields, hourly labels, schemas, masking, optional decay, validation and docs | B/C review exact feature/timing contract |
| Required files (12) | task, RST, index, tests and examples/eicu_hourly_los_lstm.py now exist | A replace/extend synthetic example with final experiment |
| Tests (11) | Small in-memory core cases; real-loader/LSTM smoke check separately opt-in | C completed available coverage; rerun after B changes |
| Ablation (5) | Runnable 2 training × 3 test missingness configurations and measured synthetic results | A must supply meaningful eICU/demo experiment and findings |
| PR formatting (7) | PR_DRAFT.md contains scope, paper link, file guide, validation, limitations | C fill missing names/NetIDs and actual final results |
| Google Form (15) | AI_USAGE.md records assistance and validation honestly | Each member reviews and submits with Illinois account |
| Video (10) | Background/implementation slides and timed outline exist | C add A's results, record 4–7 minutes, upload public link |

The five PR subsections shown in the rubric sum to 55 despite the stated
50-point total. Do not normalize or reinterpret that discrepancy; address
all listed requirements and leave scoring to the course staff.

## Corrections and risks
1. The work distribution's entire-suite-under-one-second target is stricter
   than the rubric's milliseconds per test / seconds per suite language.
   The rubric also warns about tests >1 second. Core tests are fast; the real
   loader/model check is explicitly opt-in, as the contribution guide permits.
   Startup/import time is still reported, never hidden as sub-second execution.
2. Stage 0 asks for 2–3 deliberately designed patients; the rubric permits 2–5.
   Five invented patients for the rich example are acceptable. Small isolated
   unit fixtures cover each boundary; the rich CSV example is not the default
   fast test. The contribution guide suggests 5–20 events for tiny tests, so
   avoid promoting the full multi-table example to a millisecond unit test.
3. Floor-hour bins previously exposed observations later in the same hour to
   that hour's label. Fixed: bin [b,b+1) is available only at b+1; at hour 5,
   only completed bins 0–4 are used. Dedicated regression test added.
4. The original paper already ablates decay, diagnoses, temporal/pointwise
   components, loss functions, and feature groups. Merely toggling decay is
   NOT a novel extension for this assignment. Its documented experiments do
   not include the proposed two-training-by-three-test random-cell-removal
   grid; this is a defensible proposed extension relative to that paper, not
   a claim of novelty across all literature.
5. The proposal describes raw-observation deletion and TPC training; the
   current implementation masks aggregated hourly cells and trains LSTM.
   Record both changes. Mean/right-edge binning, original features, original
   normalization, and original MSLE are not all reproduced here.
6. The smoke test uses MSE on train-scaled targets. The example reports MAD in
   days and explicitly named log1p-MSE in days. It is not the paper's log-MSE.
   Do not compare those logarithmic metrics numerically as if equivalent.
7. Course PDF says target main; current PyHealth contribution guide says
   master. The team fork uses master. Resolve the upstream target with course
   staff/current repository before submitting; do not invent a main branch.
8. A team PR must include all names and NetIDs/emails, original paper link,
   maintainer edits, current upstream rebase, and Gradescope group membership.
   A local branch or a PR only to the team fork is not the final submission.
9. The guide routes files outside approved code/docs/example/test directories
   (including project/person_c handoff material) to manual review. That is not
   rejection, but curate the upstream diff with B or accept manual review.
10. The current task depends on YAIB helper internals. Rebase against the
    actual upstream target and rerun the full task tests before posting.

## Person C: available work completed in this audit
- Added rubric-to-evidence mapping and explicit remaining dependencies.
- Fixed within-hour look-ahead and added regression coverage.
- Added the required named synthetic ablation example with measured results.
- Marked the heavyweight integration check opt-in while retaining the command
  and recorded proof that the real pipeline passes.
- Prepared PR draft, AI-use record, video narration and final submission gate.

## Cannot be completed honestly right now
- Final real-data ablation conclusions: need A's experiments.
- Verify B's unseen final code or obtain teammate agreement: need their code/review.
- Final public presentation: needs experiment results and recorded narration.
- Individual Google Form answers/submission: require each member's review.
- Publish the Git branch: prior push failed authentication; no new credentials
  have been supplied. Local commits and patch remain available.
- Fill A/B contributor identifiers: left blank at the user’s request. C is Xiao Luo, xiaoluo6.

Sources:
- Supplied Fall26CS598 DL4H Project Instructions and Rubric.pdf, pages 7–14.
- Supplied Work distribution.pdf, Stages 0–4.
- Supplied Fall26CS598 DL4H Final Project Draft.pdf, methodology/extension plan.
- https://pyhealth.readthedocs.io/en/latest/how_to_contribute.html
- https://arxiv.org/html/2007.09483v4 (Sections 6.2, 7.1 and Appendix E).
