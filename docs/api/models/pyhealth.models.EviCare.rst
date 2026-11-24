pyhealth.models.EviCare
=======================

Overview
--------

EviCare (Zhang et al., `EviCare: Enhancing Diagnosis Prediction with Deep
Model-Guided Evidence for In-Context Reasoning
<https://arxiv.org/abs/2604.10455>`_, KDD 2026,
`doi:10.1145/3770854.3780257 <https://doi.org/10.1145/3770854.3780257>`_)
predicts the CCS diagnosis categories of a patient's next visit with an
off-the-shelf LLM and no fine-tuning. A trained deep model, the *guide*,
supplies evidence that is written into the prompt:

1. **Candidate selection**: the top-``k`` CCS codes by guide logit (the paper
   uses ``k = 50``). In the ``"novel"`` setting, codes already in the
   patient's history are not candidates.
2. **Evidential prioritization**: the historical CCS codes are sorted by guide
   logit, and each historical ICD diagnosis is listed under its CCS parent in
   that order (paper Eqs. 7-8).
3. **Relational evidence**: ``G = A^T A``, where ``A`` is the binary
   patient-by-CCS matrix of the *training* patients, counts how many patients
   have both of two codes (:class:`~pyhealth.models.CooccurrenceGraph`, Eq. 9).
   Each candidate that is new for the patient is linked to the historical code
   ``argmax_h G[h, c]`` (Eq. 10).
4. **Prompt and re-ranking**: the LLM sees the visit(s) with their time gap,
   the prioritized diagnoses, the relations and the candidates, and replies
   ``Answer: <CCS 1>, <CCS 2>, ...``.

The reply is parsed defensively (:func:`~pyhealth.models.evicare.parse_answer`):
only candidates are kept, by name or code, duplicates are dropped and the
remaining codes follow in guide order
(:func:`~pyhealth.models.evicare.merge_ranking`). A malformed reply, or an
LLM call that raises, therefore falls back to the guide's ranking.

The LLM is any callable ``llm(system, prompt) -> str``:
:class:`~pyhealth.autoimprove.ClaudeLLM`,
:class:`~pyhealth.autoimprove.OpenAILLM` (also for vLLM, SGLang and other
OpenAI-compatible servers) or a plain function.

Usage
-----

:class:`~pyhealth.models.EviCare` is a :class:`~pyhealth.models.BaseModel`
whose ``forward`` is the guide's, so the guide trains with the usual
:class:`~pyhealth.trainer.Trainer`. Then:

.. code-block:: python

   task = EviCareDiagnosisPredictionMIMIC3(setting="novel")
   samples = dataset.set_task(task)
   model = EviCare(samples, setting="novel", ccs_names=task.ccs_mapping.ccs_names)
   Trainer(model=model).train(train_dataloader=..., epochs=...)
   model.fit_cooccurrence(train)          # training patients only
   guide_only = model.predict(test)       # the deep-model baseline
   evicare = model.predict(test, llm=OpenAILLM(model="Qwen/Qwen3-8B",
                                               base_url="http://localhost:8000/v1"))
   model.evaluate(evicare, ks=(5, 10))    # visit-level P@k, code-level Acc@k

The samples come from
:class:`~pyhealth.tasks.EviCareDiagnosisPredictionMIMIC3` or
:class:`~pyhealth.tasks.EviCareDiagnosisPredictionMIMIC4`; their
``history_icd_names``, ``history_ccs`` and ``days_ago`` fields carry the
evidence and are not seen by the guide. See
``examples/diagnosis_prediction/evicare_mimic3.py`` for the full pipeline,
including the paper's decoding settings and an offline fake-LLM mode.

Metrics follow the paper's Appendix A:
:func:`~pyhealth.models.evicare.precision_at_k_visit` divides the hits of
each visit by ``min(k, |truth|)`` and averages over visits;
:func:`~pyhealth.models.evicare.accuracy_at_k_code` divides all hits by all
true codes. Both take ``histories`` for the novel variants.

.. note::

   **Scope.** This implements EviCare(R), with RETAIN (embedding size 16) as
   the default guide. Any PyHealth model that returns ``"logit"`` per label
   can be the guide. BoxLM, the guide of the paper's main variant EviCare(B),
   is not included.

   **Differences from the paper and the official code**
   (`github.com/zhyccc/EviCare <https://github.com/zhyccc/EviCare>`_):

   - The official repository has no license file, so this is an independent
     re-implementation; no code or prompt text is copied from it.
   - The prompt wording is my own, under the section headers of the paper's
     prompt skeleton (Section 3.5). As in the official code, the LLM sees and
     returns CCS *names*; the parser also accepts CCS codes.
   - Relational evidence is also added in the ``"overall"`` setting, for the
     candidates that are new for the patient: the paper describes it for
     novel diagnoses, but its ablation (Table 4) shows it changing the
     overall scores. Set ``use_relations=False`` to turn it off.
   - Relational evidence follows Eq. 10 (one historical partner per
     candidate). The official code instead links each historical code to its
     top partner and renders an ICD -> CCS -> candidate chain.
   - ``G`` uses raw patient counts, as written in Eq. 9 (the paper describes
     no normalization or threshold), so very common historical codes are the
     partner of many candidates. Ties go to the higher-priority historical
     code, and candidates that co-occur with no historical code get no
     relation.
   - In the ``"novel"`` setting, seen codes are removed *before* taking the
     top ``k``, so there are ``k`` novel candidates when the vocabulary
     allows it (the paper does not specify the order).
   - Sampling, ontology and data handling follow
     :mod:`pyhealth.tasks.evicare_diagnosis`, whose page lists the
     differences there (one sample per visit, the skipped first visit, CCS
     versus CCSR).
   - The paper's numbers are not reproduced here (no credentialed MIMIC data
     or GPU were available while writing it).

API
---

.. autoclass:: pyhealth.models.EviCare
    :members:
    :undoc-members:
    :show-inheritance:

.. autoclass:: pyhealth.models.EviCarePrediction

.. autoclass:: pyhealth.models.CooccurrenceGraph
    :members:
    :undoc-members:

.. autofunction:: pyhealth.models.evicare.relational_evidence

.. autofunction:: pyhealth.models.evicare.build_prompt

.. autofunction:: pyhealth.models.evicare.parse_answer

.. autofunction:: pyhealth.models.evicare.merge_ranking

.. autofunction:: pyhealth.models.evicare.precision_at_k_visit

.. autofunction:: pyhealth.models.evicare.accuracy_at_k_code

.. autofunction:: pyhealth.models.evicare.rank_codes

.. autofunction:: pyhealth.models.evicare.select_candidates

.. autofunction:: pyhealth.models.evicare.eligible_codes

.. autofunction:: pyhealth.models.evicare.prioritize_history

.. autofunction:: pyhealth.models.evicare.format_time_gap

.. autodata:: pyhealth.models.evicare.SYSTEM_PROMPT
