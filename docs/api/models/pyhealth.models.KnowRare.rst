pyhealth.models.KnowRare
========================

Overview
--------

KnowRare (Zhu et al., `Bridging data gaps of rare conditions in ICU: a
multi-disease adaptation approach for clinical prediction
<https://arxiv.org/abs/2507.06432>`_, npj Digital Medicine 2025) predicts
clinical outcomes for ICU patients whose primary condition is rare. It
borrows data from clinically similar conditions, chosen with a condition
knowledge graph, and aligns them with the target condition by joint
adversarial domain adaptation:

1. :class:`~pyhealth.models.knowrare.ConditionKnowledgeGraph` links
   conditions by diagnosis co-occurrence, similarity of their patients'
   records and shared drugs.
2. :class:`~pyhealth.models.knowrare.TuckER` embeds the graph, and
   :func:`~pyhealth.models.knowrare.select_source_conditions` picks the
   ``k`` conditions closest to each target rare condition.
3. :meth:`KnowRare.pretrain <pyhealth.models.KnowRare.pretrain>` pre-trains
   the encoder (an LSTM over the time series plus an MLP over static
   variables) on all conditions by next-step prediction.
4. For each target, a copy of the pre-trained model is fine-tuned on the
   target and its source conditions after
   :meth:`~pyhealth.models.KnowRare.set_domains`: a discriminator reads the
   latent representation and the predicted outcome and tries to tell the
   conditions apart, and the encoder learns to fool it through a gradient
   reversal layer. Each sample's loss is weighted by the inverse prevalence
   of its condition.

This is a clean reimplementation from the paper: the authors' code
(https://github.com/JasonZuu/KnowRare) is licensed CC BY-NC-ND 4.0, which
does not allow derivative works, so it was only read to confirm details.
The module docstring lists where this implementation and that code differ.

Inputs and outputs
------------------

The model reads a dense ``[T, F]`` time series (``temporal_key``), an
optional static vector (``static_key``) and each sample's condition
(``condition_key``, a plain string field). The task
:class:`~pyhealth.tasks.KnowRareICUMortalityMIMIC4` provides all three, plus
the ``diagnoses`` and ``drugs`` fields the knowledge graph is built from.
Binary, multiclass and multilabel labels are supported; ``forward`` returns
``loss``, ``y_prob``, ``y_true``, ``logit`` and ``embed`` and also works
without labels. Because one model is fine-tuned per rare condition, the
example notebook ``examples/knowrare_mimic4_icu_mortality.ipynb`` shows the
whole loop with :class:`~pyhealth.trainer.Trainer`.

API
---

.. autoclass:: pyhealth.models.KnowRare
   :members:
   :show-inheritance:

.. autoclass:: pyhealth.models.knowrare.ConditionKnowledgeGraph
   :members:

.. autoclass:: pyhealth.models.knowrare.TuckER
   :members:

.. autofunction:: pyhealth.models.knowrare.select_source_conditions
