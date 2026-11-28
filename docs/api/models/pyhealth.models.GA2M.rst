pyhealth.models.GA2M
=====================

Doctor-interpretable Generalized Additive Model with pairwise interactions
(GA2M), reproducing Hegselmann et al. (2020), "An evaluation of the
doctor-interpretability of generalized additive models with interactions"
(PMLR 126). Each additive 1D/2D term is a lookup table of learnable weights
over quantile bins, trained jointly by ordinary SGD through
:class:`~pyhealth.trainer.Trainer` -- unlike InterpretML's
``ExplainableBoostingClassifier``, which fits terms by cyclic gradient
boosting.

A dedicated reserved bin represents a missing feature value, so missing and
observed values never share a bin and no imputation or z-score
normalization is required upstream. ``get_shape_functions()`` exposes each
term's learned curve/surface, centered for interpretability, for visual
inspection by a clinician.

See ``examples/mimic3_mortality_ga2m.py`` for the full set of experiments
(bin-count ablation, the paper's two-phase pairwise-interaction
selection, and the clinician-pruning ablation) against
:class:`~pyhealth.tasks.InHospitalMortality48hMIMIC3`.

.. autoclass:: pyhealth.models.GA2M
    :members:
    :undoc-members:
    :show-inheritance:

.. autofunction:: pyhealth.models.ga2m.rank_pairs_by_contribution_variance

.. autofunction:: pyhealth.models.ga2m.all_pairs
