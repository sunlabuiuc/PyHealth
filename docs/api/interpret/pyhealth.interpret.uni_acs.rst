pyhealth.interpret.uni_acs
==========================

Unified Auto Clinical Scoring (Uni-ACS) converts SHAP feature attributions from
binary classification models into integer scoring tables. Feature points
sum to a total score. In ``fit_from_data``, the per-feature logit
transformation of probability-scale SHAP values is approximate. Its sum
generally differs from the model's log-odds change. Held-out isotonic
calibration can map the integer total to a risk estimate.

Reference:
    Li, A., Ong, M. L., Oei, C. W., Lian, W., Phua, H. P., Htet, L. H.,
    Lim, W. Y., & Motani, M. (2022). *Unified Auto Clinical Scoring (Uni-ACS)
    with Interpretable ML models*. Proceedings of Machine Learning for
    Healthcare, PMLR 182:26–53.
    https://proceedings.mlr.press/v182/li22a.html

.. autoclass:: pyhealth.interpret.uni_acs.UniACS
    :members:
    :show-inheritance:

Functional API
--------------

.. autofunction:: pyhealth.interpret.uni_acs.find_novel_breakpoints

.. autofunction:: pyhealth.interpret.uni_acs.build_score_table

.. autofunction:: pyhealth.interpret.uni_acs.predict_score

.. autofunction:: pyhealth.interpret.uni_acs.predict_proba_from_score

.. autofunction:: pyhealth.interpret.uni_acs.make_clinical_score_table
