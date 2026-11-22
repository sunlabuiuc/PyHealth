pyhealth.interpret.uni_acs
==========================

Unified Auto Clinical Scoring (Uni-ACS) converts SHAP feature attributions from
complex binary classification models into transparent, additive integer
clinical scoring tables suitable for bedside risk assessment.

Reference:
    Li, A., Ong, M. L., Oei, C. W., Lian, W., Phua, H. P., Htet, L. H.,
    Lim, W. Y., & Motani, M. (2022). *Unified Auto Clinical Scoring (Uni-ACS)
    with Interpretable ML models*. arXiv preprint.

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
