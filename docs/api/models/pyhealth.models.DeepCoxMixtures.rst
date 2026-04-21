pyhealth.models.DeepCoxMixtures
===================================

Deep Cox Mixtures for survival regression (Nagpal et al., MLHC 2021). A shared
neural embedding feeds a softmax gate and ``k`` Cox experts; each expert
carries a non-parametric Breslow baseline hazard smoothed with a univariate
spline. Training alternates hard-assignment E-steps with gradient-descent
M-steps on the per-component Cox partial likelihood.

Paper: https://proceedings.mlr.press/v149/nagpal21a/nagpal21a.pdf

.. autoclass:: pyhealth.models.DeepCoxMixtures
    :members:
    :undoc-members:
    :show-inheritance:
