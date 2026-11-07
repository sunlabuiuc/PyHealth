Neural Fine-Gray
================

.. automodule:: pyhealth.models.neural_fine_gray
    :members:
    :show-inheritance:

For event ``r``, the cumulative incidence is

.. math::

    F_r(t \mid x) = b_r(h) \left(1 - \exp[-t m_r(t,h)]\right),
    \quad h=E(x), \quad \sum_r b_r(h)=1.

Squared weights and increasing activations keep time heads monotonic.
Automatic differentiation supplies event densities without numerical
integration. Censored records contribute log event-free probability.
The likelihood assumes independent censoring conditional on inputs.

Use ``Trainer`` with ``monitor="loss"`` and ``monitor_criterion="min"``.
The model opts out of dataset label processors, so ``mode=None`` and
``Trainer`` evaluates only its joint first-event loss, not classification metrics.
Labeled ``torch.no_grad`` evaluation works; labeled ``inference_mode`` does not.
Call ``eval()`` before time-grid prediction.

Reference: `Jeanselme et al., CHIL 2023
<https://proceedings.mlr.press/v209/jeanselme23a.html>`_.
The runnable PBC study and supplementary ablations are in
``examples/pbc_neural_fine_gray/``. Its README describes installation, local
data access, training-only model selection, and scoring conventions.
