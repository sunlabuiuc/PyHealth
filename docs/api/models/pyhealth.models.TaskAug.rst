pyhealth.models.TaskAug
===================================

TaskAug: data augmentation policies for ECGs, learned per task and with
class-specific strengths
(`Raghu et al., CHIL 2022 <https://proceedings.mlr.press/v174/raghu22a.html>`_).
The classifier trains on augmented signals; the policy is trained on the
validation loss by implicit differentiation.

.. autoclass:: pyhealth.models.TaskAug
    :members:
    :undoc-members:
    :show-inheritance:

.. autoclass:: pyhealth.models.TaskAugPolicy
    :members:
    :undoc-members:
    :show-inheritance:

.. autoclass:: pyhealth.models.ResNet1D
    :members:
    :undoc-members:
    :show-inheritance:

.. autofunction:: pyhealth.models.train_taskaug

.. autofunction:: pyhealth.models.hyper_step
