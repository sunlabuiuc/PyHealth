pyhealth.models.TaskAugResNet
==============================

TaskAug is a differentiable, task-adaptive data augmentation framework for
ECG binary classification. A K-stage Gumbel-Softmax augmentation policy
selects from eight time-series operations with class-specific learnable
magnitudes. The policy is trained jointly with a 1-D ResNet-18 backbone via
bi-level optimisation (inner loop: backbone on augmented training data; outer
loop: policy on clean validation loss).

Paper: Raghu et al. (2022). Data Augmentation for Electrocardiograms.
*Conference on Health, Inference, and Learning (CHIL)*, PMLR 174.
https://proceedings.mlr.press/v174/raghu22a.html

.. autoclass:: pyhealth.models.taskaug_resnet.TaskAugPolicy
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: pyhealth.models.taskaug_resnet._ResNet1D
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: pyhealth.models.TaskAugResNet
   :members:
   :undoc-members:
   :show-inheritance:
