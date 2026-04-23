pyhealth.models.taskaug_resnet
==============================

Overview
--------
Replication of the TaskAug framework from Raghu et al. (2022) *Data
Augmentation for Electrocardiograms* (CHIL, PMLR 174). A K-stage
differentiable augmentation policy (:class:`TaskAugPolicy`) selects among
seven ECG-specific operations via Gumbel-Softmax with class-specific learnable
magnitudes. The policy is trained jointly with a 1-D ResNet-18 backbone
(:class:`_ResNet1D`) using a bi-level optimisation scheme: the inner loop
updates backbone weights on augmented training data while the outer loop
updates policy weights on clean validation loss.

Paper: https://proceedings.mlr.press/v174/raghu22a.html

API Reference
-------------

.. autoclass:: pyhealth.models.taskaug_resnet.TaskAugResNet
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: pyhealth.models.taskaug_resnet.TaskAugPolicy
   :members:
   :undoc-members:
   :show-inheritance:
