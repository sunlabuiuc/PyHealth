pyhealth.models.Labrador
========================

Labrador implements a lab-centric architecture for jointly modeling
categorical lab codes and continuous lab values, with optional masked-language
modeling (MLM) and downstream classification.

Paper
-----

Bellamy et al., *Labrador: Exploring the Limits of Masked Language Modeling
for Laboratory Data* (ML4H 2024):
https://arxiv.org/abs/2312.11502

Main model
----------

.. autoclass:: pyhealth.models.labrador.LabradorModel
   :members:
   :undoc-members:
   :show-inheritance:

Supporting modules
------------------

.. autoclass:: pyhealth.models.labrador.LabradorEmbedding
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: pyhealth.models.labrador.LabradorValueEmbedding
   :members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: pyhealth.models.labrador.LabradorMLMHead
   :members:
   :undoc-members:
   :show-inheritance:

Example
-------

See ``examples/mimic4_mortality_labrador.py`` for a compact ablation-style
usage script with hyperparameter variants.
