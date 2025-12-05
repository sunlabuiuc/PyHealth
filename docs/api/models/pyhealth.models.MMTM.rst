pyhealth.models.MMTM
===================================

MMTM: Multimodal Transfer Module for Electronic Health Records.

Paper: Hamid Reza Vaezi Joze, Amirreza Shaban, Michael L Iuzzolino, and Kazuhito Koishida. 2020.
*Multimodal Transfer Module for CNN Fusion*. 
In IEEE Conference on Computer Vision and Pattern Recognition (CVPR), 2020.
https://arxiv.org/abs/1911.08670

MMTM is a lightweight and effective module for multimodal fusion.  
It computes cross-modal channel-wise attention to enhance information
exchange between modalities without requiring large parameter growth.

In the context of Electronic Health Records (EHR), modalities may include:
- Diagnosis codes
- Procedure codes
- Medication codes
- Laboratory results
- Clinical note embeddings

This PyHealth module provides two components:

- **MMTMLayer**: A standalone multimodal attention block for fusing
  two embeddings via shared channel-wise importance.
- **MMTM**: A full PyHealth model that embeds two modalities,
  pools patient-level representations, performs MMTM fusion,
  and predicts clinical outcomes.

The MMTM fusion mechanism is especially useful when modalities have
different levels of predictive signal and complementary latent structure,
as it allows cross-modal exchange of importance weights in a parameter-efficient way.

This implementation is adapted from the concepts in the original CVPR 2020 paper
and integrated into the PyHealth modeling interface for EHR tasks.

.. autoclass:: pyhealth.models.MMTMLayer
    :members:
    :undoc-members:
    :show-inheritance:

.. autoclass:: pyhealth.models.MMTM
    :members:
    :undoc-members:
    :show-inheritance:
