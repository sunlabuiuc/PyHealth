pyhealth.models.TPC
===================

Temporal Pointwise Convolution (TPC) model for ICU remaining length-of-stay prediction.

Overview
--------

The TPC model combines grouped temporal convolutions with pointwise (1x1) convolutions to 
capture both feature-specific temporal patterns and cross-feature interactions at each timestep. 
The architecture is specifically designed for irregularly sampled multivariate time series in 
intensive care settings.

**Paper Reference:**
Rocheteau, E., Liò, P., & Hyland, S. (2021). Temporal Pointwise Convolutional Networks for 
Length of Stay Prediction in the Intensive Care Unit. In Proceedings of the Conference on 
Health, Inference, and Learning (CHIL).

**Key Features:**

- Grouped temporal convolutions (one group per clinical feature)
- Pointwise convolutions for cross-feature learning
- Skip connections with hierarchical feature aggregation
- Custom MSLE (Masked Mean Squared Logarithmic Error) loss
- Monte Carlo Dropout for uncertainty estimation (extension)

**Model Classes:**

.. autoclass:: pyhealth.models.TPC
    :members:
    :undoc-members:
    :show-inheritance:

**Loss Functions:**

.. autoclass:: pyhealth.models.MSLELoss
    :members:
    :undoc-members:
    :show-inheritance:

.. autoclass:: pyhealth.models.MaskedMSELoss
    :members:
    :undoc-members:
    :show-inheritance:
