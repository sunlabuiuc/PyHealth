pyhealth.models.TarDiff
===================================

TarDiff: target-oriented diffusion for synthetic clinical time series. A
class-conditional diffusion model (a 1D U-Net, sampled with DDIM) generates
multivariate series of shape ``(channels, length)``. Influence guidance steers
each sampling step towards series that would most reduce a trained downstream
classifier's loss on a guidance set, so the synthetic data is aimed at that
model's task rather than only at matching the training distribution. Rebuilt
from the paper and wrapped as a PyHealth :class:`~pyhealth.models.BaseModel`.

Reference:
    Deng, B., Xu, C., Li, H., Huang, Y., Hou, M., & Bian, J. (2025).
    *TarDiff: Target-Oriented Diffusion Guidance for Synthetic Electronic
    Health Record Time Series Generation.*
    https://arxiv.org/abs/2504.17613

.. autoclass:: pyhealth.models.TarDiff
    :members:
    :undoc-members:
    :show-inheritance:

.. autoclass:: pyhealth.models.generators.tardiff.UNet1D
    :members:
