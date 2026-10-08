pyhealth.models.CorGAN
===================================

CorGAN: a Correlation-capturing Convolutional GAN for synthetic EHR generation.
A 1D-CNN (or linear) autoencoder captures local code correlations, and a WGAN
generator/critic are trained in the autoencoder's latent space. Ported from the
reference implementation
(`cor-gan <https://github.com/astorfi/cor-gan>`_) and wrapped as a PyHealth
:class:`~pyhealth.models.BaseModel`.

Reference:
    Torfi, A., & Fox, E. A. (2020).
    *CorGAN: Correlation-Capturing Convolutional Generative Adversarial
    Networks for Generating Synthetic Healthcare Records.*
    In Proceedings of the 33rd International FLAIRS Conference.
    https://arxiv.org/abs/2001.09346

.. autoclass:: pyhealth.models.CorGAN
    :members:
    :undoc-members:
    :show-inheritance:
