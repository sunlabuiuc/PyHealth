pyhealth.models.MedGAN
===================================

MedGAN: a bag-of-codes Generative Adversarial Network for synthetic EHR
generation. An autoencoder is pre-trained on multi-hot patient records, then a
GAN with residual generator and minibatch-averaging discriminator is trained in
the autoencoder's latent space. Ported from the reference implementations
(`medgan <https://github.com/mp2893/medgan>`_ and its PyTorch reimplementation)
and wrapped as a PyHealth :class:`~pyhealth.models.BaseModel`.

Reference:
    Choi, E., Biswal, S., Malin, B., Duke, J., Stewart, W. F., & Sun, J. (2017).
    *Generating Multi-label Discrete Patient Records using Generative
    Adversarial Networks.*
    In Proceedings of Machine Learning for Healthcare (MLHC) 2017.
    https://arxiv.org/abs/1703.06490

.. autoclass:: pyhealth.models.MedGAN
    :members:
    :undoc-members:
    :show-inheritance:
