pyhealth.models.GAN
===================================

A Generative Adversarial Network with two modes. In code-set mode, pass a
dataset whose single input feature is ``multi_hot`` (for example from
:class:`~pyhealth.tasks.EHRCodeSetGenerationMIMIC3`) and the GAN learns to
generate synthetic patients as sets of codes with MLP networks; see
``examples/gan_mimic3.ipynb``. In image mode, pass ``input_channel`` and
``input_size`` (32, 64 or 128) to generate small images with CNN networks; see
``examples/cxr/ChestXray-image-generation-GAN.ipynb``.

The PyHealth ``Trainer`` does not apply to GAN: train it with
:meth:`~pyhealth.models.GAN.train_model`, then call
:meth:`~pyhealth.models.GAN.generate` (code-set mode) or
:meth:`~pyhealth.models.GAN.generate_fake` (image mode).

Reference:
    Goodfellow, I., Pouget-Abadie, J., Mirza, M., Xu, B., Warde-Farley, D.,
    Ozair, S., Courville, A., & Bengio, Y. (2014).
    *Generative Adversarial Nets.*
    In Advances in Neural Information Processing Systems (NeurIPS) 2014.
    https://arxiv.org/abs/1406.2661

.. autoclass:: pyhealth.models.GAN
    :members:
    :undoc-members:
    :show-inheritance:
