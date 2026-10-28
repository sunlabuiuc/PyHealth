pyhealth.models.GAN
===================================

A Generative Adversarial Network built from a dataset with a single input
feature, whose processor picks the mode. With a ``multi_hot`` feature (for
example from :class:`~pyhealth.tasks.EHRCodeSetGenerationMIMIC3`), the GAN
learns to generate synthetic patients as sets of codes with MLP networks. With
a ``tensor`` feature of continuous values (for example from
:class:`~pyhealth.tasks.CARTVitalsGenerationEICU`), it generates new vectors in
the data's original units. With an ``image`` feature whose ``image_size`` is 32, 64 or 128 (for example from
:class:`~pyhealth.tasks.COVID19CXRClassification`), it generates small images
with CNN networks. ``examples/gan_tutorial.ipynb`` walks through both.

The PyHealth ``Trainer`` does not apply to GAN: train it with
:meth:`~pyhealth.models.GAN.train_model`, then call
:meth:`~pyhealth.models.GAN.generate`, which returns synthetic patients in
code-set mode and a tensor in continuous and image mode.

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
