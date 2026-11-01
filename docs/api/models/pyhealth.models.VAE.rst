pyhealth.models.VAE
===================================

Variational autoencoder for image and categorical EHR sequence inputs in a
``SampleDataset``. Image VAEs can be explicitly conditioned on a binary or
multiclass label; labels are ignored unless ``condition_key`` is provided.
Sequence VAEs reconstruct and generate code tokens. The original image-VAE
meaning of ``hidden_dim`` as the latent size is preserved, and new architecture
widths use ``channel_dim`` and ``sequence_hidden_dim``.

.. autoclass:: pyhealth.models.VAE
    :members:
    :undoc-members:
    :show-inheritance:
