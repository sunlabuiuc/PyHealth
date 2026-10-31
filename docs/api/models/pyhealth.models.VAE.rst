pyhealth.models.VAE
===================================

Variational autoencoder for image and categorical EHR sequence inputs in a
``SampleDataset``. Image VAEs can condition generation on a categorical label;
sequence VAEs reconstruct and generate code tokens. The model optimizes input
reconstruction and latent regularization rather than predicting labels.

.. autoclass:: pyhealth.models.VAE
    :members:
    :undoc-members:
    :show-inheritance:
