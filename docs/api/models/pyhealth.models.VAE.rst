pyhealth.models.VAE
===================================

Variational autoencoder for image and categorical EHR sequence inputs in a
``SampleDataset``. Image VAEs can be explicitly conditioned on a binary or
multiclass label; labels are ignored unless ``condition_key`` is provided.
Sequence VAEs reconstruct and generate code tokens. The image encoder, decoder,
module names, and ``hidden_dim`` latent size remain compatible with the merged
image VAE, including its image ``state_dict`` keys.
The image ``encode()`` and ``decode()`` methods remain available. Sequence
models add separate recurrent modules and use ``sequence_hidden_dim``. If an
``ImageProcessor`` has no mode, the channel count is inferred from a processed
sample; image processors must return tensors (``to_tensor=True``).

Sequence reconstruction reuses ``TimeseriesMultiClassLabelProcessor`` to pack
variable-length token targets and gather per-token predictions through the
standard ``Trainer``. The processor's ordered positions represent token order,
not elapsed clinical time. The autoregressive sequence decoder is teacher-forced
during training, so it may rely less on the latent than intended (posterior
collapse). Reconstruction quality alone does not establish that the latent is
useful for generation; this implementation does not currently use KL annealing
or word dropout.

.. autoclass:: pyhealth.models.VAE
    :members:
    :undoc-members:
    :show-inheritance:
