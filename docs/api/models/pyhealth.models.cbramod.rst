pyhealth.models.CBraMod_Wrapper
===================================

CBraMod model for EEG signal classification.

Overview
--------

CBraMod is a criss-cross attention transformer tailored for EEG decoding. The
wrapper integrates the model into the PyHealth ``BaseModel`` pipeline so it can
be trained with the standard ``Trainer`` APIs.

Input/Output
------------

- **Input:** ``signal`` tensor shaped ``(batch, channels, timesteps)`` where
  ``timesteps`` is a multiple of 200 (the patch size used by CBraMod).
- **Output (classifier_head=True):** dict with ``loss``, ``y_prob``, ``y_true``,
  ``logit``, and ``embeddings``.
- **Output (classifier_head=False):** dict with ``logit`` and ``embeddings``.

Example Usage
-------------

.. code-block:: python

    import torch
    from pyhealth.datasets import create_sample_dataset, get_dataloader
    from pyhealth.models import CBraMod_Wrapper

    n_channels = 16
    patch_size = 200
    n_patches = 10
    n_samples = patch_size * n_patches

    samples = [
        {
            "patient_id": f"patient-{i}",
            "visit_id": "visit-0",
            "signal": torch.randn(n_channels, n_samples).numpy().tolist(),
            "label": i % 6,
        }
        for i in range(8)
    ]

    dataset = create_sample_dataset(
        samples=samples,
        input_schema={"signal": "tensor"},
        output_schema={"label": "multiclass"},
        dataset_name="test_cbramod",
    )

    model = CBraMod_Wrapper(
        dataset=dataset,
        seq_len=n_patches,
        n_classes=6,
        classifier_head=True,
    )

    batch = next(iter(get_dataloader(dataset, batch_size=2, shuffle=True)))
    output = model(**batch)
    print(output["logit"].shape)

.. autoclass:: pyhealth.models.CBraMod_Wrapper
    :members:
    :undoc-members:
    :show-inheritance:
