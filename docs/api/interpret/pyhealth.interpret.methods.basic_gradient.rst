pyhealth.interpret.methods.basic_gradient
===========================================

Overview
--------

The ``BasicGradientSaliencyMaps`` class provides gradient-based saliency map visualization for 
PyHealth's image classification models. This interpretability method helps identify which regions 
of medical images most influenced the model's prediction by computing gradients of model outputs 
with respect to input pixels.

This method is particularly useful for:

- **Clinical interpretability**: Understanding which image regions drove a particular diagnosis
- **Model debugging**: Identifying if the model focuses on clinically relevant features
- **Trust and transparency**: Providing visual explanations for model predictions
- **Error analysis**: Comparing saliency maps for correct vs. incorrect predictions

The implementation computes saliency by taking the maximum absolute gradient across color channels
for each pixel, highlighting the most influential regions in the input image.

Key Features
------------

- **Dual input support**: Process batches from DataLoader or direct batch inputs
- **Flexible visualization**: Matplotlib overlay with configurable transparency
- **Label comparison**: Display both true labels and model predictions
- **Efficient storage**: Separate caching for different data sources

Usage Notes
-----------

1. **Gradients required**: Do not use within ``torch.no_grad()`` context
2. **Model compatibility**: Works with PyHealth image classification models
3. **Memory usage**: Limit batch count to control memory consumption
4. **Batch visualization**: Use ``batch_index`` for pre-computed maps, omit for on-the-fly computation

Quick Start
-----------

.. code-block:: python

    from pyhealth.interpret.methods.basic_gradient import BasicGradientSaliencyMaps
    from pyhealth.datasets import get_dataloader
    import matplotlib.pyplot as plt

    # Assume you have a trained image model and dataset
    model = TorchvisionModel(dataset=sample_dataset, ...)
    # ... train the model ...

    # Create interpretability object with dataloader
    dataloader = get_dataloader(dataset, batch_size=32, shuffle=True)
    saliency_maps = BasicGradientSaliencyMaps(
        model=model,
        dataloader=dataloader,
        batches=3
    )
    saliency_maps.init_gradient_saliency_maps()

    # Visualize from pre-computed maps
    saliency_maps.visualize_saliency_map(
        plt,
        image_index=10,
        batch_index=0,
        title="Gradient Saliency",
        id2label={0: "Normal", 1: "COVID", 2: "Pneumonia"},
        alpha=0.6
    )

Custom Batch Example
--------------------

.. code-block:: python

    import torch

    # Create a custom batch (e.g., filter by diagnosis)
    covid_samples = [s for s in dataset.samples if s['disease'].item() == covid_label]
    covid_batch = {
        'image': torch.stack([covid_samples[i]['image'] for i in range(32)]),
        'disease': torch.stack([covid_samples[i]['disease'] for i in range(32)])
    }

    # Initialize with custom batch
    saliency_maps = BasicGradientSaliencyMaps(model=model, input_batch=covid_batch)
    saliency_maps.init_gradient_saliency_maps()

    # Visualize (no batch_index means use input_batch)
    saliency_maps.visualize_saliency_map(
        plt,
        image_index=0,
        title="COVID Sample",
        id2label=id2label,
        alpha=0.6
    )

API Reference
-------------

.. autoclass:: pyhealth.interpret.methods.basic_gradient.BasicGradientSaliencyMaps
    :members:
    :undoc-members:
    :show-inheritance:
    :member-order: bysource

Algorithm Details
-----------------

The saliency computation follows these steps:

1. **Forward pass**: Compute model predictions for the input batch
2. **Target selection**: Use predicted class (argmax of probabilities)
3. **Backward pass**: Compute gradients with respect to input pixels
4. **Saliency map**: Take absolute value and max across color channels

Mathematical formula:

.. math::

    \text{saliency}(x, y) = \max_{c} \left| \frac{\partial \text{score}_{\text{predicted}}}{\partial \text{pixel}_{x,y,c}} \right|

where :math:`c` iterates over color channels (RGB or grayscale).
