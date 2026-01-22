pyhealth.interpret.utils
========================

.. automodule:: pyhealth.interpret.utils
   :members:
   :undoc-members:
   :show-inheritance:

Overview
--------

The ``pyhealth.interpret.utils`` module provides visualization utilities for 
interpretability methods in PyHealth. These functions help create visual
explanations of model predictions, particularly useful for medical imaging.

Core Functions
--------------

**Overlay Visualization**

- :func:`show_cam_on_image` - Overlay a CAM/attribution map on an image
- :func:`visualize_attribution_on_image` - Generate complete attribution visualization

**Normalization & Processing**

- :func:`normalize_attribution` - Normalize attribution values for visualization
- :func:`interpolate_attribution_map` - Resize attribution to match image dimensions

**Figure Generation**

- :func:`create_attribution_figure` - Create publication-ready figure with overlays

ViT-Specific Functions
----------------------

These functions are specifically designed for Vision Transformer (ViT) models
using attention-based interpretability methods like :class:`~pyhealth.interpret.methods.CheferRelevance`.

- :func:`generate_vit_visualization` - Generate visualization components for ViT attribution
- :func:`create_vit_attribution_figure` - Create complete ViT attribution figure
- :func:`reshape_vit_attribution` - Reshape flat patch attribution to 2D spatial map

Example: Basic Attribution Visualization
----------------------------------------

.. code-block:: python

    import numpy as np
    from pyhealth.interpret.utils import show_cam_on_image, normalize_attribution

    # Assume we have image and attribution from an interpreter
    image = np.random.rand(224, 224, 3)  # RGB image in [0, 1]
    attribution = np.random.rand(224, 224)  # Raw attribution values

    # Normalize and overlay
    attr_normalized = normalize_attribution(attribution)
    overlay = show_cam_on_image(image, attr_normalized)

Example: ViT Attribution with CheferRelevance
---------------------------------------------

.. code-block:: python

    from pyhealth.models import TorchvisionModel
    from pyhealth.interpret.methods import CheferRelevance
    from pyhealth.interpret.utils import (
        generate_vit_visualization,
        create_vit_attribution_figure,
    )
    import matplotlib.pyplot as plt

    # Initialize ViT model and interpreter
    model = TorchvisionModel(dataset, "vit_b_16", {"weights": "DEFAULT"})
    # ... train model ...
    
    interpreter = CheferRelevance(model)
    
    # Generate visualization components
    image, attr_map, overlay = generate_vit_visualization(
        interpreter=interpreter,
        **test_batch
    )
    
    # Or create a complete figure
    fig = create_vit_attribution_figure(
        interpreter=interpreter,
        class_names={0: "Normal", 1: "COVID", 2: "Pneumonia"},
        save_path="vit_attribution.png",
        **test_batch
    )

See Also
--------

- :mod:`pyhealth.interpret.methods` - Attribution methods (DeepLift, IntegratedGradients, CheferRelevance, etc.)
- :class:`pyhealth.interpret.methods.CheferRelevance` - Attention-based interpretability for Transformers
- :class:`pyhealth.models.TorchvisionModel` - ViT and other vision models



