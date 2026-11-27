Interpretability
================

We implement the following interpretability techniques to help you understand model predictions and identify important features in healthcare data.


Getting Started
---------------

New to interpretability in PyHealth? Check out these complete examples:

**Browse all examples online**: https://github.com/sunlabuiuc/PyHealth/tree/master/examples

**Basic Gradient Example:**

- ``examples/cxr/ChestXrayClassificationWithSaliency.ipynb`` - Interactive notebook demonstrating gradient-based saliency mapping for medical image classification. Shows how to:

  - Load and classify chest X-ray images using PyHealth's TorchvisionModel
  - Generate gradient saliency maps to visualize model attention
  - Interpret which regions of X-ray images influence COVID-19 predictions by the model

**DeepLift Example:**

- ``examples/interpretability/deeplift_stagenet_mimic4.py`` - Demonstrates DeepLift attributions on StageNet for mortality prediction with MIMIC-IV data. Shows how to:

  - Compute feature attributions for discrete (ICD codes) and continuous (lab values) features
  - Decode attributions back to human-readable medical codes and descriptions
  - Visualize top positive and negative attributions

**Integrated Gradients Examples:**

- ``examples/interpretability/integrated_gradients_mortality_mimic4_stagenet.py`` - Complete workflow showing:

  - How to load pre-trained models and compute attributions
  - Comparing attributions for different target classes (mortality vs. survival)
  - Interpreting results with medical context (lab categories, diagnosis codes)

- ``examples/interpretability/interpretability_metrics.py`` - Demonstrates evaluation of attribution methods using:

  - **Comprehensiveness**: Measures how much prediction drops when removing important features
  - **Sufficiency**: Measures how much prediction is retained when keeping only important features
  - Both functional API (``evaluate_attribution``) and class-based API (``Evaluator``)

**SHAP Example:**

- ``examples/interpretability/shap_stagenet_mimic4.py`` - Demonstrates SHAP (SHapley Additive exPlanations) for StageNet mortality prediction. Shows how to:

  - Compute Kernel SHAP attributions for healthcare models with discrete and continuous features
  - Interpret Shapley values to understand feature contributions based on game theory
  - Compare different baseline strategies for background sample generation
  - Decode attributions to human-readable medical codes and lab measurements

**Chefer Attribution Example:**

- ``examples/interpretability/mp_transformer_mimic4_interpret.py`` - Runs CheferRelevance (gradient-weighted attention relevance) on a PyHealth ``Transformer`` for MIMIC-IV mortality prediction with ``--methods chefer``, and scores it with comprehensiveness and sufficiency next to the other attribution methods.

- ``examples/cxr/covid19cxr_tutorial.py`` - Trains a ViT (``TorchvisionModel``) on COVID-19 chest X-rays with conformal prediction. Its Chefer step currently raises, because ``TorchvisionModel`` does not implement the ``CheferInterpretable`` interface; the file's header notes this.

**LIME Example:**

- ``examples/lime_stagenet_mimic4.py`` - Demonstrates LIME (Local Interpretable Model-agnostic Explanations) for StageNet mortality prediction. Shows how to:

  - Compute local linear approximations to explain model predictions
  - Generate perturbations around input samples to train interpretable models
  - Compare different regularization methods (Lasso vs Ridge) for feature selection
  - Test various distance kernels (cosine vs euclidean) and sample sizes
  - Decode attributions to human-readable medical codes and lab measurements

**Uni-ACS Clinical Scoring Example:**

- ``examples/mimic4_mortality_uni_acs.py`` - Demonstrates Unified Auto Clinical Scoring (Uni-ACS) with a runnable ablation over feature count (``top_k``), discretization method (``quantile`` vs ``novel``), and score-level isotonic calibration on synthetic data (and optional MIMIC-IV CSV input).

These examples provide end-to-end workflows from loading data to interpreting and evaluating attributions.

Attribution Methods
-------------------
    
.. toctree::
    :maxdepth: 4

    interpret/pyhealth.interpret.methods.gim
    interpret/pyhealth.interpret.methods.basic_gradient
    interpret/pyhealth.interpret.methods.chefer
    interpret/pyhealth.interpret.methods.attention_rollout
    interpret/pyhealth.interpret.methods.raw_attention
    interpret/pyhealth.interpret.methods.grad_cam
    interpret/pyhealth.interpret.methods.deeplift
    interpret/pyhealth.interpret.methods.integrated_gradients
    interpret/pyhealth.interpret.methods.shap
    interpret/pyhealth.interpret.methods.lime
    interpret/pyhealth.interpret.methods.tree_shap
    interpret/pyhealth.interpret.methods.ig_gim
    interpret/pyhealth.interpret.methods.ensemble
    interpret/pyhealth.interpret.methods.baseline
    interpret/pyhealth.interpret.methods.base_interpreter

Clinical Scoring
----------------

The ``pyhealth.interpret.uni_acs`` module converts SHAP feature attributions
into transparent, additive integer clinical scoring tables.

.. toctree::
    :maxdepth: 4

    interpret/pyhealth.interpret.uni_acs

Visualization Utilities
-----------------------

The ``pyhealth.interpret.utils`` module provides visualization functions for 
creating attribution overlays, heatmaps, and publication-ready figures.
Includes specialized support for Vision Transformer (ViT) attribution visualization.

.. toctree::
    :maxdepth: 4

    interpret/pyhealth.interpret.utils
