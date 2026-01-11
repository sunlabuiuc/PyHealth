Interpretability
===============

We implement the following interpretability techniques to help you understand model predictions and identify important features in healthcare data.


Getting Started
---------------

New to interpretability in PyHealth? Check out these complete examples:

**Browse all examples online**: https://github.com/sunlabuiuc/PyHealth/tree/master/examples

**Basic Gradient Example:**

- ``examples/ChestXrayClassificationWithSaliency.ipynb`` - Interactive notebook demonstrating gradient-based saliency mapping for medical image classification. Shows how to:

  - Load and classify chest X-ray images using PyHealth's TorchvisionModel
  - Generate gradient saliency maps to visualize model attention
  - Interpret which regions of X-ray images influence COVID-19 predictions by the model

**DeepLift Example:**

- ``examples/deeplift_stagenet_mimic4.py`` - Demonstrates DeepLift attributions on StageNet for mortality prediction with MIMIC-IV data. Shows how to:

  - Compute feature attributions for discrete (ICD codes) and continuous (lab values) features
  - Decode attributions back to human-readable medical codes and descriptions
  - Visualize top positive and negative attributions

**Integrated Gradients Examples:**

- ``examples/integrated_gradients_mortality_mimic4_stagenet.py`` - Complete workflow showing:

  - How to load pre-trained models and compute attributions
  - Comparing attributions for different target classes (mortality vs. survival)
  - Interpreting results with medical context (lab categories, diagnosis codes)

- ``examples/interpretability_metrics.py`` - Demonstrates evaluation of attribution methods using:

  - **Comprehensiveness**: Measures how much prediction drops when removing important features
  - **Sufficiency**: Measures how much prediction is retained when keeping only important features
  - Both functional API (``evaluate_attribution``) and class-based API (``Evaluator``)

**SHAP Example:**

- ``examples/shap_stagenet_mimic4.py`` - Demonstrates SHAP (SHapley Additive exPlanations) for StageNet mortality prediction. Shows how to:

  - Compute Kernel SHAP attributions for healthcare models with discrete and continuous features
  - Interpret Shapley values to understand feature contributions based on game theory
  - Compare different baseline strategies for background sample generation
  - Decode attributions to human-readable medical codes and lab measurements

**LIME Example:**

- ``examples/lime_stagenet_mimic4.py`` - Demonstrates LIME (Local Interpretable Model-agnostic Explanations) for StageNet mortality prediction. Shows how to:

  - Compute local linear approximations to explain model predictions
  - Generate perturbations around input samples to train interpretable models
  - Compare different regularization methods (Lasso vs Ridge) for feature selection
  - Test various distance kernels (cosine vs euclidean) and sample sizes
  - Decode attributions to human-readable medical codes and lab measurements

These examples provide end-to-end workflows from loading data to interpreting and evaluating attributions.

Available Methods
-----------------
    
.. toctree::
    :maxdepth: 4

    interpret/pyhealth.interpret.methods.gim
    interpret/pyhealth.interpret.methods.basic_gradient
    interpret/pyhealth.interpret.methods.chefer
    interpret/pyhealth.interpret.methods.deeplift
    interpret/pyhealth.interpret.methods.integrated_gradients
    interpret/pyhealth.interpret.methods.shap
    interpret/pyhealth.interpret.methods.lime
 