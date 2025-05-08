# Accuracy Extrapolation Module for PyHealth

This module provides functionality to predict model performance when trained on larger datasets based on small pilot data. It implements an improved version of the APEx-GP approach with Matern kernels and Beta priors.

## Background

In healthcare machine learning applications, data collection is often expensive and time-consuming. The Accuracy Extrapolation module helps researchers and practitioners predict how a model would perform if more data were collected, enabling better resource allocation decisions.

Key improvements over standard methods:

1. **Matern Kernels**: More realistic modeling of learning curves by allowing for less smooth functions compared to RBF kernels
2. **Beta Priors**: More appropriate modeling of bounded accuracy metrics (like AUROC) which are constrained to [0,1]
3. **Architecture Generalization**: Ability to predict performance trends across different model architectures

## Installation

The module requires GPyTorch as a dependency:

```bash
pip install gpytorch matplotlib
```

## Quick Start

```python
import numpy as np
from pyhealth.metrics.extrapolation import extrapolate_accuracy

# Generate or load some learning curve data
# training_sizes = [100, 200, 500, 1000, 2000]
# accuracies = [0.65, 0.70, 0.75, 0.78, 0.81]

# For this example, we'll generate synthetic data
training_sizes = np.array([100, 200, 500, 1000, 2000])
true_function = lambda x: 0.9 - 0.5 * np.power(x, -0.3)  # Power law
accuracies = true_function(training_sizes) + np.random.normal(0, 0.01, len(training_sizes))

# Target sizes to extrapolate to
target_sizes = [5000, 10000, 20000]

# Use the convenience function to extrapolate
predictions = extrapolate_accuracy(
    train_sizes=training_sizes,
    accuracies=accuracies,
    target_sizes=target_sizes,
    model_type="matern",  # Use Matern kernel
    nu=2.5,               # Smoothness parameter
    mean_type="powerlaw", # Mean function type
    plot=True             # Generate a plot
)

print(f"Predicted accuracies for {target_sizes}: {predictions.round(3)}")
```

## Advanced Usage

For more control over the extrapolation process, use the `AccuracyExtrapolation` class:

```python
from pyhealth.metrics.extrapolation import AccuracyExtrapolation
import numpy as np
import matplotlib.pyplot as plt

# Generate or load data
training_sizes = np.array([100, 200, 500, 1000, 2000])
accuracies = np.array([0.65, 0.70, 0.75, 0.78, 0.81])

# Initialize extrapolator
extrapolator = AccuracyExtrapolation(
    model_type="matern",
    nu=2.5,
    mean_type="powerlaw",
    use_beta_prior=True  # Use Beta prior for bounded accuracy values
)

# Fit model
training_stats = extrapolator.fit(
    train_sizes=training_sizes,
    accuracies=accuracies,
    max_iter=1000,
    lr=0.01,
    verbose=True
)

# Predict for target sizes
target_sizes = [5000, 10000, 20000, 50000]
mean_preds, lower_bounds, upper_bounds = extrapolator.predict(target_sizes)

print("Predicted accuracies:")
for size, mean, lower, upper in zip(target_sizes, mean_preds, lower_bounds, upper_bounds):
    print(f"Size {size}: {mean:.3f} (95% CI: [{lower:.3f}, {upper:.3f}])")

# Generate a plot
extrapolator.plot(
    training_sizes,
    accuracies,
    extrapolate_to=100000,
    log_scale=True,
    save_path="accuracy_extrapolation.png"
)
```

## Available Models and Options

### Kernel Types

- `"rbf"`: Standard RBF kernel (more smooth)
- `"matern"`: Matern kernel with varying smoothness parameter (`nu`)

### Mean Functions

- `"constant"`: Simple constant mean
- `"powerlaw"`: Power law function (common for learning curves)
- `"arctan"`: Arctan function (asymptotes more gradually)

### Smoothness Parameter (nu)

For Matern kernels, you can set the smoothness parameter `nu` to:
- `0.5`: Less smooth (Matern 1/2)
- `1.5`: Moderately smooth (Matern 3/2)
- `2.5`: More smooth (Matern 5/2, recommended default)

### Prior Types

- Standard Gaussian likelihood
- Beta prior (when `use_beta_prior=True`): Better for bounded metrics

## Performance Comparison

In general, our experiments show that:

1. Matern kernels achieve lower MSE compared to standard RBF kernels, with improvements ranging from 0.3% to 13.1%.
2. Beta priors provide more conservative extrapolations that respect the natural bounds of accuracy metrics.
3. The method works well across different neural network architectures (CNN, Transformer, etc.)

## Real-World Scenarios

For real-world applications, we recommend:

1. Gathering accuracies/performances at multiple dataset sizes (at least 5 points)
2. Using Matern 5/2 kernel as the default
3. Using Beta priors if you want more conservative extrapolations
4. Setting reasonable training iterations (1000-2000) for convergence

## Contributing

Contributions to improve the module are welcome. Please feel free to submit pull requests or open issues for any bugs or feature requests. 