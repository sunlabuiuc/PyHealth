# SOM-Only Model Documentation

## Overview

The **SOM-Only Model** is a clustering model based on the **Self-Organizing Map (SOM)** algorithm, which performs unsupervised learning to map high-dimensional input data onto a lower-dimensional grid of neurons (clusters). In the context of healthcare, the SOM-only model is used for discovering patterns or groups in patient data (or any time-series health data), which can aid in analyzing health states or predicting disease risk.

This model is implemented as part of the PyHealth repository, and it has been integrated to follow PyHealth's model interface, allowing it to be used with existing datasets and tasks.

## Model Description

The SOM-only model consists of the following key components:

- **SOM Layer**: This is the central part of the model. It consists of a set of neurons (clusters), where each neuron has a weight vector that represents the centroid of a group. The model computes the Euclidean distance between the input vector and the weight vectors to assign the data point to the closest cluster.
  
- **Loss Function**: The loss function computes the average Euclidean distance between the input and the closest assigned cluster, which the model minimizes during training.

- **Training Loop**: The model is trained using an optimization algorithm (Adam in this case), and the goal is to minimize the average distance between the input data points and the corresponding cluster centroids.

## File Structure

### **File Path**: `pyhealth/models/som_only.py`

This file contains the implementation of the SOM-only model, which is a subclass of `BaseModel` from the `pyhealth.models` module. It includes:

- The `SOM` class, which initializes the SOM and computes the distance between input vectors and the weight vectors of the clusters.
- The `SOMOnlyModel` class, which defines the forward pass and loss computation. It returns the cluster assignments (the closest cluster for each input data point).

### **File Path**: `train_som_only.py`

This file is used to generate synthetic data, create a model instance, and train the SOM-only model. The synthetic data is generated using `make_classification` from the `sklearn.datasets` module, which creates a simple classification problem for demonstration purposes.

---

## Implementation Details

### **SOM Class**

```python
class SOM(nn.Module):
    def __init__(self, input_dim, n_clusters):
        super(SOM, self).__init__()
        self.weights = nn.Parameter(torch.randn(n_clusters, input_dim))

    def forward(self, x):
        distances = torch.cdist(x, self.weights)
        return distances

