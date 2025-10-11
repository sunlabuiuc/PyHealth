# Chest X-Ray Analysis with PyHealth

## Introduction

This example demonstrates how to use PyHealth to perform chest X-ray classification, focusing on detecting abnormalities such as pneumonia and edema. It builds on the reproducibility efforts for the UniXGen model (Lee et al., 2023), a vision-language generative model for view-specific chest X-ray generation. The example leverages the CheXpert dataset and introduces two new PyHealth contributions:

- A task function (`chest_xray_classification_fn`) to label chest X-rays based on diagnoses.
- A metrics module (`radiographic_agreement`) to evaluate inter-rater agreement for radiographic findings.

## Setup

First, ensure PyHealth and its dependencies are installed. Then, import the required modules and set up logging.

```python
import pyhealth
from pyhealth.datasets import CheXpertDataset
from pyhealth.tasks import chest_xray_classification_fn  # Import from pyhealth.tasks
from pyhealth.metrics import radiographic_agreement  # Import from pyhealth.metrics
import matplotlib.pyplot as plt
import logging
import pandas as pd
import cv2

# Set up logging
logging.basicConfig(level=logging.INFO)

# Load CheXpert dataset
dataset = CheXpertDataset(
    root="/path/to/chexpert",  # Update with actual path
    dev=False,
    refresh_cache=True
)
```

### Notes
- Replace `/path/to/chexpert` with the actual path to your CheXpert dataset directory.
- Ensure the dataset is downloaded and formatted as expected by PyHealth (see [CheXpert documentation](https://stanfordmlgroup.github.io/competitions/chexpert)).

## Data Preprocessing

Use the `chest_xray_classification_fn` task to process the dataset and label X-ray images based on the presence of pneumonia or edema.

```python
samples = []
for patient in dataset.patients:
    patient_samples = chest_xray_classification_fn(patient)
    samples.extend(patient_samples)

# Convert to DataFrame for analysis
df = pd.DataFrame(samples)
print(df.head())
```

### Expected Output
The `df` DataFrame will contain columns like `patient_id`, `visit_id`, `xray_path`, `view_position`, and `label` (1 if pneumonia or edema is present, 0 otherwise).

## Visualization

Visualize a sample X-ray image along with its label and view position.

```python
sample = df.iloc[0]
img = cv2.imread(sample["xray_path"])
if img is not None:
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title(f"Label: {sample['label']}, View: {sample['view_position']}")
    plt.show()
else:
    print(f"Failed to load image at {sample['xray_path']}")
```

### Notes
- Ensure `cv2` (OpenCV) is installed to load and display images.
- If the image fails to load, verify the `xray_path` points to a valid file.

## Model Training (Simple Example)

Train a basic PyHealth model (e.g., `LogisticRegression`) to classify X-rays. Note that this is a placeholder; in practice, you’d likely use a deep learning model (e.g., a CNN) to extract features from X-ray images.

```python
from pyhealth.models import LogisticRegression

model = LogisticRegression(
    feature_keys=["xray_path"],
    label_key="label",
    feature_dims=512  # Placeholder for image feature dimension
)
model.fit(dataset, batch_size=32, epochs=5)
```

### To-Do
- Replace `LogisticRegression` with a more suitable model (e.g., a CNN like ResNet) and preprocess X-ray images into feature vectors.
- Adjust `feature_dims` based on your feature extraction method.

## Evaluation

Evaluate the model’s predictions using the `radiographic_agreement` metric to measure inter-rater agreement between true and predicted labels.

```python
# Placeholder for predictions (replace with actual model predictions)
y_true = [sample["label"] for sample in samples]
y_pred = model.predict(dataset)  # Adjust based on actual model

# Compute agreement
agreement_metrics = radiographic_agreement(y_true, y_pred)
print(f"Cohen's Kappa: {agreement_metrics['kappa']:.3f}")
print(f"Percent Agreement: {agreement_metrics['percent_agreement']:.2f}%")
```

### Expected Output
- `Cohen's Kappa`: A value between -1 and 1, where 1 indicates perfect agreement.
- `Percent Agreement`: Percentage of matching labels (0-100%).

## Conclusion

This example demonstrates how PyHealth can be used to classify chest X-ray abnormalities using the CheXpert dataset. The `chest_xray_classification_fn` task simplifies data preprocessing, while the `radiographic_agreement` metric provides a robust evaluation of model performance. Future work could integrate these outputs with UniXGen’s generated X-rays for enhanced analysis.

## References

- Lee, Hyungyung, et al. "Vision-Language Generative Model for View-Specific Chest X-Ray Generation." arXiv preprint arXiv:2302.12172, 2023.
- PyHealth Documentation: [https://pyhealth.readthedocs.io/](https://pyhealth.readthedocs.io/)