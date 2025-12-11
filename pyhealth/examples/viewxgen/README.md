# ViewXGen Reproduction Example

This example demonstrates a lightweight reproduction of the ViewXGen
multimodal chest X ray generation workflow using PyHealth.

Because the full ViewXGen model is too large for a course environment,
this example uses:
- a tiny dummy dataset loaded with `SampleDataset`
- a reduced transformer with 2 layers
- short token sequences to simulate VQ GAN image tokens
- a simple training loop
- a minimal evaluation script

This example is fully self-contained and does not require access to the
restricted MIMIC CXR dataset.

## File Structure

viewxgen_reproduction/
    dataset_demo.py
    model_demo.py
    train_demo.py
    evaluate_demo.py

## Running the Example

1. Install Dependencies:
pip install torch torchvision einops transformers pillow scikit-learn

2. Run the dataset demo:
python dataset_demo.py

3. Train the model:
python train_demo.py

4. Evaluate:
python evaluate_demo.py
