from datasets import load_dataset

# Load the dataset
dataset = load_dataset("PULSE-ECG/ECGBench")

# Check the available splits
print(dataset)

# Example: explore the first sample of the train set
print(dataset['train'][0])
