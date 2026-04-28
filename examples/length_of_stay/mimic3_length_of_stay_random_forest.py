"""
This study uses a RandomForest model on the MIMIC-III dataset to predict whether length
of stay exceeds 3 days, with hyperparameter tuning to maximize AUROC.

Setup:
- Dataset: MIMIC-III
- Task: Binary classification (Length of Stay > 3 days)
- Patient Data Split: 70% train / 10% validation / 20% test
- Evaluation metric: AUROC

Hyperparameter Tuning:
Grid search over the following hyperparameters:
- n_estimators: [100, 200, 300]
- max_depth: [5, 7, 10]
- min_samples_leaf: [1, 2]
- min_samples_split: [2, 3]
- class_weight: ["balanced", None]
- bootstrap: [True, False]

Each hyperparameter configuration was trained on the training set and evaluated on the
validation set. Results were ranked by AUROC.

Hyperparameter Tuning Findings
- The found best-performing random forest classifier model configuration achieved an
AUROC of ~0.77 using: bootstrap = True, class_weight = None, max_depth = 5,
min_samples_leaf = 1, min_samples_split = 2, and n_estimators = 200. Average AUROC
over all tuned parameters was ~0.70, Min: 0.52
- Shallow trees improve performance. Likely due to limited number of patients in dataset
- Increasing n_estimators (trees) improved performance
- Using class_weight="balanced" reduced AUROC

Final Model
The best hyperparameter configuration was used to train a final model, which was then
evaluated on the test set.

This experiment also serves as am example for how to use the PyHealth
RandomForest model and the Length of Stay Threshold binary prediction task
demonstrating:
1. Loading MIMIC-III data
2. Setting the Length of Stay Greater Than X Days Prediction task
3. Splitting the dataset and getting the dataloaders
4. Tuning hyperparameters
5. Creating and Fitting a RandomForest model
6. Evaluating a Random Forest model
"""
import tempfile

from pyhealth.datasets import MIMIC3Dataset, get_dataloader, split_by_patient
from pyhealth.models import RandomForest
from pyhealth.tasks.length_of_stay_prediction import \
    LengthOfStayThresholdPredictionMIMIC3


def print_results_table(performance_results):
    """ Helper to print the results of performance comparison across different
    configurations.
    """
    if not performance_results:
        print("No results to display.")
        return

    print("\n" + "*" * 25)
    print("Hyperparameter Tuning Results...")
    print("*" * 25 + "\n")

    # Alphabetize
    param_keys = sorted(performance_results[0]["params"].keys())

    # Create a table header of the hyperparams and a column for the resulting auroc
    # score
    header = param_keys + ["auroc score"]
    print(" | ".join(f"{h:^15}" for h in header))
    print("-" * (18 * len(header)))

    # Print each row
    for p in performance_results:
        row = [p["params"].get(k) for k in param_keys] + [p["score"]]
        print(" | ".join(f"{str(v):^15}" for v in row))


if __name__ == "__main__":

    # Constants
    BATCH_SIZE = 32

    # STEP 1: Load Dataset
    print("\n" + "*" * 25)
    print("Loading MIMIC3 Dataset...")
    print("*" * 25 + "\n")
    base_dataset = MIMIC3Dataset(
        root = "https://storage.googleapis.com/pyhealth/Synthetic_MIMIC-III",
        tables = ["DIAGNOSES_ICD", "PROCEDURES_ICD", "PRESCRIPTIONS"],
        cache_dir = tempfile.TemporaryDirectory().name,
        dev = True,
    )
    base_dataset.stats()

    # STEP 2: Set Task
    # Define the Length of Stay > 3 prediction classification task
    print("\n" + "*" * 25)
    print("Setting a LOS Threshold Binary Prediction Task...")
    print("*" * 25 + "\n")
    task = LengthOfStayThresholdPredictionMIMIC3(3, exclude_minors = False)
    sample_dataset = base_dataset.set_task(task)

    # STEP 3: Split Datasets
    print("\n" + "*" * 25)
    print("Creating Dataset Splits...")
    print("*" * 25 + "\n")
    train_dataset, val_dataset, test_dataset = split_by_patient(
        sample_dataset, [0.7, 0.1, 0.2]
    )

    train_loader = get_dataloader(train_dataset, batch_size = BATCH_SIZE,
                                  shuffle = True)
    val_loader = get_dataloader(val_dataset, batch_size = BATCH_SIZE, shuffle = False)
    test_loader = get_dataloader(test_dataset, batch_size = BATCH_SIZE, shuffle = False)

    # STEP 4: Define hyperparameters to tune and conduct hyperparameter tuning loop
    # Here we define a hyperparameter dictionary of values to try out in order to
    # determine what model configuration yields the best metrics over the validation
    # dataset
    tuning_params = {
        "n_estimators": [100, 200, 300],
        "max_depth": [5, 7, 10],
        "min_samples_leaf": [1, 2],
        "min_samples_split": [2, 3],
        "class_weight": ["balanced", None],
        "bootstrap": [True, False],
    }

    print("\n" + "*" * 20)
    print("Tuning Parameters...")
    print("*" * 20 + "\n")

    best_params, best_score, results = RandomForest.tune(
        sample_dataset,
        train_loader,
        val_loader,
        tuning_params,
        return_all = True
    )

    # Sort results so best auroc is on top
    results = sorted(
        results,
        key = lambda x: x["score"] if x["score"] is not None else -float("inf"),
        reverse = True,
    )
    print_results_table(results)

    print("\nBest Hyperparameter Combination:", best_params, best_score)

    # STEP 5: Create Final Random Forest Model
    final_model = RandomForest(
        dataset = sample_dataset,
        **best_params if best_score is not None else {},
    )
    final_model.fit(train_loader)

    # STEP 6: Final Evaluation on Test Dataset
    print("*" * 41)
    print("Final Evaluation Using Best Parameters...")
    print("*" * 41 + "\n")

    test_metrics = final_model.evaluate(test_loader)

    print("\n" + "*" * 58)
    print("Final Metrics Using Best Parameters on the Test Dataset...")
    print("*" * 58 + "\n")

    for k, v in test_metrics.items():
        print(f"{k}: {v:.4f}" if v is not None else f"{k}: None")
