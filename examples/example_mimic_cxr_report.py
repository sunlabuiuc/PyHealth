from pyhealth.datasets.mimic_cxr import MIMICCXRReportDataset
from pyhealth.datasets import split_by_patient
from pyhealth.tasks import add_prediction_task
from pyhealth.models import Bert
from pyhealth.trainer import Trainer

# Load dataset CSV (user-provided)
dataset = MIMICCXRReportDataset(
    root=".",
    csv_path="mimic_cxr_reports_with_doc_labels.csv",
    id_col="report_id",
    text_col="report_text",
    label_col="doc_label",
)

# Add prediction task
task_dataset = add_prediction_task(
    dataset=dataset,
    task_name="cxr_abnormal_prediction",
    label_key="label",
    text_key="report_text",
)

# Split dataset
train_ds, val_ds, test_ds = split_by_patient(task_dataset, [0.7, 0.1, 0.2])

# Build model (standard PyHealth BERT)
model = Bert(
    task=task_dataset.get_task("cxr_abnormal_prediction"),
    pretrained_model_name="bert-base-uncased",
)

# Train
trainer = Trainer(
    model=model,
    train_dataset=train_ds,
    val_dataset=val_ds,
    test_dataset=test_ds,
    batch_size=4,
    lr=2e-5,
    epochs=1,
)

trainer.train()

# Evaluate
results = trainer.evaluate(split="test")
print("Test results:", results)
