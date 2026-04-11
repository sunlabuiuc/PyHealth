from pyhealth.datasets import create_sample_dataset, get_dataloader
from pyhealth.models.attention_lstm import AttentionLSTM

samples = [
    {
        "patient_id": "patient-0",
        "visit_id": "visit-0",
        "conditions": ["cond-33", "cond-86", "cond-80"],
        "procedures": ["proc-12", "proc-45"],
        "label": 1,
    },
    {
        "patient_id": "patient-1",
        "visit_id": "visit-1",
        "conditions": ["cond-12", "cond-52"],
        "procedures": ["proc-23"],
        "label": 0,
    },
]

dataset = create_sample_dataset(
    samples=samples,
    input_schema={"conditions": "sequence", "procedures": "sequence"},
    output_schema={"label": "binary"},
    dataset_name="test",
)

loader = get_dataloader(dataset, batch_size=2, shuffle=False)
model = AttentionLSTM(dataset=dataset, embedding_dim=128, hidden_dim=64)

batch = next(iter(loader))
ret = model(**batch)

loss = ret["loss"]
loss.backward()

print("Backward pass successful")