import torch
from pyhealth.datasets import create_sample_dataset, get_dataloader
from pyhealth.models.ehr_mamba import EhrMamba
from pyhealth.models.gru_baseline import GRUBaseline
 
 # python examples/mimic3_readmission_ehrmamba.py
def make_dataset(feature_keys=("conditions",)):
    samples = []
    for i in range(20):
        sample = {
            "patient_id": f"p-{i}",
            "visit_id": f"v-{i}",
            "label": i % 2,
        }
        if "conditions" in feature_keys:
            sample["conditions"] = [f"cond-{(i * 3 + j) % 20}" for j in range(3)]
        if "procedures" in feature_keys:
            sample["procedures"] = [f"proc-{(i * 2 + j) % 15}" for j in range(2)]
        samples.append(sample)
 
    input_schema = {k: "sequence" for k in feature_keys}
    return create_sample_dataset(
        samples=samples,
        input_schema=input_schema,
        output_schema={"label": "binary"},
        dataset_name="synthetic_ehr",
    )
 
 
def train_model(model, dataset, epochs=5, lr=0.001):
    loader = get_dataloader(dataset, batch_size=8, shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
 
    final_loss = None
    for epoch in range(epochs):
        for batch in loader:
            output = model(**batch)
            loss = output["loss"]
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            final_loss = loss.item()
 
    return final_loss
 
 
def ablation_model_type():
    print("ABLATION 1: EhrMamba vs GRU Baseline")
    dataset = make_dataset(["conditions"])
 
    for model_class, name in [
        (EhrMamba, "EhrMamba (Conv1D)"),
        (GRUBaseline, "GRU Baseline"),
    ]:
        model = model_class(dataset=dataset, hidden_dim=128)
        loss = train_model(model, dataset)
        print(f"  {name:30s} | Final Loss: {loss:.4f}")
 
 
def ablation_hidden_dim():
    print("ABLATION 2: Hidden Dimension Scaling (EhrMamba)")
    dataset = make_dataset(["conditions"])
 
    for hidden_dim in [64, 128, 256]:
        model = EhrMamba(dataset=dataset, hidden_dim=hidden_dim)
        n_params = sum(p.numel() for p in model.parameters())
        loss = train_model(model, dataset)
        print(f"  hidden_dim={hidden_dim:3d} | Params: {n_params:,} | Final Loss: {loss:.4f}")
 
 
def ablation_dropout():
    print("ABLATION 3: Dropout Rate (EhrMamba)")
    dataset = make_dataset(["conditions"])
 
    for dropout in [0.1, 0.3, 0.5]:
        model = EhrMamba(dataset=dataset, dropout=dropout)
        loss = train_model(model, dataset)
        print(f"  dropout={dropout:.1f} | Final Loss: {loss:.4f}")
 
 
def ablation_feature_sets():
    print("ABLATION 4: Feature Set Variations (EhrMamba)")

    configs = [
        (["conditions"], "Conditions only"),
        (["procedures"], "Procedures only"),
        (["conditions", "procedures"], "Conditions + Procedures"),
    ]
 
    for feature_keys, name in configs:
        dataset = make_dataset(feature_keys)
        model = EhrMamba(dataset=dataset)
        loss = train_model(model, dataset)
        print(f"  {name:35s} | Final Loss: {loss:.4f}")
 
 
if __name__ == "__main__":
    print("EhrMamba Ablation Study")
    print("Paper: Fallahpour et al., ML4H 2024")
    print("https://proceedings.mlr.press/v259/fallahpour24a.html")
 
    ablation_model_type()
    ablation_hidden_dim()
    ablation_dropout()
    ablation_feature_sets()
 
    print("\nDone.")
