import sys

sys.path.append("/home/chaoqiy2/github/PyHealth-OMOP")
from warnings import simplefilter

# ignore all warnings
simplefilter(action="ignore")


# Step 1: Load the data
from pyhealth.datasets import MIMIC3BaseDataset

base_ds = MIMIC3BaseDataset(
    root="/srv/local/data/physionet.org/files/mimiciii/1.4", flag="prod"
)

# Step 2: task specific dataset
from pyhealth.tasks import DrugRecDataset

drugrec_ds = DrugRecDataset(base_ds)

from pyhealth.data.split import split_by_pat

train_loader, val_loader, test_loader = split_by_pat(
    drugrec_ds,
    ratios=[2 / 3, 1 / 6, 1 / 6],
    batch_size=64,
    seed=12345,
)

# step 3: load the model
from pyhealth.models import RETAIN, MLModel, RNN, Transformer, GAMENet, SafeDrug, MICRON
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier


# ------ DL model -------
# for GAMENet
visit_ls = train_loader.dataset.indices
ehr_adj = train_loader.dataset.dataset.generate_ehr_adj_for_GAMENet(visit_ls)
ddi_adj = train_loader.dataset.dataset.get_ddi_matrix()
gamenet = GAMENet(
    task="drug_recommendation",
    voc_size=drugrec_ds.voc_size,
    tokenizers=drugrec_ds.tokenizers,
    emb_dim=64,
    ehr_adj=ehr_adj,
    ddi_adj=ddi_adj,
)

# for SafeDrug
bipartite_info = train_loader.dataset.dataset.generate_ddi_mask_H_for_SafeDrug()
MPNN_info = train_loader.dataset.dataset.generate_med_molecule_info_for_SafeDrug()
# ddi_adj = train_loader.dataset.dataset.get_ddi_matrix()
safedrug_008 = SafeDrug(
    task="drug_recommendation",
    voc_size=drugrec_ds.voc_size,
    tokenizers=drugrec_ds.tokenizers,
    emb_dim=64,
    MPNN_info=MPNN_info,
    bipartite_info=bipartite_info,
    ddi_adj=ddi_adj,
    target_ddi=0.08,
)

safedrug_006 = SafeDrug(
    task="drug_recommendation",
    voc_size=drugrec_ds.voc_size,
    tokenizers=drugrec_ds.tokenizers,
    emb_dim=64,
    MPNN_info=MPNN_info,
    bipartite_info=bipartite_info,
    ddi_adj=ddi_adj,
    target_ddi=0.06,
)

safedrug_004 = SafeDrug(
    task="drug_recommendation",
    voc_size=drugrec_ds.voc_size,
    tokenizers=drugrec_ds.tokenizers,
    emb_dim=64,
    MPNN_info=MPNN_info,
    bipartite_info=bipartite_info,
    ddi_adj=ddi_adj,
    target_ddi=0.04,
)

safedrug_002 = SafeDrug(
    task="drug_recommendation",
    voc_size=drugrec_ds.voc_size,
    tokenizers=drugrec_ds.tokenizers,
    emb_dim=64,
    MPNN_info=MPNN_info,
    bipartite_info=bipartite_info,
    ddi_adj=ddi_adj,
    target_ddi=0.02,
)

rnn = RNN(
    task="drug_recommendation",
    voc_size=drugrec_ds.voc_size,
    tokenizers=drugrec_ds.tokenizers,
    emb_dim=64,
)

transformer = Transformer(
    task="drug_recommendation",
    voc_size=drugrec_ds.voc_size,
    tokenizers=drugrec_ds.tokenizers,
    emb_dim=64,
)

retain = RETAIN(
    task="drug_recommendation",
    voc_size=drugrec_ds.voc_size,
    tokenizers=drugrec_ds.tokenizers,
    emb_dim=64,
)

micron = MICRON(
    task="drug_recommendation",
    voc_size=drugrec_ds.voc_size,
    tokenizers=drugrec_ds.tokenizers,
    emb_dim=64,
)

# ------ ML model -----------
LR = MLModel(
    output_path="../output",
    task="drug_recommendation",
    classifier=LogisticRegression(random_state=0, max_iter=10),
    voc_size=drugrec_ds.voc_size,
    tokenizers=drugrec_ds.tokenizers,
)

RF = MLModel(
    output_path="../output",
    task="drug_recommendation",
    classifier=RandomForestClassifier(random_state=0, n_estimators=20, max_depth=3),
    voc_size=drugrec_ds.voc_size,
    tokenizers=drugrec_ds.tokenizers,
)

mlp = MLModel(
    output_path="../output",
    task="drug_recommendation",
    classifier=MLPClassifier(max_iter=50, alpha=1e-3, hidden_layer_sizes=(16, 16)),
    voc_size=drugrec_ds.voc_size,
    tokenizers=drugrec_ds.tokenizers,
)


# step 4: train and eval the model
from pyhealth.trainer import Trainer
from pyhealth.evaluator.evaluating_multilabel import evaluate_multilabel

# --------- ML model -------------
for model in [LR, RF, mlp]:
    print("------------ Training model: ", model, "-------------")
    model.fit(
        train_loader=train_loader,
        evaluate_fn=evaluate_multilabel,
        eval_loader=val_loader,
        monitor="jaccard",
    )
    result = evaluate_multilabel(model, val_loader)
    print(
        "{:.4f} | {:.4f} | {:.4f} | {:.4f}".format(
            result["ddi"], result["jaccard"], result["prauc"], result["f1"]
        )
    )
    print("------------------")

# # ----------- DL model ---------
for model in [
    rnn,
    transformer,
    retain,
    gamenet,
    micron,
    safedrug_008,
    safedrug_006,
    safedrug_004,
    safedrug_002,
]:
    print("------------ Training model: ", model, "-------------")
    trainer = Trainer(enable_logging=True, output_path="../output")
    trainer.fit(
        model,
        train_loader=train_loader,
        epochs=50,
        evaluate_fn=evaluate_multilabel,
        eval_loader=val_loader,
        monitor="jaccard",
    )
    result = evaluate_multilabel(model, val_loader)
    print(
        "{:.4f} | {:.4f} | {:.4f} | {:.4f}".format(
            result["ddi"], result["jaccard"], result["prauc"], result["f1"]
        )
    )
    print("------------------")
