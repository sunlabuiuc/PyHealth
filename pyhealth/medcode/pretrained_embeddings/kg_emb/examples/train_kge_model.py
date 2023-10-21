from pyhealth.medcode.pretrained_embeddings.kg_emb.datasets import UMLSDataset, split
from pyhealth.medcode.pretrained_embeddings.kg_emb.tasks import link_prediction_fn
from pyhealth.datasets import get_dataloader
from pyhealth.medcode.pretrained_embeddings.kg_emb.models import TransE, RotatE, ComplEx, DistMult
from pyhealth.trainer import Trainer
from pyhealth.medcode import InnerMap
import torch
import json
import pickle

"""
This is an example to show you how to train a KG embedding model using our package

"""


umls_ds = UMLSDataset(
    root="/data/pj20/umls/",
    # root="https://storage.googleapis.com/pyhealth/umls/",
    dev=False,
    refresh_cache=True
)

# check the dataset statistics before setting task
print(umls_ds.stat()) 

# check the relation numbers in the dataset
print("Relations in KG:", umls_ds.relation2id)

umls_ds = umls_ds.set_task(link_prediction_fn, negative_sampling=64, save=False)

# save the id2entity, id2relation
with open("/data/pj20/umls_kge/pretrained_model/umls_transe_new/id2entity.json", "w") as f:
    json.dump(umls_ds.id2entity, f, indent=6)

with open("/data/pj20/umls_kge/pretrained_model/umls_transe_new/id2relation.json", "w") as f:
    json.dump(umls_ds.id2relation, f, indent=6)

# check the dataset statistics after setting task
print(umls_ds.stat())

# split the dataset and get the dataloaders
train_dataset, val_dataset, test_dataset = split(umls_ds, [0.9, 0.05, 0.05])
train_loader = get_dataloader(train_dataset, batch_size=8, shuffle=True)
# val_loader = get_dataloader(val_dataset, batch_size=2, shuffle=False)
# test_loader = get_dataloader(test_dataset, batch_size=2, shuffle=False)


# initialize a KGE model
model = TransE(
    dataset=umls_ds,
    e_dim=512, 
    r_dim=512, 
)

print('Loaded model: ', model)
state_dict = torch.load("/data/pj20/umls_kge/pretrained_model/umls_transe_new/1_250000_last.ckpt")
model.load_state_dict(state_dict)

# initialize a trainer and start training
trainer = Trainer(
    model=model, 
    device='cuda:5', 
    metrics=['hits@n', 'mean_rank'], 
    output_path='/data/pj20/umls_kge/pretrained_model',
    exp_name='umls_transe_new'
    )

trainer.train(
    train_dataloader=train_loader,
    # val_dataloader=val_loader,
    epochs=10,
    # steps_per_epoch=100,
    # evaluation_steps=1,
    optimizer_params={'lr': 1e-3},
    monitor='mean_rank',
    monitor_criterion='min'
)

# save the entity embedding and relation embedding
with open("/data/pj20/umls_kge/pretrained_model/umls_transe_new/entity_embedding.pkl", "wb") as f:
    pickle.dump(model.E_emb, f)

with open("/data/pj20/umls_kge/pretrained_model/umls_transe_new/relation_embedding.pkl", "wb") as f:
    pickle.dump(model.R_emb, f)

# evaluate the trained model
# trainer.evaluate(test_loader)

# use the trained model to handle head/tail entity prediction, and use the CodeMap to map the code to free text
umls_code_map = InnerMap.load("UMLS", refresh_cache=True)

head = 'C0000039'
relation = "PAR"

model.to('cpu')
result_eid = model.inference(
    head=umls_ds.entity2id[head], 
    relation=umls_ds.relation2id[relation],
    tail=None,
    top_k=3
    )

print(f"Input Head: {head} - {umls_code_map.lookup(head)}")
print(f"Input Relation: {relation}")

print("Tail Prediction:")
for idx, eid in enumerate(result_eid):
    tail = umls_ds.id2entity[eid]
    print(f"{idx}: {tail} - {umls_code_map.lookup(tail)}")


tail = 'C5162542'
relation = "CHD"

result_eid = model.inference(
    head=None, 
    relation=umls_ds.relation2id[relation],
    tail=umls_ds.entity2id[tail],
    top_k=3
    )

print(f"Input Tail: {tail} - {umls_code_map.lookup(tail)}")
print(f"Input Relation: {relation}")

print("Head Prediction:")
for idx, eid in enumerate(result_eid):
    head = umls_ds.id2entity[eid]
    print(f"{idx}: {head} - {umls_code_map.lookup(head)}")


head = 'C0000039'
tail = 'C0000039'
relation = "SY"

classification_score = model.inference(
    head=umls_ds.entity2id[head], 
    relation=umls_ds.relation2id[relation],
    tail=umls_ds.entity2id[tail],
    )

print(f"Classification Score: {classification_score}")
