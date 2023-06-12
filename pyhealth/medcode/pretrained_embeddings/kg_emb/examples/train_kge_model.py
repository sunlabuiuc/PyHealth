from pyhealth.medcode.pretrained_embeddings.kg_emb.datasets import UMLSDataset, split
from pyhealth.medcode.pretrained_embeddings.kg_emb.tasks import link_prediction_fn
from pyhealth.datasets import get_dataloader
from pyhealth.medcode.pretrained_embeddings.kg_emb.models import TransE, RotatE, ComplEx, DistMult
from pyhealth.trainer import Trainer
from pyhealth.medcode import InnerMap


"""
This is an example to show you how to train a KG embedding model using our package

"""


umls_ds = UMLSDataset(
    root="https://storage.googleapis.com/pyhealth/umls/",
    dev=False,
    refresh_cache=False
)

# check the dataset statistics before setting task
print(umls_ds.stat()) 

# check the relation numbers in the dataset
print("Relations in KG:", umls_ds.relation2id)

umls_ds = umls_ds.set_task(link_prediction_fn, negative_sampling=512, save=False)

# check the dataset statistics after setting task
print(umls_ds.stat())

# split the dataset and get the dataloaders
train_dataset, val_dataset, test_dataset = split(umls_ds, [0.99, 0.005, 0.005])
train_loader = get_dataloader(train_dataset, batch_size=256, shuffle=True)
val_loader = get_dataloader(val_dataset, batch_size=4, shuffle=False)
test_loader = get_dataloader(test_dataset, batch_size=4, shuffle=False)


# initialize a KGE model
model = RotatE(
    dataset=umls_ds,
    e_dim=512, 
    r_dim=256, 
)

# initialize a trainer and start training
trainer = Trainer(
    model=model, 
    device='cuda:1', 
    metrics=['hits@n', 'mean_rank'], 
    output_path='./pretrained_model',
    exp_name='umls_rotate_new'
    )

trainer.train(
    train_dataloader=train_loader,
    val_dataloader=val_loader,
    epochs=100,
    # steps_per_epoch=100,
    evaluation_steps=10,
    optimizer_params={'lr': 1e-3},
    monitor='mean_rank',
    monitor_criterion='min'
)

# evaluate the trained model
trainer.evaluate(test_loader)

# use the trained model to handle head/tail entity prediction, and use the CodeMap to map the code to free text
umls_code_map = InnerMap.load("UMLS")

head = 'C0070122'
relation = "may_be_treated_by"

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


tail = 'C0677949'
relation = "disease_mapped_to_gene"

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


head = 'C0677949'
tail = 'C0162832'
relation = "disease_mapped_to_gene"

classification_score = model.inference(
    head=umls_ds.entity2id[head], 
    relation=umls_ds.relation2id[relation],
    tail=umls_ds.entity2id[tail],
    )

print(f"Classification Score: {classification_score}")
