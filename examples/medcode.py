from pyhealth.medcode import CrossMap, InnerMap

ndc = InnerMap.load("NDC")
print("Looking up for NDC code 00597005801")
print(ndc.lookup("00597005801"))

codemap = CrossMap.load("NDC", "ATC")
print("Mapping NDC code 00597005801 to ATC")
print(codemap.map("00597005801"))

atc = InnerMap.load("ATC")
print("Looking up for ATC code G04CA02")
print(atc.lookup("G04CA02"))


# Knowledge Graph Embedding (KGE) training over medical code knowledge graphs.
# See pyhealth.medcode.pretrained_embeddings.kg_emb for TransE, RotatE,
# DistMult, and ComplEx.
from pyhealth.medcode.pretrained_embeddings.kg_emb.datasets import SampleKGDataset
from pyhealth.medcode.pretrained_embeddings.kg_emb.models import TransE

entity2id = {"aspirin": 0, "headache": 1, "ibuprofen": 2}
relation2id = {"treats": 0}
samples = [
    {
        "triple": (0, 0, 1),
        "ground_truth_head": [0, 2],
        "ground_truth_tail": [1],
        "subsampling_weight": 1.0,
    },
]
kg_dataset = SampleKGDataset(
    samples=samples,
    dataset_name="toy_kg",
    entity_num=len(entity2id),
    relation_num=len(relation2id),
    entity2id=entity2id,
    relation2id=relation2id,
)
print("KG dataset stats:")
kg_dataset.stat()

kge_model = TransE(kg_dataset, e_dim=32, r_dim=32)
print("Entity embedding shape:", kge_model.E_emb.shape)
