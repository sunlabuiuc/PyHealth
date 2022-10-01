import torch
import numpy as np


def extend(ls_of_ls):
    tmp = []
    for ls in ls_of_ls:
        tmp.extend(ls)
    return tmp


# data loader
def data_loader(dataset, batch_size, shuffle=True):
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        collate_fn=lambda x: {k: [d[k] for d in x] for k in x[0]},
    )


# split by patient
def split_by_pat(dataset, ratios, batch_size=64):
    """
    ratio for train / val / test
    """
    index_group = dataset.index_group
    N_pat = len(index_group)

    # select for train / val / test
    np.random.shuffle(index_group)
    train_index_group = index_group[: int(N_pat * ratios[0])]
    val_index_group = index_group[
        int(N_pat * ratios[0]) : int(N_pat * (ratios[0] + ratios[1]))
    ]
    test_index_group = index_group[int(N_pat * (ratios[0] + ratios[1])) :]

    train_dataset = torch.utils.data.Subset(dataset, extend(train_index_group))
    val_dataset = torch.utils.data.Subset(dataset, extend(val_index_group))
    test_dataset = torch.utils.data.Subset(dataset, extend(test_index_group))
    print("1. finish data splitting")
    print("2. generate train / val / test data loaders")
    return (
        data_loader(train_dataset, batch_size, True),
        data_loader(val_dataset, batch_size, False),
        data_loader(test_dataset, batch_size, False),
    )


# TODO: add more split methods (i.e., split by time)


if __name__ == "__main__":
    from pyhealth.datasets import MIMIC3BaseDataset
    from pyhealth.tasks.drug_recommendation import DrugRecDataLoader

    base_dataset = MIMIC3BaseDataset(
        root="/srv/local/data/physionet.org/files/mimiciii/1.4"
    )
    drug_recommendation_dataset = DrugRecDataLoader(base_dataset)
    print(len(drug_recommendation_dataset))
    print(drug_recommendation_dataset[0])
    tmp = random_split(
        drug_recommendation_dataset,
        [len(drug_recommendation_dataset) // 2, len(drug_recommendation_dataset) // 2],
    )
    print(len(tmp[0]))
