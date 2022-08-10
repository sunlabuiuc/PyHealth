import torch


def random_split(dataset, ratios):
    assert sum(ratios) == 1.0
    lengths = [int(len(dataset) * r) for r in ratios]
    lengths[-1] = len(dataset) - sum(lengths[:-1])
    return torch.utils.data.random_split(dataset, lengths)


# TODO: add more split methods (i.e., split by time)


if __name__ == "__main__":
    from pyhealth.datasets import MIMIC3BaseDataset
    from pyhealth.tasks.drug_recommendation import DrugRecommendationDataset

    base_dataset = MIMIC3BaseDataset(root="/srv/local/data/physionet.org/files/mimiciii/1.4")
    drug_recommendation_dataset = DrugRecommendationDataset(base_dataset)
    print(len(drug_recommendation_dataset))
    print(drug_recommendation_dataset[0])
    tmp = random_split(drug_recommendation_dataset,
                       [len(drug_recommendation_dataset) // 2, len(drug_recommendation_dataset) // 2])
    print(len(tmp[0]))
