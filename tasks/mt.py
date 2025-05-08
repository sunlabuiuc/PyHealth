def chexphoto_multilabel_task(dataset):
    samples = []
    for sample in dataset.data:
        samples.append({
            "input": sample["image"],
            "label": sample["label"]
        })
    return samples