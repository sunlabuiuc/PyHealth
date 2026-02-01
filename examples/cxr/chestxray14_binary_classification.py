import tempfile

from pyhealth.datasets import ChestXray14Dataset, get_dataloader, split_by_sample
from pyhealth.models import CNN
from pyhealth.tasks import ChestXray14BinaryClassification
from pyhealth.trainer import Trainer

# Since PyHealth uses multiprocessing, it is best practice to use a main guard.
if __name__ == '__main__':
    # Use tempfile to automate cleanup
    dataset_dir = tempfile.TemporaryDirectory()
    cache_dir = tempfile.TemporaryDirectory()

    dataset = ChestXray14Dataset(
        root=dataset_dir.name,
        cache_dir=cache_dir.name,
        download=True,
        partial=True,
    )
    dataset.stats()

    task = ChestXray14BinaryClassification(disease="infiltration")
    samples = dataset.set_task(task)

    train_dataset, val_dataset, test_dataset = split_by_sample(samples, [0.7, 0.1, 0.2])

    train_loader = get_dataloader(train_dataset, batch_size=16, shuffle=True)
    val_loader = get_dataloader(val_dataset, batch_size=16, shuffle=False)
    test_loader = get_dataloader(test_dataset, batch_size=16, shuffle=False)

    model = CNN(dataset=samples)

    trainer = Trainer(model=model)
    trainer.train(
        train_dataloader=train_loader,
        val_dataloader=val_loader,
        epochs=1,
    )

    trainer.evaluate(test_loader)

    samples.close()
