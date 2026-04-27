import os
import tempfile

from pyhealth.datasets import MRIDataset, get_dataloader, split_by_sample
from pyhealth.models import CNN
from pyhealth.processors import NiftiImageProcessor
from pyhealth.tasks import MRIBinaryClassification
from pyhealth.trainer import Trainer

# Since PyHealth uses multiprocessing, it is best practice to use a main guard.
if __name__ == "__main__":
    # Use tempfile to automate cleanup
    dataset_dir = tempfile.TemporaryDirectory()
    cache_dir = tempfile.TemporaryDirectory()

    dataset = MRIDataset(
        root=dataset_dir.name,
        cache_dir=cache_dir.name,
        download=True,
        partial=True,
    )
    dataset.stats()

    task = MRIBinaryClassification(disease="alzheimer")
    samples = dataset.set_task(
        task,
        input_processors={"image": NiftiImageProcessor()},
    )

    train_dataset, val_dataset, test_dataset = split_by_sample(samples, [0.7, 0.1, 0.2])

    train_loader = get_dataloader(train_dataset, batch_size=8, shuffle=True)
    val_loader = get_dataloader(val_dataset, batch_size=8, shuffle=False)
    test_loader = get_dataloader(test_dataset, batch_size=8, shuffle=False)

    model = CNN(dataset=samples)

    # Default to CPU to avoid CUDA runtime mismatch on unsupported GPUs.
    device = os.environ.get("PYHEALTH_DEVICE", "cpu")
    trainer = Trainer(model=model, device=device)
    trainer.train(
        train_dataloader=train_loader,
        val_dataloader=val_loader,
        epochs=1,
    )

    trainer.evaluate(test_loader)

    samples.close()
