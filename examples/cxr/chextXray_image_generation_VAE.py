"""Train a VAE on chest X-rays and generate new ones.

Uses the COVID-19 Radiography Database. Set ``root`` to the folder that holds
its four class folders and their metadata spreadsheets.
"""

import matplotlib.pyplot as plt

from pyhealth.datasets import COVID19CXRDataset, get_dataloader, split_by_sample
from pyhealth.models import VAE
from pyhealth.tasks import COVID19CXRImageGeneration
from pyhealth.trainer import Trainer


def main(root: str = "/srv/local/data/COVID-19_Radiography_Dataset") -> None:
    # step 1: load the dataset
    base_dataset = COVID19CXRDataset(root)

    # step 2: one grayscale 128 x 128 image per sample, with no label
    sample_dataset = base_dataset.set_task(COVID19CXRImageGeneration())
    train_dataset, val_dataset, test_dataset = split_by_sample(
        sample_dataset, [0.8, 0.1, 0.1], seed=0
    )
    train_dataloader = get_dataloader(train_dataset, batch_size=64, shuffle=True)
    val_dataloader = get_dataloader(val_dataset, batch_size=64, shuffle=False)
    test_dataloader = get_dataloader(test_dataset, batch_size=64, shuffle=False)

    # step 3: the model reads the image shape from the task
    model = VAE(dataset=sample_dataset, hidden_dim=128)

    # step 4: train, keeping the epoch with the lowest validation reconstruction
    # error
    trainer = Trainer(model=model, metrics=["mse", "mae"])
    trainer.train(
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
        epochs=10,
        monitor="mse",
        monitor_criterion="min",
        optimizer_params={"lr": 1e-3},
    )
    print(trainer.evaluate(test_dataloader))

    # step 5: a real X-ray next to its reconstruction
    y_true, y_prob, _ = trainer.inference(test_dataloader)
    plt.figure()
    plt.subplot(1, 2, 1)
    plt.imshow(y_true[0, 0], cmap="gray")
    plt.title("real")
    plt.subplot(1, 2, 2)
    plt.imshow(y_prob[0, 0], cmap="gray")
    plt.title("reconstructed")
    plt.savefig("chestxray_vae_comparison.png")

    # step 6: new X-rays decoded from random points of the latent space
    images = model.generate(num_samples=4).cpu()
    plt.figure()
    for i, image in enumerate(images):
        plt.subplot(1, 4, i + 1)
        plt.imshow(image[0], cmap="gray")
        plt.axis("off")
    plt.savefig("chestxray_vae_synthetic.png")


if __name__ == "__main__":
    main()
