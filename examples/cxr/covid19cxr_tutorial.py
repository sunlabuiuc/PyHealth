"""COVID-19 CXR Tutorial: ViT Training, Conformal Prediction & Interpretability."""
import os, numpy as np, torch, matplotlib.pyplot as plt
from pyhealth.calib.predictionset import LABEL
from pyhealth.datasets import COVID19CXRDataset, get_dataloader, split_by_sample_conformal
from pyhealth.models import TorchvisionModel
from pyhealth.trainer import Trainer
from pyhealth.interpret.methods import CheferRelevance
from pyhealth.interpret.utils import visualize_image_attr

# Paths
ROOT = "/home/johnwu3/projects/PyHealth_Branch_Testing/datasets/COVID-19_Radiography_Dataset"
CACHE, TASK_CACHE = "/home/johnwu3/projects/covid19cxr_base_cache", "/home/johnwu3/projects/covid19cxr_task_cache"
CKPT = "/home/johnwu3/projects/covid19cxr_vit_model.ckpt"

if __name__ == "__main__":
    # 1. Load dataset
    dataset = COVID19CXRDataset(ROOT, cache_dir=CACHE, num_workers=4).set_task(cache_dir=TASK_CACHE, num_workers=4)
    id2label = {v: k for k, v in dataset.output_processors["disease"].label_vocab.items()}
    train, val, cal, test = split_by_sample_conformal(dataset, ratios=[0.6, 0.1, 0.15, 0.15])

    # 2. Train/load ViT model
    model = TorchvisionModel(dataset=dataset, model_name="vit_b_16", model_config={"weights": "DEFAULT"})
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    if os.path.exists(CKPT):
        trainer = Trainer(model=model, device=device, checkpoint_path=CKPT)
    else:
        trainer = Trainer(model=model, device=device)
        trainer.train(train_dataloader=get_dataloader(train, 64, shuffle=True),
                      val_dataloader=get_dataloader(val, 64), epochs=20, monitor="accuracy")
        trainer.save_ckpt(CKPT)

    # 3. Conformal prediction with LABEL
    label_predictor = LABEL(model=model, alpha=0.1)
    label_predictor.calibrate(cal_dataset=cal)
    _, _, _, extra = Trainer(model=label_predictor).inference(
        get_dataloader(test, 64), additional_outputs=["y_predset"])
    predset = torch.tensor(extra["y_predset"])
    print(f"Coverage: {1 - predset.float().mean():.1%}, Avg set size: {predset.sum(1).float().mean():.2f}")

    # 4. Interpretability visualization
    model.eval()
    chefer = CheferRelevance(model)
    loader = get_dataloader(test, batch_size=1, shuffle=False)
    fig, axes = plt.subplots(3, 3, figsize=(12, 12))

    for idx, batch in enumerate([b for i, b in enumerate(loader) if i < 3]):
        with torch.no_grad():
            pred_prob = model(**batch)["y_prob"][0]
        attr = chefer.attribute(class_index=pred_prob.argmax().item(), **batch)["image"]
        img, heatmap, overlay = visualize_image_attr(batch["image"][0], attr[0, 0])

        axes[idx, 0].imshow(img, cmap='gray'); axes[idx, 0].set_title(f"True: {id2label[batch['disease'].item()]}")
        axes[idx, 1].imshow(heatmap, cmap='hot'); axes[idx, 1].set_title(f"Pred: {id2label[pred_prob.argmax().item()]}")
        axes[idx, 2].imshow(overlay); axes[idx, 2].set_title(f"Conf: {pred_prob.max():.1%}")
        [ax.axis('off') for ax in axes[idx]]

    plt.tight_layout()
    plt.savefig("covid19_cxr_interpretability.png", dpi=150, bbox_inches='tight')
