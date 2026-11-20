"""MedSigLIP image embeddings (MedMod extension): a tiny random SigLIP and
synthetic images, no download and no real data."""

from __future__ import annotations

import numpy as np
import pytest
import torch
from PIL import Image

from pyhealth.datasets.medmod.medsiglip import (
    compute_embeddings,
    embed_images,
    list_images,
    load_embeddings,
    load_medsiglip,
    main,
    pooled,
)

transformers = pytest.importorskip("transformers")


@pytest.fixture(scope="module")
def checkpoint(tmp_path_factory):
    from transformers import SiglipConfig, SiglipImageProcessor, SiglipModel

    torch.manual_seed(0)
    small = {"hidden_size": 16, "intermediate_size": 24, "num_hidden_layers": 2,
             "num_attention_heads": 2}
    config = SiglipConfig(
        text_config={"vocab_size": 50, "max_position_embeddings": 8, **small},
        vision_config={"image_size": 32, "patch_size": 8, **small},
    )
    path = tmp_path_factory.mktemp("tiny siglip")
    SiglipModel(config).save_pretrained(path)
    SiglipImageProcessor(size={"height": 32, "width": 32}).save_pretrained(path)
    return path


@pytest.fixture(scope="module")
def images(tmp_path_factory):
    folder = tmp_path_factory.mktemp("my images")  # a space in the path
    rng = np.random.default_rng(0)
    paths = []
    for i in range(5):
        path = folder / f"img{i}.jpg"
        Image.fromarray(rng.integers(0, 256, (40 + i, 36), dtype=np.uint8)).save(path)
        paths.append(path)
    return paths


def test_embedding_matches_full_model_and_batching(checkpoint, images):
    from transformers import SiglipModel

    processor, model = load_medsiglip(str(checkpoint), device="cpu")
    assert not model.training
    assert not any(p.requires_grad for p in model.parameters())
    pics = [Image.open(p) for p in images]
    together = embed_images(pics, processor, model)
    assert together.shape == (5, 16) and together.dtype == np.float32
    one_by_one = np.concatenate([embed_images([p], processor, model) for p in pics])
    np.testing.assert_allclose(one_by_one, together, atol=1e-5)
    full = SiglipModel.from_pretrained(checkpoint).eval()
    pixels = processor(images=[p.convert("RGB") for p in pics], return_tensors="pt")
    with torch.no_grad():
        ref = pooled(full.get_image_features(pixel_values=pixels["pixel_values"]))
    np.testing.assert_allclose(together, ref.numpy(), atol=1e-6)


def test_resume_and_list_file(checkpoint, images, tmp_path, capsys):
    out = tmp_path / "emb.npz"
    listing = tmp_path / "list.txt"
    listing.write_text("\n".join(str(p) for p in images[:3]) + "\n")
    args = ["--out", str(out), "--model", str(checkpoint), "--device", "cpu",
            "--batch-size", "2", "--save-every", "2"]
    main(["--images", str(listing), *args])
    assert len(load_embeddings(out)[0]) == 3
    main(["--images", str(images[0].parent), *args])  # skips the 3 done
    assert "'already_done': 3, 'computed': 2" in capsys.readouterr().out

    index, embeddings, meta = load_embeddings(out)
    assert len(index) == 5 and embeddings.shape == (5, 16)
    assert meta["processor"]["size"] == {"height": 32, "width": 32}
    processor, model = load_medsiglip(str(checkpoint), device="cpu")
    expected = embed_images([Image.open(images[4])], processor, model)[0]
    np.testing.assert_allclose(embeddings[index["img4"]], expected, atol=1e-5)
    with pytest.raises(ValueError, match="other settings"):
        compute_embeddings(images, out, processor, model, {**meta, "dim": 1})
    listed = [p.name for p in list_images(listing)]
    assert listed == ["img0.jpg", "img1.jpg", "img2.jpg"]
