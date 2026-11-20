"""Frozen MedSigLIP image embeddings for the MedMod extension.

    python -m pyhealth.datasets.medmod.medsiglip \\
        --images /data/mimic-cxr-jpg/2.0.0/resized --out /data/medsiglip_448.npz

The extension swaps the ResNet-34 image encoder of both models for a frozen
MedSigLIP image encoder (``google/medsiglip-448``). A frozen encoder gives the
same output for an image every time, so each image is embedded once here and
the models read the stored vectors: :func:`lookup_embeddings` gives a batch's
image input for a model built with ``cxr_backbone=nn.Identity()`` and
``cxr_feature_dim=1152`` (the ``--embeddings`` option of the MedMod examples).

* Model: ``google/medsiglip-448`` at a pinned revision (``REVISION``); only the
  vision tower is loaded (``SiglipVisionModel``), frozen and in eval mode.
  transformers lists the text tower's weights as unused; that is expected.
* Images: ``Image.open(...).convert("RGB")`` (the model card's example), then
  the checkpoint's own image processor, so size, resampling, rescaling, mean
  and std all come from its ``preprocessor_config.json``. The inputs are the
  512-px-wide resized JPEGs.
* Embedding: the vision tower's pooled output, which is what
  ``get_image_features`` returns; stored unnormalized (the model card's
  ``image_embeds`` are the same vectors L2-normalized).
* Output: one ``.npz`` with ``keys`` (image file name without extension),
  ``embeddings`` (float32, ``[n, dim]``) and ``meta`` (model, revision,
  processor settings, as JSON). It is rewritten every ``save_every`` images,
  and a rerun skips the images already in it. Prints counts only.

Runs with transformers 4.x and 5.x, and also as a standalone script
(``python medsiglip.py ...``; needs numpy, Pillow, torch, transformers).
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

MODEL_ID = "google/medsiglip-448"
# ``main`` of google/medsiglip-448 on the Hugging Face Hub on 2026-11-19.
REVISION = "9cea28a1a1195f665105faa6e8544c112fd960a4"
IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png")
PROCESSOR_KEYS = (
    "size",
    "resample",
    "do_resize",
    "do_rescale",
    "rescale_factor",
    "do_normalize",
    "image_mean",
    "image_std",
    "do_convert_rgb",
)


def default_device() -> str:
    """CUDA if available, then MPS, else CPU.

    Returns:
        The device name.

    Examples:
        >>> from pyhealth.datasets.medmod.medsiglip import default_device
        >>> default_device() in ("cuda", "mps", "cpu")
        True
    """
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def pooled(output: Any) -> torch.Tensor:
    """The pooled image features, whether a model returned a tensor
    (transformers 4.x ``get_image_features``) or an output object (5.x, and
    the vision tower in both).

    Args:
        output: A tensor, or an object with ``pooler_output``.

    Returns:
        The tensor.

    Examples:
        >>> import torch
        >>> from pyhealth.datasets.medmod.medsiglip import pooled
        >>> pooled(torch.ones(2, 3)).shape
        torch.Size([2, 3])
    """
    return output if isinstance(output, torch.Tensor) else output.pooler_output


def load_medsiglip(
    model_id: str = MODEL_ID,
    revision: str | None = REVISION,
    device: str | None = None,
) -> tuple[Any, Any]:
    """The checkpoint's image processor and its frozen vision tower.

    Args:
        model_id: Hub id, or a local folder (then ``revision`` is ignored).
        revision: Hub revision.
        device: Torch device; None picks :func:`default_device`.

    Returns:
        ``(processor, model)``.

    Examples:
        >>> from pyhealth.datasets.medmod.medsiglip import load_medsiglip
        >>> processor, model = load_medsiglip()  # doctest: +SKIP
    """
    import transformers
    from transformers import AutoImageProcessor, SiglipVisionModel

    if Path(model_id).is_dir():
        revision = None
    # transformers 5 renamed ``torch_dtype`` to ``dtype``.
    major = int(transformers.__version__.split(".")[0])
    dtype_arg = {"dtype" if major >= 5 else "torch_dtype": torch.float32}
    processor = AutoImageProcessor.from_pretrained(model_id, revision=revision)
    model = SiglipVisionModel.from_pretrained(model_id, revision=revision, **dtype_arg)
    model.requires_grad_(False)
    model.eval().to(device or default_device())
    return processor, model


@torch.inference_mode()
def embed_images(
    images: Sequence[Image.Image], processor: Any, model: Any
) -> np.ndarray:
    """Embeddings of PIL images, float32 ``[n, dim]``.

    Args:
        images: PIL images of any mode and size.
        processor: From :func:`load_medsiglip`.
        model: From :func:`load_medsiglip`.

    Returns:
        The embeddings.

    Examples:
        >>> from PIL import Image
        >>> from pyhealth.datasets.medmod.medsiglip import (
        ...     embed_images, load_medsiglip,
        ... )
        >>> processor, model = load_medsiglip()  # doctest: +SKIP
        >>> pic = Image.new("L", (512, 600))
        >>> embed_images([pic], processor, model).shape  # doctest: +SKIP
        (1, 1152)
    """
    rgb = [image.convert("RGB") for image in images]
    pixels = processor(images=rgb, return_tensors="pt")["pixel_values"]
    out = model(pixel_values=pixels.to(model.device, model.dtype))
    return pooled(out).float().cpu().numpy()


def processor_settings(processor: Any) -> dict[str, Any]:
    """The processor values that shape the input, for the record.

    Args:
        processor: A transformers image processor.

    Returns:
        Size, resampling, rescaling and normalization settings.

    Examples:
        >>> from transformers import SiglipImageProcessor
        >>> from pyhealth.datasets.medmod.medsiglip import processor_settings
        >>> p = SiglipImageProcessor(size={"height": 448, "width": 448})
        >>> processor_settings(p)["size"]
        {'height': 448, 'width': 448}
    """
    config = processor.to_dict()
    settings = {k: config[k] for k in PROCESSOR_KEYS if k in config}
    settings["type"] = type(processor).__name__
    return json.loads(json.dumps(settings, default=int))


def image_key(path: str | Path) -> str:
    """The key of an image: its file name without the extension.

    Args:
        path: Image path.

    Returns:
        The key.

    Examples:
        >>> from pyhealth.datasets.medmod.medsiglip import image_key
        >>> image_key("/data/resized/abc.jpg")
        'abc'
    """
    return Path(path).stem


def list_images(images: str | Path) -> list[Path]:
    """Image files in a folder (not recursive), or listed one per line in a
    text file (so paths may contain spaces).

    Args:
        images: A folder, or a list file.

    Returns:
        Sorted paths.

    Examples:
        >>> import os, tempfile
        >>> from pyhealth.datasets.medmod.medsiglip import list_images
        >>> listing = os.path.join(tempfile.mkdtemp(), "list.txt")
        >>> _ = open(listing, "w").write("/my data/b.jpg\\n\\n/my data/a.jpg\\n")
        >>> [p.name for p in list_images(listing)]
        ['a.jpg', 'b.jpg']
    """
    images = Path(images)
    if images.is_dir():
        return sorted(p for p in images.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)
    with open(images) as f:
        return sorted(Path(line.rstrip("\r\n")) for line in f if line.strip())


def save_embeddings(
    path: str | Path, keys: Sequence[str], embeddings: np.ndarray, meta: dict
) -> Path:
    """Writes the ``.npz`` atomically (a temporary file, then a rename).

    Args:
        path: Output file.
        keys: One key per row.
        embeddings: ``[n, dim]``, stored as float32.
        meta: Settings, stored as JSON.

    Returns:
        The path.

    Examples:
        >>> import os, tempfile
        >>> import numpy as np
        >>> from pyhealth.datasets.medmod.medsiglip import save_embeddings
        >>> out = os.path.join(tempfile.mkdtemp(), "emb.npz")
        >>> save_embeddings(out, ["a"], np.zeros((1, 4)), {"model": "x"}).name
        'emb.npz'
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=".tmp-", suffix=".npz")
    with os.fdopen(fd, "wb") as f:
        np.savez(
            f,
            keys=np.array(list(keys), dtype=str),
            embeddings=np.asarray(embeddings, dtype=np.float32),
            meta=np.array(json.dumps(meta)),
        )
    os.replace(tmp, path)
    return path


def load_embeddings(path: str | Path) -> tuple[dict[str, int], np.ndarray, dict]:
    """Reads a file written by :func:`save_embeddings`.

    Args:
        path: The ``.npz``.

    Returns:
        ``(index, embeddings, meta)``: ``index`` maps a key to its row.

    Examples:
        >>> import os, tempfile
        >>> import numpy as np
        >>> from pyhealth.datasets.medmod.medsiglip import (
        ...     load_embeddings, save_embeddings,
        ... )
        >>> out = os.path.join(tempfile.mkdtemp(), "emb.npz")
        >>> _ = save_embeddings(out, ["a", "b"], np.eye(2), {"model": "x"})
        >>> index, emb, meta = load_embeddings(out)
        >>> index, emb.shape, meta
        ({'a': 0, 'b': 1}, (2, 2), {'model': 'x'})
    """
    with np.load(path, allow_pickle=False) as data:
        keys = [str(k) for k in data["keys"]]
        return (
            {k: i for i, k in enumerate(keys)},
            data["embeddings"],
            json.loads(str(data["meta"])),
        )


def lookup_embeddings(
    paths: Sequence[str | Path], index: dict[str, int], embeddings: np.ndarray
) -> torch.Tensor:
    """The stored embeddings of a batch of images, looked up by image path.

    Args:
        paths: Image paths, e.g. a batch's ``image_path``.
        index: Key to row, from :func:`load_embeddings`.
        embeddings: ``[n, dim]``, from :func:`load_embeddings`.

    Returns:
        A float32 tensor ``[len(paths), dim]``: the image input of a model
        built with ``cxr_backbone=nn.Identity()`` and ``cxr_feature_dim=dim``.

    Raises:
        KeyError: An image has no embedding (its name is not shown).

    Examples:
        >>> import numpy as np
        >>> from pyhealth.datasets.medmod.medsiglip import lookup_embeddings
        >>> table = np.arange(6, dtype=np.float32).reshape(2, 3)
        >>> lookup_embeddings(["/x/b.jpg", "/y/a.jpg"], {"a": 0, "b": 1}, table)
        tensor([[3., 4., 5.],
                [0., 1., 2.]])
    """
    try:
        rows = [index[image_key(p)] for p in paths]
    except KeyError:
        raise KeyError("An image has no embedding in the file.") from None
    return torch.from_numpy(embeddings[rows])


def compute_embeddings(
    paths: Sequence[str | Path],
    out: str | Path,
    processor: Any,
    model: Any,
    meta: dict,
    batch_size: int = 16,
    save_every: int = 256,
) -> dict[str, float]:
    """Embeds the images not yet in ``out`` and saves as it goes.

    Args:
        paths: Image files with unique file names.
        out: The ``.npz``; extended if it exists and was made with ``meta``.
        processor: From :func:`load_medsiglip`.
        model: From :func:`load_medsiglip`.
        meta: Settings recorded in the file (see :func:`main`).
        batch_size: Images per forward pass.
        save_every: Rewrite the file after about this many new images.

    Returns:
        Counts: ``images``, ``already_done``, ``computed``, ``seconds``.

    Examples:
        >>> from pyhealth.datasets.medmod.medsiglip import compute_embeddings
        >>> out = "emb.npz"
        >>> compute_embeddings(paths, out, processor, model, meta)  # doctest: +SKIP
    """
    keys = [image_key(p) for p in paths]
    if len(set(keys)) != len(keys):
        raise ValueError(f"{len(keys) - len(set(keys))} images share a file name.")
    done_keys: list[str] = []
    rows: list[np.ndarray] = []
    if Path(out).exists():
        index, embeddings, old_meta = load_embeddings(out)
        if old_meta != json.loads(json.dumps(meta)):
            raise ValueError(f"{out} was made with other settings; use another file.")
        done_keys, rows = list(index), [embeddings]
    done = set(done_keys)
    todo = [p for p, k in zip(paths, keys) if k not in done]

    start, since_save = time.time(), 0
    for i in range(0, len(todo), batch_size):
        batch = todo[i : i + batch_size]
        images = []
        for p in batch:
            with Image.open(p) as image:
                image.load()
                images.append(image)
        rows.append(embed_images(images, processor, model))
        done_keys.extend(image_key(p) for p in batch)
        since_save += len(batch)
        last = i + batch_size >= len(todo)
        if since_save >= save_every or last:
            save_embeddings(out, done_keys, np.concatenate(rows), meta)
            rows, since_save = [np.concatenate(rows)], 0
            n = i + len(batch)
            rate = n / (time.time() - start)
            print(f"  {n}/{len(todo)} new images saved, {rate:.2f} img/s", flush=True)
    return {
        "images": len(paths),
        "already_done": len(paths) - len(todo),
        "computed": len(todo),
        "seconds": round(time.time() - start, 1),
    }


def main(argv: Sequence[str] | None = None) -> None:
    """Command line: embed a folder (or list file) of images.

    Args:
        argv: Arguments; None reads ``sys.argv``.

    Examples:
        >>> from pyhealth.datasets.medmod.medsiglip import main
        >>> main(["--images", "resized", "--out", "emb.npz"])  # doctest: +SKIP
    """
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--images", required=True, help="folder, or a list file")
    ap.add_argument("--out", required=True, help="output .npz")
    ap.add_argument("--model", default=MODEL_ID, help="Hub id or local folder")
    ap.add_argument("--revision", default=REVISION)
    ap.add_argument("--device", default=None)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--save-every", type=int, default=256)
    ap.add_argument("--limit", type=int, default=0, help="first N images only")
    args = ap.parse_args(argv)

    paths = list_images(args.images)
    if args.limit:
        paths = paths[: args.limit]
    processor, model = load_medsiglip(args.model, args.revision, args.device)
    meta = {
        "model_id": args.model,
        "revision": None if Path(args.model).is_dir() else args.revision,
        "embedding": "vision tower pooler_output, float32, not L2-normalized",
        "dim": int(model.config.hidden_size),
        "processor": processor_settings(processor),
    }
    print(f"{len(paths)} images on {model.device}; {json.dumps(meta)}")
    counts = compute_embeddings(
        paths, args.out, processor, model, meta, args.batch_size, args.save_every
    )
    print(f"done: {counts}")


if __name__ == "__main__":
    main()
