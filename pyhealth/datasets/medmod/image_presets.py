"""Chest X-ray transforms per model: training and evaluation.

    get_preset("early_fusion")              # Early Fusion, val/test
    get_preset("early_fusion", train=True)  # Early Fusion, train (augmented)
    get_preset("convirt")                   # ConVIRT + linear eval, val/test
    get_preset("convirt", train=True)       # ConVIRT + linear eval, train

Early Fusion uses MedFuse's loaders and ConVIRT MedMod's ``simclrv2`` ones
(see the MedModInHospitalMortality docs, "Per-model settings").
Training transforms:

``early_fusion``, train (``medfuse_train``): MedFuse
    datasets/cxr_dataset.py:65-72 (literal sizes, not --resize): Resize(256),
    RandomHorizontalFlip(), RandomAffine(degrees=45,
    scale=(.85, 1.15), shear=0, translate=(0.15, 0.15)), CenterCrop(224),
    ToTensor(), ImageNet Normalize. Applied to the train split
    (cxr_dataset.py:99).
``convirt``, train (``medmod_simclrv2_train``): MedMod src/data/utils.py:80-101
    (``simclrv2``, chosen by --transforms_cxr simclrv2 in the ConVIRT scripts,
    utils.py:115-116):
    Resize([256, 256]), RandomAffine(degrees=(-45, 45), translate=(0.1, 0.1),
    scale=(0.7, 1.5), shear=(-25, 25)), ``RandomCrop`` (square, side drawn
    from U(0.6 * 256, 256), lines 46-53), Resize([224, 224], bicubic),
    RandomHorizontalFlip(), ``RandomColorDistortion`` (ColorJitter(0.8, 0.8,
    0.8, 0.2) with probability 0.8, then RandomGrayscale(0.2), lines 57-75),
    ToTensor(), no Normalize. Applied to the train split (utils.py:127), in
    pretraining and in linear eval / fine-tuning alike. As in MedMod, the two
    random draws use numpy's global RNG (``np.random.uniform``); seed it in
    each DataLoader worker for reproducible runs.

Evaluation transforms:

``early_fusion``, eval (``medfuse_eval``): MedFuse test transforms
    (github.com/nyuad-cai/MedFuse datasets/cxr_dataset.py:75-82, with
    --resize 256 / --crop 224 defaults
    from arguments.py:10-11): Resize(256), CenterCrop(224), ToTensor(),
    ImageNet Normalize. MedMod's own copy, src/data/utils.py:37-40, is the
    same minus Normalize (it defines `normalize` at line 27 but never appends
    it).
``convirt``, eval (``medmod_simclrv2_eval``): MedMod's `simclrv2` test transforms,
    src/data/utils.py:99-107 with --resize 256 (src/parser.py:44):
    Resize([256, 256]), CenterCrop([224, 224]) (int(0.875 * 256) = 224),
    Resize([224, 224], interpolation=3 i.e. bicubic), ToTensor(), no Normalize.

Both MedFuse and MedMod open images with ``.convert('RGB')``
(MedMod src/data/datasets.py:64). They also read pre-resized copies:
MedMod globs ``{tmp_dir}/resized/**/*.jpg`` (src/data/utils.py:125), made by
MedFuse resize.py:27-37 (width -> 512, height scaled, saved as JPEG).
``pyhealth.datasets.medmod.resize_cxr`` reproduces that step on disk (settings and
sources
are listed there). ``load_cxr(..., medfuse_preresize=True)`` does the same
in memory for an original image: same resize, then a JPEG round trip with
Pillow's default quality.
"""

from __future__ import annotations

import io

import numpy as np
from PIL import Image
from torchvision import transforms
from torchvision.transforms import InterpolationMode

from .resize_cxr import medfuse_resize

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def medfuse_eval(resize: int = 256, crop: int = 224) -> transforms.Compose:
    """MedFuse's eval transform: resize, center crop, ImageNet
    normalization.

    Examples:
        >>> from PIL import Image
        >>> from pyhealth.datasets.medmod.image_presets import medfuse_eval
        >>> medfuse_eval()(Image.new("RGB", (512, 600))).shape
        torch.Size([3, 224, 224])
    """
    return transforms.Compose(
        [
            transforms.Resize(resize),
            transforms.CenterCrop(crop),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


def medfuse_train() -> transforms.Compose:
    """MedFuse's train transform: ``medfuse_eval`` plus flip and
    affine augmentation.

    Examples:
        >>> from PIL import Image
        >>> from pyhealth.datasets.medmod.image_presets import medfuse_train
        >>> medfuse_train()(Image.new("RGB", (512, 600))).shape
        torch.Size([3, 224, 224])
    """
    return transforms.Compose(
        [
            transforms.Resize(256),
            transforms.RandomHorizontalFlip(),
            transforms.RandomAffine(
                degrees=45, scale=(0.85, 1.15), shear=0, translate=(0.15, 0.15)
            ),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


class RandomCrop:
    """MedMod ``RandomCrop`` (src/data/utils.py:46-53): a square crop whose
    side is drawn from U(0.6 * 256, 256), truncated to int.

    Examples:
        >>> from PIL import Image
        >>> from pyhealth.datasets.medmod.image_presets import RandomCrop
        >>> w, h = RandomCrop()(Image.new("RGB", (256, 256))).size
        >>> w == h and 153 <= w <= 256
        True
    """

    def __call__(self, img):
        resize = 256
        size = int(np.random.uniform(0.6 * resize, resize, 1)[0])
        return transforms.RandomCrop(size)(img)


class RandomColorDistortion:
    """MedMod ``RandomColorDistortion`` (src/data/utils.py:57-75).

    Examples:
        >>> from PIL import Image
        >>> from pyhealth.datasets.medmod.image_presets import RandomColorDistortion
        >>> RandomColorDistortion()(Image.new("RGB", (224, 224))).size
        (224, 224)
    """

    def __call__(self, img):
        strength = 1.0
        if np.random.uniform(0, 1, 1)[0] < 0.8:
            img = transforms.ColorJitter(
                0.8 * strength, 0.8 * strength, 0.8 * strength, 0.2 * strength
            )(img)
        return transforms.RandomGrayscale(p=0.2)(img)


def medmod_simclrv2_train(resize: int = 256) -> transforms.Compose:
    """MedMod's simclrv2 train transform (pixels in [0, 1],
    no normalization).

    Examples:
        >>> from PIL import Image
        >>> from pyhealth.datasets.medmod.image_presets import medmod_simclrv2_train
        >>> medmod_simclrv2_train()(Image.new("RGB", (512, 600))).shape
        torch.Size([3, 224, 224])
    """
    return transforms.Compose(
        [
            transforms.Resize([resize, resize]),
            transforms.RandomAffine(
                degrees=(-45, 45),
                translate=(0.1, 0.1),
                scale=(0.7, 1.5),
                shear=(-25, 25),
            ),
            RandomCrop(),
            transforms.Resize([224, 224], interpolation=InterpolationMode.BICUBIC),
            transforms.RandomHorizontalFlip(),
            RandomColorDistortion(),
            transforms.ToTensor(),
        ]
    )


def medmod_simclrv2_eval(resize: int = 256) -> transforms.Compose:
    """MedMod's simclrv2 eval transform: resize, center crop
    0.875, resize to 224 (pixels in [0, 1]).

    Examples:
        >>> from PIL import Image
        >>> from pyhealth.datasets.medmod.image_presets import medmod_simclrv2_eval
        >>> medmod_simclrv2_eval()(Image.new("RGB", (512, 600))).shape
        torch.Size([3, 224, 224])
    """
    c = int(0.875 * resize)
    return transforms.Compose(
        [
            transforms.Resize([resize, resize]),
            transforms.CenterCrop([c, c]),
            transforms.Resize([224, 224], interpolation=InterpolationMode.BICUBIC),
            transforms.ToTensor(),
        ]
    )


# name -> (train, eval)
PRESETS = {
    "early_fusion": (
        medfuse_train,
        medfuse_eval,
    ),  # MedFuse loaders, ImageNet-normalized
    "convirt": (medmod_simclrv2_train, medmod_simclrv2_eval),  # MedMod simclrv2, [0, 1]
}


def get_preset(name: str, train: bool = False) -> transforms.Compose:
    """The image transform for a model: ``"early_fusion"`` (MedFuse's,
    ImageNet-normalized) or ``"convirt"`` (MedMod's simclrv2, pixels in [0, 1];
    pretraining and linear eval). ``train=True`` adds the training augmentation;
    use it for the train split only.

    Examples:
        >>> from PIL import Image
        >>> from pyhealth.datasets.medmod.image_presets import get_preset
        >>> get_preset("convirt", train=True)(Image.new("RGB", (512, 600))).shape
        torch.Size([3, 224, 224])
    """
    if name not in PRESETS:
        raise ValueError(f"unknown preset {name!r}; choose from {sorted(PRESETS)}")
    train_fn, eval_fn = PRESETS[name]
    return train_fn() if train else eval_fn()


def _medfuse_preresize(img: Image.Image) -> Image.Image:
    """MedFuse resize.py:27-37 plus the JPEG save/load round trip."""
    buf = io.BytesIO()
    medfuse_resize(img).save(buf, format="JPEG")  # default quality, as resize.py:37
    buf.seek(0)
    return Image.open(buf)


def load_cxr(path: str, medfuse_preresize: bool = False) -> Image.Image:
    """Open a chest X-ray as RGB. ``medfuse_preresize=True`` first applies
    MedFuse's 512-px resize and JPEG round trip (for originals).

    Examples:
        >>> import os, tempfile
        >>> from PIL import Image
        >>> from pyhealth.datasets.medmod.image_presets import load_cxr
        >>> path = os.path.join(tempfile.mkdtemp(), "cxr.jpg")
        >>> Image.new("L", (1024, 1200)).save(path)
        >>> img = load_cxr(path, medfuse_preresize=True)
        >>> img.size, img.mode
        ((512, 600), 'RGB')
    """
    img = Image.open(path)
    if medfuse_preresize:
        img = _medfuse_preresize(img)
    return img.convert("RGB")
