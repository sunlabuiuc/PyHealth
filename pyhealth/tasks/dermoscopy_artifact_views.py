# Contributor: Aayan Verma (aayanv2)
# Contributor: Bryson Lee (brysonl3)
# Contributor: Xin Zhang (xinz13)
# Paper: A Study of Artifacts on Melanoma Classification under Diffusion-Based
# Perturbations (Jin and Ghassemi, CHIL 2025)
# https://proceedings.mlr.press/v287/jin25b.html
"""Image views and synthetic artifacts for dermoscopic artifact-sensitivity studies.

Pure NumPy/PIL helpers used by
:class:`~pyhealth.tasks.PH2ArtifactClassification`. They reproduce the twelve
"dataset modes" of the paper, and add hand-drawn (non-generative) artifacts
as a control for the paper's diffusion-inpainted ones.

All images are ``float32`` arrays of shape ``(H, W, 3)`` with values in
``[0, 1]``. All masks are boolean arrays of shape ``(H, W)``, True on the
lesion.
"""

from collections.abc import Callable

import numpy as np
from PIL import Image, ImageDraw

#: The twelve dataset modes of the paper. ``*_high`` / ``*_low`` are the
#: high-/low-pass filtered versions of ``whole``, ``lesion`` and ``background``.
VIEWS: tuple[str, ...] = (
    "whole",
    "lesion",
    "background",
    "bbox",
    "bbox70",
    "bbox90",
    "whole_high",
    "lesion_high",
    "background_high",
    "whole_low",
    "lesion_low",
    "background_low",
)

#: The five artifacts the paper teaches its diffusion models.
ARTIFACTS: tuple[str, ...] = (
    "dark_corner",
    "gel_bubble",
    "ink",
    "patches",
    "ruler",
)

_BBOX_COVERAGE = {"bbox": None, "bbox70": 0.7, "bbox90": 0.9}
_FILTER_SIGMA = 1.0  # sigma of the paper's Gaussian low/high-pass filters


# --------------------------------------------------------------------------
# Views
# --------------------------------------------------------------------------
def gaussian_blur(image: np.ndarray, sigma: float = _FILTER_SIGMA) -> np.ndarray:
    """Blur an ``(H, W, 3)`` image with a separable Gaussian (edge-replicated).

    Args:
        image: Float image of shape ``(H, W, 3)``.
        sigma: Standard deviation of the Gaussian, in pixels.

    Returns:
        The blurred image, same shape and dtype as ``image``.

    Examples:
        >>> flat = np.full((8, 8, 3), 0.5, dtype=np.float32)
        >>> bool(np.allclose(gaussian_blur(flat), 0.5))
        True
    """
    radius = max(1, int(np.ceil(3 * sigma)))
    offsets = np.arange(-radius, radius + 1)
    kernel = np.exp(-(offsets**2) / (2 * sigma**2))
    kernel = (kernel / kernel.sum()).astype(image.dtype)
    out = image
    for axis in (0, 1):
        pad = [(0, 0)] * out.ndim
        pad[axis] = (radius, radius)
        padded = np.pad(out, pad, mode="edge")
        size = out.shape[axis]
        out = sum(
            weight * np.take(padded, np.arange(i, i + size), axis=axis)
            for i, weight in enumerate(kernel)
        )
    return out.astype(image.dtype)


def lesion_bbox(mask: np.ndarray) -> tuple[int, int, int, int]:
    """Return the tight bounding box ``(top, bottom, left, right)`` of a mask.

    ``bottom`` and ``right`` are exclusive. An empty mask returns an empty box
    at the image centre.

    Args:
        mask: Boolean lesion mask of shape ``(H, W)``.

    Returns:
        The box as ``(top, bottom, left, right)``.

    Examples:
        >>> m = np.zeros((10, 10), dtype=bool)
        >>> m[2:4, 5:8] = True
        >>> lesion_bbox(m)
        (2, 4, 5, 8)
    """
    rows, cols = np.where(mask)
    if rows.size == 0:
        h, w = mask.shape
        return h // 2, h // 2, w // 2, w // 2
    return int(rows.min()), int(rows.max()) + 1, int(cols.min()), int(cols.max()) + 1


def bbox_mask(mask: np.ndarray, coverage: float | None = None) -> np.ndarray:
    """Return a boolean box around the lesion, optionally grown to a coverage.

    With ``coverage=None`` this is the lesion's tight bounding box. Otherwise
    the box is scaled about its centre (keeping its aspect ratio) until it
    covers at least ``coverage`` of the image, as in the paper's ``bbox70`` and
    ``bbox90`` modes. The box is clipped at the image border, so for a lesion
    near a corner it grows mostly away from that corner; the search runs up to
    the scale at which the box fills the whole image, so any target is reached.
    If the tight box already covers more, it is returned unchanged.

    Args:
        mask: Boolean lesion mask of shape ``(H, W)``.
        coverage: Target fraction of the image to cover, in ``(0, 1)``, or
            None for the tight box.

    Returns:
        Boolean array of shape ``(H, W)``, True inside the box.

    Examples:
        >>> m = np.zeros((20, 20), dtype=bool)
        >>> m[8:12, 8:12] = True
        >>> int(bbox_mask(m).sum())
        16
        >>> bool(bbox_mask(m, 0.7).mean() >= 0.7)
        True
    """
    h, w = mask.shape
    top, bottom, left, right = lesion_bbox(mask)
    centre_y, centre_x = (top + bottom) / 2, (left + right) / 2
    half_h = max((bottom - top) / 2, 0.5)
    half_w = max((right - left) / 2, 0.5)

    def box(scale: float) -> np.ndarray:
        """Return the tight box scaled by ``scale`` about its centre.

        Args:
            scale: Factor applied to the box's half height and half width.

        Returns:
            Boolean array of shape ``(H, W)``, True inside the scaled box.
        """
        out = np.zeros((h, w), dtype=bool)
        y0 = max(0, int(np.floor(centre_y - half_h * scale)))
        y1 = min(h, int(np.ceil(centre_y + half_h * scale)))
        x0 = max(0, int(np.floor(centre_x - half_w * scale)))
        x1 = min(w, int(np.ceil(centre_x + half_w * scale)))
        out[y0:y1, x0:x1] = True
        return out

    tight = box(1.0)
    if coverage is None or tight.mean() >= coverage:
        return tight
    # At this scale the box reaches every image border from any centre.
    low, high = 1.0, max(h / half_h, w / half_w)
    for _ in range(30):  # bisection on the (monotone) scale
        middle = (low + high) / 2
        if box(middle).mean() >= coverage:
            high = middle
        else:
            low = middle
    return box(high)


def make_view(image: np.ndarray, mask: np.ndarray, view: str) -> np.ndarray:
    """Build one of the paper's twelve dataset modes from an image and mask.

    - ``whole``: the image unchanged.
    - ``lesion`` / ``background``: everything outside / inside the lesion
      set to black.
    - ``bbox`` / ``bbox70`` / ``bbox90``: the image with a box around the
      lesion set to black, so the lesion's outline is hidden; for ``bbox70``
      and ``bbox90`` the box is grown until it hides about 70% / 90% of the
      image.
    - ``<region>_low`` / ``<region>_high``: ``whole``, ``lesion`` or
      ``background`` after a Gaussian low-pass (sigma = 1) or high-pass
      (image minus its low-pass, shifted by 0.5 so it stays in ``[0, 1]``).

    Args:
        image: Float image of shape ``(H, W, 3)`` in ``[0, 1]``.
        mask: Boolean lesion mask of shape ``(H, W)``.
        view: One of :data:`VIEWS`.

    Returns:
        The view, shaped like ``image``.

    Raises:
        ValueError: If ``view`` is not one of :data:`VIEWS`.

    Examples:
        >>> img = np.ones((6, 6, 3), dtype=np.float32)
        >>> m = np.zeros((6, 6), dtype=bool)
        >>> m[2:4, 2:4] = True
        >>> float(make_view(img, m, "lesion").sum())
        12.0
    """
    if view not in VIEWS:
        raise ValueError(f"view must be one of {VIEWS}, got {view!r}.")
    if view in _BBOX_COVERAGE:
        hidden = bbox_mask(mask, _BBOX_COVERAGE[view])
        return image * (~hidden)[..., None]
    region, _, band = view.partition("_")
    if region == "lesion":
        image = image * mask[..., None]
    elif region == "background":
        image = image * (~mask)[..., None]
    if band == "low":
        return gaussian_blur(image)
    if band == "high":
        return np.clip(image - gaussian_blur(image) + 0.5, 0.0, 1.0)
    return image


# --------------------------------------------------------------------------
# Artifacts (hand-drawn controls for the paper's diffusion-inpainted ones)
# --------------------------------------------------------------------------
def _overlay_to_image(
    image: np.ndarray, draw_fn: Callable[[ImageDraw.ImageDraw], None]
) -> np.ndarray:
    """Alpha-composite what ``draw_fn`` draws on an RGBA layer onto ``image``.

    Called by every shape-drawing artifact (bubbles, ink, patches, ruler).

    Args:
        image: Float image of shape ``(H, W, 3)`` in ``[0, 1]``.
        draw_fn: Function that draws on a transparent RGBA PIL layer the size
            of ``image``.

    Returns:
        The blended image, float array of shape ``(H, W, 3)``.
    """
    h, w = image.shape[:2]
    layer = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    draw_fn(ImageDraw.Draw(layer))
    rgba = np.asarray(layer, dtype=np.float32) / 255.0
    alpha = rgba[..., 3:4]
    return image * (1 - alpha) + rgba[..., :3] * alpha


def _add_dark_corner(
    image: np.ndarray, mask: np.ndarray, rng: np.random.Generator
) -> np.ndarray:
    """Black out the four corners with a smooth circular vignette.

    Called by :func:`add_artifact`, which then restores the lesion pixels.

    Args:
        image: Float image of shape ``(H, W, 3)`` in ``[0, 1]``.
        mask: Boolean lesion mask of shape ``(H, W)``.
        rng: Random generator that sets the artifact's placement.

    Returns:
        The image with the artifact, float array of shape ``(H, W, 3)``.
    """
    h, w = image.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w]
    radius = np.hypot((yy - (h - 1) / 2) / (h / 2), (xx - (w - 1) / 2) / (w / 2))
    start = rng.uniform(1.0, 1.1)  # normalised radius where the vignette begins
    alpha = np.clip((radius - start) / 0.2, 0.0, 1.0)[..., None]
    return image * (1 - alpha)


def _add_gel_bubble(
    image: np.ndarray, mask: np.ndarray, rng: np.random.Generator
) -> np.ndarray:
    """Scatter translucent white bubbles with bright rims.

    Called by :func:`add_artifact`, which then restores the lesion pixels.

    Args:
        image: Float image of shape ``(H, W, 3)`` in ``[0, 1]``.
        mask: Boolean lesion mask of shape ``(H, W)``.
        rng: Random generator that sets the artifact's placement.

    Returns:
        The image with the artifact, float array of shape ``(H, W, 3)``.
    """
    h, w = image.shape[:2]
    short = min(h, w)

    def draw(d: ImageDraw.ImageDraw) -> None:
        """Draw the bubbles on the transparent layer ``d``."""
        for _ in range(int(rng.integers(8, 15))):
            r = rng.uniform(0.015, 0.04) * short
            x, y = rng.uniform(0, w), rng.uniform(0, h)
            d.ellipse(
                (x - r, y - r, x + r, y + r),
                fill=(255, 255, 255, 60),
                outline=(255, 255, 255, 190),
                width=max(1, int(r / 5)),
            )

    return _overlay_to_image(image, draw)


def _add_ink(
    image: np.ndarray, mask: np.ndarray, rng: np.random.Generator
) -> np.ndarray:
    """Draw a wobbly dark-blue marker ring around the lesion's bounding box.

    Called by :func:`add_artifact`, which then restores the lesion pixels.

    Args:
        image: Float image of shape ``(H, W, 3)`` in ``[0, 1]``.
        mask: Boolean lesion mask of shape ``(H, W)``.
        rng: Random generator that sets the artifact's placement.

    Returns:
        The image with the artifact, float array of shape ``(H, W, 3)``.
    """
    h, w = image.shape[:2]
    top, bottom, left, right = lesion_bbox(mask)
    centre = np.array([(left + right) / 2, (top + bottom) / 2])
    radii = np.array([(right - left) / 2, (bottom - top) / 2]) * 1.3 + 0.03 * min(h, w)
    angles = np.linspace(0, 2 * np.pi, 72)
    wobble = 1 + 0.04 * np.sin(3 * angles + rng.uniform(0, 6.28))
    points = [
        tuple(centre + radii * wobble[i] * np.array([np.cos(a), np.sin(a)]))
        for i, a in enumerate(angles)
    ]

    def draw(d: ImageDraw.ImageDraw) -> None:
        """Draw the ink ring on the transparent layer ``d``."""
        d.line(points, fill=(30, 20, 120, 235), width=max(2, int(0.012 * min(h, w))))

    return _overlay_to_image(image, draw)


def _add_patches(
    image: np.ndarray, mask: np.ndarray, rng: np.random.Generator
) -> np.ndarray:
    """Stick two small solid coloured rectangles near the image edge.

    Called by :func:`add_artifact`, which then restores the lesion pixels.

    Args:
        image: Float image of shape ``(H, W, 3)`` in ``[0, 1]``.
        mask: Boolean lesion mask of shape ``(H, W)``.
        rng: Random generator that sets the artifact's placement.

    Returns:
        The image with the artifact, float array of shape ``(H, W, 3)``.
    """
    h, w = image.shape[:2]
    colours = [(220, 40, 40), (40, 90, 220), (240, 200, 40), (40, 170, 90)]

    def draw(d: ImageDraw.ImageDraw) -> None:
        """Draw the patches on the transparent layer ``d``."""
        for _ in range(2):
            pw, ph = rng.uniform(0.05, 0.1) * w, rng.uniform(0.05, 0.1) * h
            x = rng.choice([rng.uniform(0, 0.15 * w), rng.uniform(0.75 * w, w - pw)])
            y = rng.uniform(0, h - ph)
            colour = colours[int(rng.integers(len(colours)))]
            d.rectangle((x, y, x + pw, y + ph), fill=(*colour, 255))

    return _overlay_to_image(image, draw)


def _add_ruler(
    image: np.ndarray, mask: np.ndarray, rng: np.random.Generator
) -> np.ndarray:
    """Lay a pale ruler with black tick marks along the bottom edge.

    Called by :func:`add_artifact`, which then restores the lesion pixels.

    Args:
        image: Float image of shape ``(H, W, 3)`` in ``[0, 1]``.
        mask: Boolean lesion mask of shape ``(H, W)``.
        rng: Random generator that sets the artifact's placement.

    Returns:
        The image with the artifact, float array of shape ``(H, W, 3)``.
    """
    h, w = image.shape[:2]
    band = max(4, int(0.07 * h))
    step = max(3, int(0.03 * w))

    def draw(d: ImageDraw.ImageDraw) -> None:
        """Draw the ruler band and its ticks on the transparent layer ``d``."""
        d.rectangle((0, h - band, w, h), fill=(235, 235, 225, 235))
        for i, x in enumerate(range(0, w, step)):
            length = band * (0.6 if i % 5 == 0 else 0.3)
            d.line((x, h - band, x, h - band + length), fill=(0, 0, 0, 255), width=1)

    return _overlay_to_image(image, draw)


_ARTIFACT_FUNCTIONS = {
    "dark_corner": _add_dark_corner,
    "gel_bubble": _add_gel_bubble,
    "ink": _add_ink,
    "patches": _add_patches,
    "ruler": _add_ruler,
}


def add_artifact(
    image: np.ndarray, mask: np.ndarray, artifact: str, seed: int = 0
) -> np.ndarray:
    """Draw one synthetic artifact on an image, leaving the lesion untouched.

    The artifact is drawn on the whole image and then every lesion pixel is
    restored from the original, so the lesion is bit-identical before and
    after. (Artifacts that would cover the lesion are clipped at its border.)
    This is the "non-generative control" for the paper's diffusion-inpainted
    artifacts: it is deterministic given ``seed`` and cannot alter the lesion.

    Args:
        image: Float image of shape ``(H, W, 3)`` in ``[0, 1]``.
        mask: Boolean lesion mask of shape ``(H, W)``.
        artifact: One of :data:`ARTIFACTS`.
        seed: Seed for the artifact's random placement.

    Returns:
        The image with the artifact, shaped like ``image``.

    Raises:
        ValueError: If ``artifact`` is not one of :data:`ARTIFACTS`.

    Examples:
        >>> img = np.full((32, 32, 3), 0.5, dtype=np.float32)
        >>> m = np.zeros((32, 32), dtype=bool)
        >>> m[12:20, 12:20] = True
        >>> out = add_artifact(img, m, "ruler")
        >>> bool(np.array_equal(out[m], img[m]))
        True
    """
    if artifact not in _ARTIFACT_FUNCTIONS:
        raise ValueError(f"artifact must be one of {ARTIFACTS}, got {artifact!r}.")
    rng = np.random.default_rng(seed)
    out = _ARTIFACT_FUNCTIONS[artifact](image, mask, rng).astype(image.dtype)
    out = np.clip(out, 0.0, 1.0)
    out[mask] = image[mask]
    return out
