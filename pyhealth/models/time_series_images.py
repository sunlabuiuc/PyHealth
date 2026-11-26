"""Images of irregular multivariate time series, rendered on the fly.

Each variable gets one square cell of a grid, so a sample with 36 variables
becomes a 6 x 6 grid image, as in ViTST (Li et al., NeurIPS 2023, "Time
Series as Images: Vision Transformer for Irregularly Sampled Time Series").
The cells hold one of:

- ``"line"``: the variable's line graph, observed points joined by
  straight lines and marked, in a colour per variable (ViTST);
- ``"spectrum"``: the line graph of the magnitude spectrum of the variable
  resampled onto an even grid;
- ``"gasf"`` / ``"gadf"``: Gramian angular summation / difference fields
  (Wang and Oates, IJCAI 2015);
- ``"mtf"``: the Markov transition field (Wang and Oates, 2015);
- ``"recurrence"``: the unthresholded recurrence plot (Hatami et al.,
  ICMV 2017).

All are written in torch and batched, so they run on the model's device
inside the forward pass and need no image files. This is a reimplementation
from the papers' descriptions; ViTST draws its line graphs with matplotlib.
"""

from __future__ import annotations

import math

import torch

IMAGE_TYPES = ("line", "spectrum", "gasf", "gadf", "mtf", "recurrence")

_PALETTE = torch.tensor([
    (0.12, 0.47, 0.71), (1.00, 0.50, 0.05), (0.17, 0.63, 0.17),
    (0.84, 0.15, 0.16), (0.58, 0.40, 0.74), (0.55, 0.34, 0.29),
    (0.89, 0.47, 0.76), (0.50, 0.50, 0.50), (0.74, 0.74, 0.13),
    (0.09, 0.75, 0.81), (0.00, 0.00, 0.50), (0.50, 0.00, 0.00),
])


def variable_colors(num_variables: int) -> torch.Tensor:
    """A fixed RGB colour per variable, ``[num_variables, 3]`` in [0, 1].

    Cycles through 12 distinct colours, darkened on each further cycle.

    Examples:
        >>> variable_colors(14).shape
        torch.Size([14, 3])
    """
    index = torch.arange(num_variables)
    shade = 0.75 ** (index // len(_PALETTE)).float()
    return _PALETTE[index % len(_PALETTE)] * shade.unsqueeze(1)


def grid_shape(num_variables: int) -> tuple[int, int]:
    """Rows and columns of the most nearly square grid of cells.

    Examples:
        >>> grid_shape(36), grid_shape(34), grid_shape(17)
        ((6, 6), (6, 6), (4, 5))
    """
    cols = math.ceil(math.sqrt(num_variables))
    return math.ceil(num_variables / cols), cols


def _tile(cells: torch.Tensor, rows: int, cols: int, fill: float) -> torch.Tensor:
    """``[B, V, C, s, s]`` cells to ``[B, C, rows * s, cols * s]`` grid images."""
    batch, n_vars, channels, size, _ = cells.shape
    pad = rows * cols - n_vars
    if pad:
        cells = torch.cat(
            [cells, cells.new_full((batch, pad, channels, size, size), fill)], dim=1
        )
    grid = cells.view(batch, rows, cols, channels, size, size)
    return grid.permute(0, 3, 1, 4, 2, 5).reshape(
        batch, channels, rows * size, cols * size
    )


def line_graph_cells(
    values: torch.Tensor,
    times: torch.Tensor,
    mask: torch.Tensor,
    cell_size: int,
    time_range: tuple[float, float],
    value_range: tuple[float, float],
    colors: torch.Tensor,
    marker_radius: int = 1,
) -> torch.Tensor:
    """Line graphs of each variable, ``[B, V, 3, cell_size, cell_size]``.

    Consecutive observed points of a variable are joined by a straight line
    one pixel wide and each point is marked with a square of side
    ``2 * marker_radius + 1``, on a white background. Values are clipped to
    ``value_range``.

    Args:
        values: ``[B, L, V]``; only entries where ``mask`` is set are drawn.
        times: ``[B, L]`` (or ``[B, L, V]``), non-decreasing along ``L``.
        mask: ``[B, L, V]``, true where a value was observed.
        cell_size: Pixels per side of a cell.
        time_range: Times mapped to the left and right edges.
        value_range: Values mapped to the bottom and top edges.
        colors: ``[V, 3]`` line colours.
        marker_radius: Half the marker side, in pixels (0 for one pixel).

    Examples:
        >>> import torch
        >>> x, t = torch.randn(2, 5, 3), torch.arange(5.0).expand(2, 5)
        >>> line_graph_cells(x, t, torch.ones(2, 5, 3), 16, (0, 4), (-3, 3),
        ...                  variable_colors(3)).shape
        torch.Size([2, 3, 3, 16, 16])
    """
    batch, length, n_vars = values.shape
    device = values.device
    if times.dim() == 2:
        times = times.unsqueeze(-1).expand(batch, length, n_vars)
    mask = mask.bool() & torch.isfinite(times)
    span = cell_size - 1
    t0, t1 = time_range
    v0, v1 = value_range
    px = ((times - t0) / (t1 - t0)).clamp(0, 1) * span
    py = (1 - ((values - v0) / (v1 - v0)).clamp(0, 1)) * span
    px = torch.nan_to_num(px)
    py = torch.nan_to_num(py)

    # The next observed point of the same variable, for each observed point.
    steps = torch.arange(length, device=device).view(1, length, 1)
    marked = torch.where(mask, steps, length)
    next_obs = torch.flip(torch.cummin(torch.flip(marked, [1]), dim=1).values, [1])
    next_obs = torch.cat([next_obs[:, 1:], next_obs.new_full((batch, 1, n_vars), length)], 1)
    has_next = mask & (next_obs < length)
    gather = next_obs.clamp(max=length - 1)
    nx = torch.gather(px, 1, gather)
    ny = torch.gather(py, 1, gather)

    b_idx, l_idx, v_idx = torch.nonzero(has_next, as_tuple=True)
    x_a, y_a = px[b_idx, l_idx, v_idx], py[b_idx, l_idx, v_idx]
    x_b, y_b = nx[b_idx, l_idx, v_idx], ny[b_idx, l_idx, v_idx]
    frac = torch.linspace(0, 1, 2 * cell_size, device=device)
    seg_x = (x_a.unsqueeze(1) + (x_b - x_a).unsqueeze(1) * frac).round().long()
    seg_y = (y_a.unsqueeze(1) + (y_b - y_a).unsqueeze(1) * frac).round().long()
    seg_bv = (b_idx * n_vars + v_idx).unsqueeze(1).expand_as(seg_x)

    b_pt, l_pt, v_pt = torch.nonzero(mask, as_tuple=True)
    offsets = torch.arange(-marker_radius, marker_radius + 1, device=device)
    dy, dx = torch.meshgrid(offsets, offsets, indexing="ij")
    mx = (px[b_pt, l_pt, v_pt].round().long().unsqueeze(1) + dx.reshape(1, -1))
    my = (py[b_pt, l_pt, v_pt].round().long().unsqueeze(1) + dy.reshape(1, -1))
    m_bv = (b_pt * n_vars + v_pt).unsqueeze(1).expand_as(mx)

    xs = torch.cat([seg_x.reshape(-1), mx.reshape(-1)]).clamp(0, span)
    ys = torch.cat([seg_y.reshape(-1), my.reshape(-1)]).clamp(0, span)
    bv = torch.cat([seg_bv.reshape(-1), m_bv.reshape(-1)])
    hit = torch.zeros(batch * n_vars, cell_size * cell_size, device=device)
    hit[bv, ys * cell_size + xs] = 1.0
    hit = hit.view(batch, n_vars, 1, cell_size, cell_size)
    color = colors.to(device).view(1, n_vars, 3, 1, 1)
    return 1.0 - hit * (1.0 - color)


def resample(
    values: torch.Tensor,
    times: torch.Tensor,
    mask: torch.Tensor,
    num_points: int,
    time_range: tuple[float, float],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Linear interpolation of each variable onto an even time grid.

    Before the first and after the last observation the nearest observed
    value is held. A variable never observed is all zeros.

    Args:
        values: ``[B, L, V]``.
        times: ``[B, L]``, non-decreasing along ``L`` where finite.
        mask: ``[B, L, V]``, true where a value was observed.
        num_points: Points of the grid.
        time_range: First and last grid times.

    Returns:
        ``(series, observed)``: the series ``[B, V, num_points]`` and
        ``[B, V]``, true for the variables with any observation.

    Examples:
        >>> import torch
        >>> x = torch.tensor([[[0.0], [2.0]]])
        >>> series, _ = resample(x, torch.tensor([[1.0, 3.0]]),
        ...                      torch.ones(1, 2, 1), 5, (0.0, 4.0))
        >>> series[0, 0]
        tensor([0., 0., 1., 2., 2.])
    """
    batch, length, n_vars = values.shape
    device = values.device
    mask = mask.bool() & torch.isfinite(times).unsqueeze(-1)
    grid = torch.linspace(*time_range, num_points, device=device)
    t = torch.nan_to_num(times, nan=float("inf")).unsqueeze(-1).expand(batch, length, n_vars)
    t = torch.where(mask, t, torch.full_like(t, float("inf")))
    t_sorted, order = torch.sort(t.transpose(1, 2), dim=-1)  # [B, V, L]
    v_sorted = torch.gather(torch.nan_to_num(values).transpose(1, 2), -1, order)
    count = mask.sum(1)  # [B, V]
    q = grid.view(1, 1, -1).expand(batch, n_vars, -1).contiguous()
    right = torch.searchsorted(t_sorted.contiguous(), q)  # first time >= q
    last = (count - 1).clamp(min=0).unsqueeze(-1)
    hi = torch.minimum(right, last)
    lo = torch.minimum((right - 1).clamp(min=0), last)
    t_lo, t_hi = torch.gather(t_sorted, -1, lo), torch.gather(t_sorted, -1, hi)
    v_lo, v_hi = torch.gather(v_sorted, -1, lo), torch.gather(v_sorted, -1, hi)
    gap = t_hi - t_lo
    w = torch.where(gap > 0, (q - t_lo) / torch.where(gap > 0, gap, 1.0), 0.0).clamp(0, 1)
    series = v_lo + w * (v_hi - v_lo)
    observed = count > 0
    return torch.where(observed.unsqueeze(-1), series, 0.0), observed


def field_cells(
    series: torch.Tensor,
    kind: str,
    value_range: tuple[float, float],
    num_bins: int = 8,
) -> torch.Tensor:
    """Two-dimensional fields of evenly sampled series, ``[B, V, 1, n, n]``.

    Values are first rescaled from ``value_range`` to [-1, 1] (clipped).
    The fields are in [0, 1].

    Args:
        series: ``[B, V, n]``.
        kind: ``"gasf"``, ``"gadf"``, ``"mtf"`` or ``"recurrence"``.
        value_range: Values mapped to -1 and 1.
        num_bins: Value bins of the Markov transition field, of equal width
            over ``value_range``.

    Examples:
        >>> import torch
        >>> field_cells(torch.randn(2, 3, 16), "gasf", (-3, 3)).shape
        torch.Size([2, 3, 1, 16, 16])
    """
    v0, v1 = value_range
    x = (2 * (series - v0) / (v1 - v0) - 1).clamp(-1, 1)
    xi, xj = x.unsqueeze(-1), x.unsqueeze(-2)
    root = torch.sqrt((1 - x * x).clamp(min=0))
    ri, rj = root.unsqueeze(-1), root.unsqueeze(-2)
    if kind == "gasf":  # cos(phi_i + phi_j)
        field = (xi * xj - ri * rj + 1) / 2
    elif kind == "gadf":  # sin(phi_i - phi_j)
        field = (ri * xj - xi * rj + 1) / 2
    elif kind == "recurrence":
        field = (xi - xj).abs() / 2
    elif kind == "mtf":
        bins = ((x + 1) / 2 * num_bins).long().clamp(max=num_bins - 1)
        onehot = torch.nn.functional.one_hot(bins, num_bins).float()  # [B, V, n, Q]
        counts = onehot[..., :-1, :].transpose(-1, -2) @ onehot[..., 1:, :]
        transition = counts / counts.sum(-1, keepdim=True).clamp(min=1)
        field = onehot @ transition @ onehot.transpose(-1, -2)
    else:
        raise ValueError(f"Unknown field {kind!r}")
    return field.unsqueeze(2)


def render_images(
    values: torch.Tensor,
    times: torch.Tensor,
    mask: torch.Tensor,
    image_type: str = "line",
    cell_size: int = 64,
    time_range: tuple[float, float] = (0.0, 48.0),
    value_range: tuple[float, float] = (-3.0, 3.0),
    colors: torch.Tensor | None = None,
) -> torch.Tensor:
    """RGB grid images of a batch of irregular series, ``[B, 3, H, W]`` in [0, 1].

    Each variable is one ``cell_size`` square of the most nearly square grid
    (:func:`grid_shape`); unused cells are white.

    Args:
        values: ``[B, L, V]``, ideally z-scored.
        times: ``[B, L]``; non-finite times are padding.
        mask: ``[B, L, V]``, true where a value was observed.
        image_type: One of :data:`IMAGE_TYPES`.
        cell_size: Pixels per cell side.
        time_range: The time axis of the line graphs and resampling.
        value_range: The value axis; values outside are clipped.
        colors: ``[V, 3]`` line colours; :func:`variable_colors` by default.

    Examples:
        >>> import torch
        >>> x = torch.randn(2, 10, 5)
        >>> t = torch.linspace(0, 48, 10).expand(2, 10)
        >>> render_images(x, t, torch.rand(2, 10, 5) > 0.5, cell_size=16).shape
        torch.Size([2, 3, 32, 48])
    """
    n_vars = values.shape[-1]
    rows, cols = grid_shape(n_vars)
    if colors is None:
        colors = variable_colors(n_vars)
    if image_type == "line":
        cells = line_graph_cells(
            values, times, mask, cell_size, time_range, value_range, colors
        )
        return _tile(cells, rows, cols, 1.0)
    if image_type not in IMAGE_TYPES:
        raise ValueError(f"image_type must be one of {IMAGE_TYPES}, not {image_type!r}")
    series, observed = resample(values, times, mask, cell_size, time_range)
    if image_type == "spectrum":
        spectrum = torch.log1p(torch.fft.rfft(series, dim=-1).abs())  # [B, V, n//2+1]
        bins = spectrum.shape[-1]
        freq = torch.arange(bins, device=values.device, dtype=values.dtype)
        freq = freq.view(1, bins).expand(values.shape[0], bins)
        top = math.log1p(cell_size * max(abs(value_range[0]), abs(value_range[1])))
        shown = observed.unsqueeze(1).expand(-1, bins, -1)
        cells = line_graph_cells(
            spectrum.transpose(1, 2), freq, shown, cell_size,
            (0.0, float(bins - 1)), (0.0, top), colors, marker_radius=0,
        )
        return _tile(cells, rows, cols, 1.0)
    cells = field_cells(series, image_type, value_range)
    cells = torch.where(observed.view(*observed.shape, 1, 1, 1), cells, 0.0)
    return _tile(cells.expand(-1, -1, 3, -1, -1), rows, cols, 0.0)
