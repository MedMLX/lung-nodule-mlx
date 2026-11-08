"""Source-matched anchors, decoding, selection and window-map combination.

Host postprocessing mirrors MONAI's CPU sliding-window accumulator and box
selector. The supplied network executes the learned graph on its configured device.

Conventions: volumes are ``(1, 1, X, Y, Z)`` float32 with the spatial axes in the
order of the RAS-oriented array, already scaled to [0, 1] from HU. Boxes are
corner boxes ``(x0, y0, z0, x1, y1, z1)`` in voxel units of that array, with
``x`` along array axis 0. Head maps are per feature-pyramid level, finest first,
as float32 ``(1, C, x, y, z)`` with ``C = 3`` classification logits (one per
anchor) or ``C = 18`` box deltas (six per anchor).
"""

from __future__ import annotations

import itertools
import math
from collections.abc import Callable

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float32]
Maps = dict[str, list[FloatArray]]
PATCH = (512, 512, 192)
DIVISIBLE = (16, 16, 8)


def numpy_topk(scores: FloatArray, k: int) -> NDArray[np.int64]:
    """Return the indices of the largest finite scores in descending order.

    Returns the ``k`` largest-score indices, sorted by descending score with
    ties broken by ascending original index (a stable sort). ``select_boxes``
    re-sorts its concatenated candidates by score with its own stable sort, so
    when exactly tied scores are kept, their relative order here still affects
    which box is tested first by NMS; a plain unordered top-k set is not enough.

    PyTorch does not guarantee the order of ties in top-k. Float32 scores can
    tie, and choosing a different tied box first can change which overlapping
    boxes survive NMS compared with a recorded MONAI run.

    Raises:
        ValueError: ``scores`` has a non-finite value, or ``k`` is outside
            ``0 <= k <= len(scores)``.
    """
    values = np.ascontiguousarray(scores, dtype=np.float32)
    if not np.isfinite(values).all() or not 0 <= k <= len(values):
        raise ValueError("top-k requires finite scores and 0 <= k <= count")
    return np.argsort(-values, kind="stable")[:k].astype(np.int64)


def anchors(shape: tuple[int, ...], grid: tuple[int, ...], level: int) -> FloatArray:
    """Anchor boxes for one feature-pyramid level, in voxels of the padded image.

    Args:
        shape: Padded image spatial shape ``(X, Y, Z)``.
        grid: Spatial shape of this level's head maps; the stride per axis is
            ``shape // grid``.
        level: Pyramid level index; the base anchor shapes ``(6, 8, 4)``,
            ``(8, 6, 5)`` and ``(10, 10, 6)`` voxels (from the bundle config) are
            scaled by ``2 ** level``.

    Returns:
        Float32 ``(prod(grid) * 3, 6)`` corner boxes. Rows run over grid
        positions in C order, then over the three anchor shapes, matching the
        channel grouping of the head outputs. Centers sit at ``index * stride``,
        as in MONAI 1.4.0, which the bundle pins and was trained with. MONAI 1.6
        adds ``stride // 2``, which moves every decoded box of this bundle by half
        a stride (1.4 to 2.8 mm on its grid).
    """
    step = [a // b for a, b in zip(shape, grid, strict=True)]
    shift = np.stack(
        np.meshgrid(
            *(np.arange(n, dtype=np.float32) * s for n, s in zip(grid, step, strict=True)),
            indexing="ij",
        ),
        -1,
    )
    sizes = np.array([[6, 8, 4], [8, 6, 5], [10, 10, 6]], dtype=np.float32) * (2**level)
    # AnchorGeneratorWithAnchorShape rounds half-sizes, including 2.5 -> 2.
    base = np.round(np.concatenate((-sizes, sizes), axis=1) / 2)
    return np.asarray(
        (np.concatenate((shift, shift), -1)[..., None, :] + base).reshape(-1, 6), dtype=np.float32
    )


def decode(regression: FloatArray, anchor: FloatArray) -> FloatArray:
    """Apply box deltas to anchors.

    Args:
        regression: Float32 ``(N, 6)`` deltas ``(dx, dy, dz, dw, dh, dd)``.
        anchor: Float32 ``(N, 6)`` corner anchors from :func:`anchors`.

    Returns:
        Float32 ``(N, 6)`` corner boxes. Each center moves by its delta times the
        anchor size, and each extent is the anchor size times ``exp(delta)``, with
        the size delta capped at ``log(1000 / 16)`` as MONAI's box coder does.

    Raises:
        ValueError: A decoded coordinate is not finite.
    """
    sizes = anchor[:, 3:] - anchor[:, :3]
    centers = (anchor[:, :3] + anchor[:, 3:]) / np.float32(2)
    centers = regression[:, :3] * sizes + centers
    extent = np.exp(np.minimum(regression[:, 3:], np.float32(math.log(1000 / 16)))) * sizes
    result = np.concatenate(
        (centers - extent * np.float32(0.5), centers + extent * np.float32(0.5)), axis=1
    )
    if not np.isfinite(result).all():
        raise ValueError("Non-finite decoded lung boxes")
    return np.asarray(result, dtype=np.float32)


def select_boxes(
    boxes: list[FloatArray], logits: list[FloatArray], shape: tuple[int, ...]
) -> tuple[FloatArray, FloatArray, NDArray[np.int64]]:
    """Score, threshold, per-level top-k, clip and NMS the decoded boxes.

    A candidate needs ``sigmoid(logit) > 0.02`` (strict). Each level keeps its
    ``min(1000, count)`` best candidates, then boxes thinner than one voxel on any
    axis after clipping are dropped. Greedy NMS then runs across all levels in
    descending score order, suppressing boxes with IoU above 0.22 and stopping at
    300 boxes.

    Args:
        boxes: Per pyramid level, float32 ``(N_l, 6)`` corner boxes in voxels.
        logits: Per level, float32 ``(N_l, 1)`` classification logits (one class).
        shape: Original, unpadded image spatial shape; boxes are clipped to
            ``[0, shape]`` on each axis.

    Returns:
        ``(boxes, scores, labels)``: float32 ``(K, 6)`` corner boxes in voxels,
        float32 ``(K,)`` sigmoid scores in descending order, and int64 ``(K,)``
        labels (all 0, the single nodule class), with ``K <= 300``.
    """
    candidates: list[FloatArray] = []
    confidences: list[FloatArray] = []
    for box, logit in zip(boxes, logits, strict=True):
        # Torch CPU sigmoid uses float32 arithmetic. Threshold is strictly greater.
        with np.errstate(over="ignore"):
            scores = np.float32(1) / (np.float32(1) + np.exp(-logit.reshape(-1)))
        keep = np.flatnonzero(scores > np.float32(0.02))
        keep = keep[numpy_topk(scores[keep], min(1000, len(keep)))]
        clipped = np.clip(box[keep], 0, np.array(shape * 2, dtype=np.float32))
        valid = np.all(clipped[:, 3:] >= clipped[:, :3] + 1, axis=1)
        candidates.append(clipped[valid])
        confidences.append(np.asarray(scores[keep][valid], dtype=np.float32))
    all_boxes, all_scores = np.concatenate(candidates), np.concatenate(confidences)
    remaining = np.argsort(-all_scores, kind="stable")
    picked: list[int] = []
    volumes = np.prod(all_boxes[:, 3:] - all_boxes[:, :3], axis=1)
    while len(remaining):
        i = int(remaining[0])
        picked.append(i)
        if len(picked) == 300:
            break
        others = remaining[1:]
        intersection = np.prod(
            np.maximum(
                0,
                np.minimum(all_boxes[others, 3:], all_boxes[i, 3:])
                - np.maximum(all_boxes[others, :3], all_boxes[i, :3]),
            ),
            axis=1,
        )
        iou = intersection / (
            volumes[others] + volumes[i] - intersection + np.finfo(np.float32).eps
        )
        remaining = others[iou <= np.float32(0.22)]
    return all_boxes[picked], all_scores[picked], np.zeros(len(picked), dtype=np.int64)


def detections(
    maps: Maps, padded_shape: tuple[int, ...], original_shape: tuple[int, ...]
) -> tuple[FloatArray, FloatArray, NDArray[np.int64]]:
    """Decode head maps into final detections.

    Args:
        maps: Head maps from :func:`predict_maps`; only batch index 0 is read.
        padded_shape: Spatial shape of the padded image the maps were computed on.
        original_shape: Spatial shape of the image before padding, used for clipping.

    Returns:
        The ``(boxes, scores, labels)`` of :func:`select_boxes`: corner boxes in
        voxels of the unpadded image, which shares its origin with the padded one.
    """
    boxes: list[FloatArray] = []
    logits: list[FloatArray] = []
    for level, (cls, reg) in enumerate(
        zip(maps["classification"], maps["box_regression"], strict=True)
    ):
        grid = tuple(cls.shape[2:])
        anchor = anchors(padded_shape, grid, level)
        boxes.append(decode(reg[0].transpose(1, 2, 3, 0).reshape(-1, 6), anchor))
        logits.append(cls[0].transpose(1, 2, 3, 0).reshape(-1, 1))
    return select_boxes(boxes, logits, original_shape)


def predict_maps(
    image: FloatArray,
    network: Callable[[FloatArray], Maps],
    *,
    roi: tuple[int, int, int] = PATCH,
    force_windows: bool = False,
) -> tuple[Maps, tuple[int, ...]]:
    """Run the network on a volume, windowing large inputs, and return head maps.

    The volume is zero-padded at the end of each axis to multiples of
    ``DIVISIBLE``; after intensity scaling, 0 is -1024 HU. A volume with fewer
    voxels than ``roi`` (compared by voxel count, as the bundle's inferer does)
    goes through the network in one pass. Otherwise it is centered in a volume at
    least ``roi`` large and covered by windows of size ``roi`` with 25% overlap.
    Window maps are summed with constant weights, divided by the number of
    windows covering each position, and cropped back to the padded extent.

    Args:
        image: Float32 ``(1, 1, X, Y, Z)`` volume scaled to [0, 1].
        network: Maps a float32 ``(1, 1, x, y, z)`` array whose spatial size is a
            multiple of ``DIVISIBLE`` to head maps.
        roi: Sliding-window size in voxels.
        force_windows: Use windows even when the volume is smaller than ``roi``.

    Returns:
        ``(maps, padded_shape)``, with ``padded_shape`` the spatial shape rounded
        up to multiples of ``DIVISIBLE``.
    """
    original = image.shape[2:]
    divisible = tuple(math.ceil(n / d) * d for n, d in zip(original, DIVISIBLE, strict=True))
    padded = np.pad(
        image, ((0, 0), (0, 0), *((0, n - o) for o, n in zip(original, divisible, strict=True)))
    )
    if not force_windows and math.prod(original) < math.prod(roi):
        return network(padded), divisible
    # Pad (centered) to at least the roi on every axis; the crop at the end undoes it.
    shape = tuple(max(n, r) for n, r in zip(divisible, roi, strict=True))
    before = tuple((n - d) // 2 for n, d in zip(shape, divisible, strict=True))
    padded = np.pad(
        padded,
        (
            (0, 0),
            (0, 0),
            *((b, n - d - b) for b, n, d in zip(before, shape, divisible, strict=True)),
        ),
    )
    starts: list[list[int]] = []
    for n, r in zip(shape, roi, strict=True):
        # 25% overlap; the last window is shifted back so it ends at the volume edge.
        step = max(int(r * 0.75), 1)
        starts.append([min(i * step, n - r) for i in range(math.ceil((n - r) / step) + 1)])
    combined: Maps = {"classification": [], "box_regression": []}
    counts: list[FloatArray] = []
    for index, start in enumerate(itertools.product(*starts)):
        crop = tuple(slice(s, s + r) for s, r in zip(start, roi, strict=True))
        maps = network(padded[(slice(None), slice(None), *crop)])
        for key in combined:
            for level, output in enumerate(maps[key]):
                scale = tuple(o / r for o, r in zip(output.shape[2:], roi, strict=True))
                full_shape = tuple(int(n * z) for n, z in zip(shape, scale, strict=True))
                region = tuple(
                    slice(int(s * z), int((s + r) * z))
                    for s, r, z in zip(start, roi, scale, strict=True)
                )
                if index == 0:
                    combined[key].append(np.zeros((1, output.shape[1], *full_shape), np.float32))
                    if key == "classification":
                        counts.append(np.zeros(full_shape, np.float32))
                combined[key][level][(slice(None), slice(None), *region)] += output
                if key == "classification":
                    counts[level][region] += 1
    for key, outputs in combined.items():
        for level, output in enumerate(outputs):
            output /= counts[level]
            scale = tuple(o / n for o, n in zip(output.shape[2:], shape, strict=True))
            crop = tuple(
                slice(round(b * z), round((b + d) * z))
                for b, d, z in zip(before, divisible, scale, strict=True)
            )
            combined[key][level] = output[(slice(None), slice(None), *crop)]
    return combined, divisible
