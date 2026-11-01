"""Explicit endpoint-affine box interchange (not an enclosing eight-corner AABB)."""

from __future__ import annotations

from typing import cast

import numpy as np
from numpy.typing import NDArray

from lung_nodule_mlx._typing import RealArray


def boxes_to_world(
    boxes: NDArray[np.float32],
    affine: RealArray,
) -> NDArray[np.float32]:
    """MONAI AffineBoxToWorldCoordinated followed by ConvertBoxModed(cccwhd).

    ``boxes`` has shape (N, 6), with voxel columns (x0, y0, z0, x1, y1, z1).
    Return (cx, cy, cz, width, height, depth) in the affine's world units.
    Both corners and the homogeneous affine are multiplied in float32. Centers
    use (lower + upper) / 2, as CenterSizeMode does. Pass the prepared image's
    affine (RAS, mm); for the bundle's patient-LPS output, negate its first two
    rows first. Only the two opposite corners are transformed, so an oblique or
    sheared affine gives a different box than the enclosing box of all eight corners.
    """
    matrix = np.asarray(affine, dtype=np.float32)
    ones = np.ones((len(boxes), 1), dtype=np.float32)
    lower = (matrix @ np.concatenate((boxes[:, :3], ones), axis=1).T).T[:, :3]
    upper = (matrix @ np.concatenate((boxes[:, 3:], ones), axis=1).T).T[:, :3]
    # A flipping affine can swap which transformed corner is lower; keep sizes non-negative.
    lo, hi = np.minimum(lower, upper), np.maximum(lower, upper)
    return cast(NDArray[np.float32], np.concatenate(((lo + hi) / np.float32(2), hi - lo), axis=1))
