"""Native NumPy port of the bundle's MONAI Spacing (bilinear, border) transform.

Adapted from MONAI 1.6.0, Copyright MONAI Consortium, Apache-2.0; see root THIRD_PARTY_NOTICES.md.
Only the pinned bundle's 3D, diagonal=False, align_corners=False configuration
is implemented. Computation is float64 and the returned voxels are float32.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import numpy as np
from medmlx_core.errors import InvalidInputError
from numpy.typing import NDArray

BUNDLE_SPACING = (0.703125, 0.703125, 1.25)
_AFFINE_TOL = 1e-3  # monai.data.utils.AFFINE_TOL
_CHUNK_VOXELS = 1 << 18


@dataclass(frozen=True, slots=True)
class SpacingGrid:
    """The output voxel grid and its pull transform into the source grid.

    ``shape`` is the output voxel shape, ``affine`` the 4x4 output voxel-to-world
    affine (mm), and ``pull`` the 4x4 map from output voxel indices to source
    voxel indices (identity when no resampling is needed).
    """

    shape: tuple[int, int, int]
    affine: NDArray[np.float64]
    pull: NDArray[np.float64]


def _spacing_grid(shape: tuple[int, ...], affine: NDArray[np.float64]) -> SpacingGrid:
    """MONAI zoom_affine and compute_shape_offset, including shear removal."""
    linear = affine[:3, :3]
    original = np.sqrt(np.sum(linear * linear, axis=0))
    target = np.asarray(BUNDLE_SPACING, dtype=np.float64)
    keep = (target - _AFFINE_TOL <= original) & (original <= target + _AFFINE_TOL)
    target = np.where(keep, original, target)
    try:
        zoom_shear = np.linalg.cholesky(linear.T @ linear).T
        destination = np.eye(4, dtype=np.float64)
        destination[:3, :3] = linear @ np.linalg.inv(zoom_shear) @ np.diag(target)
        corners = np.asarray(np.meshgrid(*[(0.0, n - 1.0) for n in shape], indexing="ij")).reshape(
            3, -1
        )
        corners = np.concatenate((corners, np.ones_like(corners[:1])))
        output_corners = np.linalg.solve(destination, affine) @ corners
        world_corners = affine @ corners
        output_shape = np.round(np.ptp(output_corners[:3], axis=1) + 1.0).astype(int)
        for index in range(corners.shape[1]):
            distance = np.min(output_corners[:3] - output_corners[:3, index : index + 1], axis=1)
            if np.allclose(distance, 0.0, rtol=_AFFINE_TOL):
                destination[:3, 3] = world_corners[:3, index]
                break
        else:
            destination[:3, 3] = (
                linear @ (np.asarray(shape) / 2.0)
                + affine[:3, 3]
                - destination[:3, :3] @ (output_shape / 2.0)
            )
        pull = np.linalg.solve(affine, destination)
    except np.linalg.LinAlgError as error:
        raise InvalidInputError("Lung resampling requires an invertible image affine") from error
    unchanged = tuple(output_shape) == shape and (
        np.allclose(affine, destination, atol=_AFFINE_TOL, rtol=1e-5)
        or np.allclose(pull, np.eye(4), atol=_AFFINE_TOL, rtol=1e-5)
    )
    if unchanged:
        return SpacingGrid(cast(tuple[int, int, int], tuple(shape)), affine.copy(), np.eye(4))
    # MetaTensor records source @ pull, rather than copying the requested affine.
    return SpacingGrid(
        cast(tuple[int, int, int], tuple(int(n) for n in output_shape)), affine @ pull, pull
    )


def require_bundle_grid(shape: tuple[int, ...], affine: NDArray[np.float64]) -> None:
    """Reject an image unless the bundle's ``Spacingd`` would leave it unchanged.

    This is the pre-resampled contract: every axis within MONAI's ``AFFINE_TOL``
    (1e-3 mm) of the bundle spacing and no shear, so the default path sees the
    same voxels and affine as ``resample_raw_input=True``. Rotation is allowed,
    as ``Spacingd`` keeps it.

    Raises:
        InvalidInputError: The image is off the bundle grid or its affine is
            singular.
    """
    if not np.array_equal(_spacing_grid(shape, affine).pull, np.eye(4)):
        linear = affine[:3, :3]
        spacing = ", ".join(f"{v:.6g}" for v in np.sqrt(np.sum(linear * linear, axis=0)))
        raise InvalidInputError(
            f"Lung MLX expects the bundle's 0.703125 x 0.703125 x 1.25 mm grid "
            f"(within 1e-3 mm per axis, no shear); this image has ({spacing}) mm "
            "voxel axes. Pass resample_raw_input=True to resample it."
        )


def _trilinear(
    data: NDArray[np.float32],
    coordinates: NDArray[np.float64],
) -> NDArray[np.float32]:
    """grid_sample's border clamping and eight-corner weight/sum order."""
    coordinates = np.clip(coordinates, 0.0, np.asarray(data.shape)[:, None] - 1.0)
    lower = np.floor(coordinates).astype(np.int64)
    upper = np.minimum(lower + 1, np.asarray(data.shape)[:, None] - 1)
    fraction = coordinates - lower
    result = np.zeros(coordinates.shape[1], dtype=np.float64)
    # ATen's (tnw, tne, tsw, tse, bnw, bne, bsw, bse) order.
    for i, j, k in np.ndindex(2, 2, 2):
        indices = [upper[axis] if side else lower[axis] for axis, side in enumerate((i, j, k))]
        weights = [
            fraction[axis] if side else 1.0 - fraction[axis] for axis, side in enumerate((i, j, k))
        ]
        weight = weights[2] * weights[1] * weights[0]
        result += data[tuple(indices)].astype(np.float64) * weight
    return result.astype(np.float32)


def resample_image(
    data: NDArray[np.float32],
    affine: NDArray[np.float64],
) -> tuple[NDArray[np.float32], NDArray[np.float64]]:
    """Resample scalar 3D HU voxels to the bundle grid, before intensity scaling.

    Coordinates are voxel centers: source = solve(input_affine, output_affine)
    @ output_index. This is MONAI's normalized affine_grid/grid_sample mapping
    with align_corners=False; out-of-bounds centers use border values. Temporary
    float64 coordinates and gathered corners are bounded by chunk size.

    Args:
        data: Finite float32 3D HU voxels, indexed by the affine's voxel axes.
        affine: Float64 4x4 voxel-to-world affine in mm.

    Returns:
        ``(voxels, affine)``: the voxels on the bundle grid (0.703125 x 0.703125 x
        1.25 mm, axes within 1e-3 mm of that kept as they are) and its affine. If
        no resampling is needed, the input voxels and a copy of the affine.

    Raises:
        InvalidInputError: The data is not a finite float32 3D array, or the
            affine is not a finite, invertible, homogeneous 4x4 matrix.
    """
    if data.ndim != 3 or any(n <= 0 for n in data.shape):
        raise InvalidInputError("Lung resampling requires a scalar 3D image")
    if affine.shape != (4, 4) or not np.isfinite(affine).all():
        raise InvalidInputError("Lung resampling requires a finite 4x4 image affine")
    if data.dtype != np.float32 or not np.isfinite(data).all():
        raise InvalidInputError("Lung resampling requires finite float32 image data")
    if not np.array_equal(affine[3], [0, 0, 0, 1]):
        raise InvalidInputError("Lung resampling requires a homogeneous image affine")
    grid = _spacing_grid(data.shape, affine)
    if np.array_equal(grid.pull, np.eye(4)):
        return np.ascontiguousarray(data), grid.affine
    result = np.empty(grid.shape, dtype=np.float32)
    flat = result.reshape(-1)
    for start in range(0, flat.size, _CHUNK_VOXELS):
        stop = min(flat.size, start + _CHUNK_VOXELS)
        indices = np.asarray(np.unravel_index(np.arange(start, stop), grid.shape), np.float64)
        coordinates = grid.pull[:3, :3] @ indices + grid.pull[:3, 3, None]
        flat[start:stop] = _trilinear(data, coordinates)
    return result, grid.affine
