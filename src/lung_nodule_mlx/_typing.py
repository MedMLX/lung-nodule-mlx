"""Array, NIfTI and serialized detection contracts at dependency boundaries."""

from __future__ import annotations

from pathlib import Path
from typing import NotRequired, Protocol, TypedDict

import numpy as np
from medmlx_core.outputs import RunResult as _RunResult
from numpy.typing import ArrayLike, NDArray

type FloatArray = NDArray[np.float32]
type DoubleArray = NDArray[np.float64]
type RealArray = NDArray[np.bool_ | np.integer | np.floating]
type Maps = dict[str, list[FloatArray]]


class NiftiHeader(Protocol):
    binaryblock: bytes

    def get_xyzt_units(self) -> tuple[str, str]:
        """Return the spatial and temporal units stored in the header."""
        ...
    def set_xyzt_units(self, xyz: str | int | None = None, t: str | int | None = None) -> None:
        """Set the spatial and temporal unit codes in the header."""
        ...


class NiftiImage(Protocol):
    shape: tuple[int, ...]
    dataobj: ArrayLike
    affine: DoubleArray
    header: NiftiHeader


class NiftiIo(Protocol):
    def load(self, path: str | Path) -> NiftiImage:
        """Load a NIfTI image from a path."""
        ...
    def as_closest_canonical(self, image: NiftiImage) -> NiftiImage:
        """Reorient an image to the closest canonical axes."""
        ...


class BundleRow(TypedDict):
    """One image's row of the MONAI bundle output, with parallel per-box lists.

    ``box`` holds ``(cx, cy, cz, w, h, d)`` in mm and patient LPS.
    """

    image: str
    box: list[list[float]]
    label: list[int]
    label_scores: list[float]


class Detection(TypedDict):
    """One detection: ``box`` is ``(cx, cy, cz, w, h, d)`` in mm and NIfTI RAS."""

    box: list[float]
    label: int
    score: float
    source_image: str


class DetectionRecord(TypedDict):
    """The ``medmlx.detections.v1`` JSON record written to ``detections.json``."""

    schema_version: str
    model_id: str
    source_format: str
    coordinate_space: str
    coordinate_convention: str
    source_coordinate_space: str
    units: str
    box_mode: str
    classes: dict[str, str]
    source_output: str
    detections: list[Detection]


class RunDetails(TypedDict):
    """Run facts returned by ``run``.

    ``input_shape`` is the prepared NCDHW shape and ``padded_shape`` the spatial
    shape after padding to multiples of ``(16, 16, 8)``. ``prepared_affine`` is
    the 4x4 RAS affine in mm. ``max_score`` is None when nothing was detected.
    """

    weights_revision: NotRequired[str | None]
    detection_count: int
    max_score: float | None
    coordinate_convention: str
    box_mode: str
    units: str
    classes: dict[str, str]
    input_shape: list[int]
    resample_raw_input: bool
    prepared_affine: list[list[float]]
    padded_shape: list[int]
    dtype: str
    device: str
    peak_memory_bytes: int | None


class RunOutputs(TypedDict):
    """Absolute POSIX paths of the files ``run`` writes."""

    detections: str
    bundle_detections: str


# Return value of ``lung_nodule_mlx.run``: detection files by role, with artifacts.
type RunResult = _RunResult[RunDetails, RunOutputs]
