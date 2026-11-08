"""MedMLX runner for the pinned MONAI lung nodule CT detection bundle.

Research software: it runs MONAI's 3D RetinaNet nodule detector natively in MLX
and writes boxes in world millimeters. Its outputs are not a clinical finding.
"""

from __future__ import annotations

import json
import os
from collections.abc import Callable
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import TYPE_CHECKING, cast

import numpy as np
from medmlx_core.errors import AssetNotReadyError, InvalidInputError, ModelExecutionError
from medmlx_core.weights import weights_dir, weights_revision

from lung_nodule_mlx._typing import (
    ArrayRuntime,
    BundleRow,
    Detection,
    DetectionRecord,
    DoubleArray,
    FloatArray,
    NiftiIo,
    RunResult,
)

if TYPE_CHECKING:
    from mlx.core import array

MODEL = {
    "id": "lung-nodule",
    "label": "Lung nodule CT detection",
    "task": "detection",
    "run": "lung_nodule_mlx.medmlx:run",
    # Measured MLX peak for a 512x512x147 chest CT resampled to the model grid;
    # larger inputs need more.
    "peak_memory_gb": 10.8,
    "summary": "MONAI 3D RetinaNet lung nodule boxes and scores for chest CT (bundle 0.6.10).",
}

DETECTIONS = "detections.json"
BUNDLE_DETECTIONS = "bundle_detections.json"
BUNDLE_ID = "lung_nodule_ct_detection"
BUNDLE_VERSION = "0.6.10"
CLASSES = {"0": "nodule"}
_RAS_LPS_FLIP = (-1.0, -1.0, 1.0)
_WEIGHTS_HINT = (
    "Pass a model folder containing network.safetensors, or the checkpoint file, "
    "as weights_path."
)


def prepare_image(
    path: Path,
    *,
    resample_raw_input: bool = False,
) -> tuple[FloatArray, DoubleArray]:
    """Load a NIfTI CT and apply the pinned bundle's preprocessing.

    The steps are: load, reorient to the closest RAS axes (no resampling), convert
    spatial units to mm, optionally resample (the bundle's ``Spacingd``), and
    scale HU from [-1024, 300] linearly into [0, 1] with clipping. The bundle
    disables ``Spacingd`` by default (``whether_raw_luna16=false``) without
    checking spacing. The default here requires an image already on the
    bundle's 0.703125 x 0.703125 x 1.25 mm grid, meaning one ``Spacingd`` would
    leave unchanged (each axis within 1e-3 mm, no shear), and rejects any other.
    ``resample_raw_input=True`` enables the resampling, as the bundle's
    ``whether_raw_luna16=true`` does.

    Returns:
        Contiguous float32 ``(1, 1, X, Y, Z)`` voxels, with the spatial axes in
        the RAS-oriented array order, and the float64 4x4 voxel-to-world affine in
        mm (RAS) of those voxels.

    Raises:
        InvalidInputError: The file is not a readable real, finite scalar 3D NIfTI
            with an invertible affine, its spatial units are not mm, meter or micron,
            ``resample_raw_input`` is not a bool, the image cannot be resampled,
            or, without resampling, it is off the bundle grid.
    """

    import nibabel as nib
    from nibabel.filebasedimages import ImageFileError

    if not isinstance(cast(object, resample_raw_input), bool):
        raise InvalidInputError("resample_raw_input must be a boolean")
    try:
        image = cast(NiftiIo, nib).load(str(path))
    except (ImageFileError, OSError) as error:
        raise InvalidInputError(f"Lung MLX requires a readable NIfTI image: {path}") from error
    if len(image.shape) != 3 or any(n <= 0 for n in image.shape):
        raise InvalidInputError("Lung MLX requires a scalar 3D NIfTI image")
    input_affine = np.asarray(image.affine, dtype=np.float64)
    if not np.isfinite(input_affine).all():
        raise InvalidInputError("Lung MLX requires finite image data and affine")
    if np.linalg.matrix_rank(input_affine[:3, :3]) < 3:
        raise InvalidInputError("Lung MLX requires an invertible image affine")
    image = cast(NiftiIo, nib).as_closest_canonical(image)
    raw = np.asanyarray(image.dataobj)
    if np.iscomplexobj(raw):
        raise InvalidInputError("Lung MLX requires real-valued CT voxels")
    with np.errstate(over="ignore", invalid="ignore"):
        data = np.asarray(raw, dtype=np.float32)
    if not np.isfinite(data).all() or not np.isfinite(image.affine).all():
        raise InvalidInputError("Lung MLX requires finite image data and affine")
    try:
        unit = image.header.get_xyzt_units()[0]
    except KeyError as error:
        raise InvalidInputError("Lung NIfTI spatial units must be mm, meter or micron") from error
    millimeter_scale = {"mm": 1.0, "meter": 1000.0, "micron": 0.001}
    if unit not in millimeter_scale:
        raise InvalidInputError("Lung NIfTI spatial units must be mm, meter or micron")
    affine = np.array(image.affine, dtype=np.float64, copy=True)
    affine[:3] *= millimeter_scale[unit]
    from lung_nodule_mlx.spacing import require_bundle_grid, resample_image

    if resample_raw_input:
        data, affine = resample_image(data, affine)
    else:
        require_bundle_grid(data.shape, affine)
    data = np.clip((data - np.float32(-1024)) / np.float32(1324), 0, 1)
    return np.ascontiguousarray(data[None, None]), affine


def bundle_rows(source_image: str, world_lps: FloatArray, scores: FloatArray) -> list[BundleRow]:
    """MONAI bundle output rows: patient-LPS ``cccwhd`` boxes with class 0 scores.

    ``world_lps`` is a float32 ``(N, 6)`` array of ``(cx, cy, cz, w, h, d)`` in mm
    and ``scores`` the matching ``(N,)`` scores. The bundle writes one row per
    image, so the result has a single element with parallel per-box lists.
    """

    return [
        {
            "image": source_image,
            "box": world_lps.tolist(),
            "label": [0] * len(world_lps),
            "label_scores": scores.tolist(),
        }
    ]


def detection_record(rows: list[BundleRow], source_output: str) -> DetectionRecord:
    """Flatten bundle rows into the RAS ``medmlx.detections.v1`` record.

    Box centers move from the rows' patient LPS to NIfTI RAS by negating x and y;
    sizes are unchanged. ``source_output`` names the bundle-format file the
    detections came from.
    """

    detections: list[Detection] = []
    for row in rows:
        for box, label, score in zip(row["box"], row["label"], row["label_scores"], strict=True):
            center = np.asarray(box[:3], dtype=np.float64) * _RAS_LPS_FLIP
            detections.append(
                {
                    "box": [
                        float(center[0]),
                        float(center[1]),
                        float(center[2]),
                        float(box[3]),
                        float(box[4]),
                        float(box[5]),
                    ],
                    "label": int(label),
                    "score": float(score),
                    "source_image": row["image"],
                }
            )
    return {
        "schema_version": "medmlx.detections.v1",
        "model_id": BUNDLE_ID,
        "source_format": f"monai.{BUNDLE_ID}.v{BUNDLE_VERSION}",
        "coordinate_space": "image_physical",
        "coordinate_convention": "nifti_ras",
        "source_coordinate_space": "dicom_patient_lps",
        "units": "mm",
        "box_mode": "cccwhd",
        "classes": dict(CLASSES),
        "source_output": source_output,
        "detections": detections,
    }


def load_weights(path: Path, mx: ArrayRuntime) -> dict[str, array]:
    """Load converted safetensors and require the released graph's FP32 ODHWI tensors.

    Raises:
        AssetNotReadyError: The file is unreadable, its tensor names differ from
            the released graph (batch-norm counters are not expected), or a tensor
            has the wrong shape or is not float32.
    """

    from lung_nodule_mlx.mlx_model import required_shapes

    try:
        if path.suffix != ".safetensors":
            raise ValueError("The native checkpoint must use the .safetensors suffix")
        weights = mx.load(str(path), format="safetensors")
    except (OSError, RuntimeError, ValueError) as error:
        raise AssetNotReadyError(
            f"Lung detector weights are not a readable safetensors file: {path}",
            reason=f"unreadable {path}",
            hint=_WEIGHTS_HINT,
        ) from error
    expected = {
        name: shape if len(shape) != 5 else (shape[0], *shape[2:], shape[1])
        for name, shape in required_shapes().items()
        if not name.endswith("num_batches_tracked")
    }
    missing = sorted(expected.keys() - weights.keys())
    unexpected = sorted(weights.keys() - expected.keys())
    if missing or unexpected:
        raise AssetNotReadyError(
            f"Lung detector weights do not match bundle {BUNDLE_VERSION}: "
            f"missing {missing[:3]}, unexpected {unexpected[:3]}",
            reason=f"mismatched {path}",
            hint=_WEIGHTS_HINT,
        )
    for name, shape in expected.items():
        value = weights[name]
        if tuple(value.shape) != shape or value.dtype != mx.float32:
            raise AssetNotReadyError(
                f"Lung detector tensor {name} must be float32 {shape}, "
                f"got {value.dtype} {tuple(value.shape)}",
                reason=f"mismatched {path}",
                hint=_WEIGHTS_HINT,
            )
    return weights


def _check_aliases(outputs: tuple[Path, ...], inputs: tuple[Path, ...]) -> None:
    """Refuse an output that aliases an input or another output."""
    for index, destination in enumerate(outputs):
        for other in (*inputs, *outputs[index + 1 :]):
            if destination.resolve() == other.resolve() or (
                destination.exists() and other.exists() and destination.samefile(other)
            ):
                raise InvalidInputError(
                    f"Lung detector output must not alias another path: {destination}"
                )


def _publish(texts: dict[Path, Callable[[], str]]) -> None:
    """Stage output text in a private directory, then rename it into place.

    Earlier outputs stay untouched until every new file is complete, and a failed
    write leaves no partial files behind.
    """
    with TemporaryDirectory(prefix=".lung-nodule-mlx-", dir=next(iter(texts)).parent) as staging:
        partials = {path: Path(staging) / path.name for path in texts}
        for path, text in texts.items():
            partials[path].write_text(text(), encoding="utf-8")
        for path, partial in partials.items():
            os.replace(partial, path)


def run(
    *,
    image_path: str | Path,
    weights_path: str | Path | None = None,
    output_dir: str | Path,
    overwrite: bool = False,
    resample_raw_input: bool = False,
) -> RunResult:
    """Detect lung nodule candidates in one chest CT volume.

    The default reorients to RAS and requires images already on the bundle's
    (0.703125, 0.703125, 1.25) mm grid (each axis within 1e-3 mm, no shear),
    rejecting others; the bundle's own ``Spacingd`` is disabled by default, and
    so is resampling here. Set ``resample_raw_input=True`` for
    original CT images, mirroring the bundle's ``whether_raw_luna16=true`` mode.
    Spacing uses trilinear interpolation, border padding, align_corners=False,
    float64 computation and float32 output; rotation is retained and shear
    removed. Both modes map decoded boxes with the prepared affine and MONAI's
    opposite-corner convention. Intensities are clipped from -1024 to 300 HU and
    scaled linearly into [0, 1], padded to (16, 16, 8) multiples, and passed
    through FP32 RetinaNet, with 512x512x192 constant-weight sliding windows for
    larger volumes. Up to 300 boxes remain after the score threshold (0.02),
    top-1000 selection per level, and NMS (IoU 0.22). Ties are ordered by
    original index; this can select different overlapping boxes from MONAI.

    Research software; the boxes are model output for review, not a diagnosis.

    Args:
        image_path: Chest CT volume in Hounsfield units (NIfTI), with explicit
            mm, meter or micron spatial units.
        weights_path: Optional model folder containing ``network.safetensors``. Omit
            to download ``MedMLX/retinanet-lung-nodule-mlx`` automatically. Explicit native
            ``.safetensors`` checkpoint paths remain accepted. The native weights were
            originally published at ``MedMLX/retinanet-lung-nodule-mlx`` revision
            ``d2d635fbac34254f14192ac348a05d275a949317``.
        output_dir: Directory for the outputs; created if missing.
        overwrite: Replace existing outputs (default ``False``); must be a bool.
            Outputs may not alias each other or the input or weights. New outputs
            replace earlier ones only after the whole run succeeds. Alias detection is
            a preflight check, not protection against concurrent path changes.
        resample_raw_input: Resample original CT images to the bundle grid. False
            requires inputs already on that grid and rejects others.

    Returns:
        ``outputs`` maps ``detections`` to ``detections.json``, a
        ``medmlx.detections.v1`` record of ``cccwhd`` boxes (world center x/y/z
        and size in mm) in the input image's NIfTI RAS space with class
        ``0`` (nodule) scores, and ``bundle_detections`` to
        ``bundle_detections.json``, the same boxes in the MONAI bundle's
        patient-LPS output format. ``details`` holds the detection count and
        runtime facts.

    Raises:
        InvalidInputError: The input is missing, not a finite scalar 3D image
            with mm, meter or micron units, off the bundle grid without
            ``resample_raw_input``, ``overwrite`` or ``resample_raw_input``
            is not a bool, ``output_dir`` is not a directory, outputs exist and
            ``overwrite`` is false, or an output aliases an input or another output.
            All but the image contents are checked before weights are downloaded.
        AssetNotReadyError: The weights are missing, unreadable, or do not match
            the released graph.
        ModelExecutionError: The network produced non-finite head maps.
    """

    source = Path(image_path).expanduser()
    if not source.is_file():
        raise InvalidInputError(f"Missing input image: {source}")
    for name, flag in (("overwrite", overwrite), ("resample_raw_input", resample_raw_input)):
        if not isinstance(cast(object, flag), bool):
            raise InvalidInputError(f"{name} must be true or false")
    output = Path(output_dir).expanduser()
    if output.exists() and not output.is_dir():
        raise InvalidInputError(f"output_dir is not a directory: {output}")
    detections_file = output / DETECTIONS
    bundle_file = output / BUNDLE_DETECTIONS
    outputs = (detections_file, bundle_file)
    _check_aliases(outputs, (source,))
    existing = [path.as_posix() for path in outputs if path.exists() or path.is_symlink()]
    if existing and not overwrite:
        raise InvalidInputError(
            f"Lung detector outputs already exist: {', '.join(existing)}; "
            "pass overwrite=true to replace them"
        )
    bundle_dir = (
        weights_dir("lung-nodule")
        if weights_path is None else Path(weights_path).expanduser()
    )
    weights_file = bundle_dir / "network.safetensors" if bundle_dir.is_dir() else bundle_dir
    if not weights_file.is_file():
        raise AssetNotReadyError(
            f"Missing lung detector asset: {weights_file}",
            reason=f"missing {weights_file}",
            hint=_WEIGHTS_HINT,
        )
    _check_aliases(outputs, (weights_file,))

    from medmlx_core.runtime import import_mlx

    from lung_nodule_mlx.boxes import boxes_to_world
    from lung_nodule_mlx.mlx_detector import detections, predict_maps
    from lung_nodule_mlx.mlx_model import RetinaNet

    # Host preparation precedes asset loading so input errors surface before MLX work.
    image, affine = prepare_image(source, resample_raw_input=resample_raw_input)
    mx = cast(ArrayRuntime, import_mlx())
    mx.set_cache_limit(512 * 1024**2)
    mx.reset_peak_memory()
    weights = load_weights(weights_file, mx)
    mx.eval(weights)
    network = RetinaNet(weights, mx=mx)
    maps, shape = predict_maps(image, network)
    if any(not np.isfinite(a).all() for values in maps.values() for a in values):
        raise ModelExecutionError("Non-finite lung detector head maps")
    boxes, scores, _ = detections(maps, shape, image.shape[2:])
    # The bundle's output convention is patient LPS; the record converts centers to RAS.
    lps = affine.copy()
    lps[:2] *= -1
    world = boxes_to_world(boxes, lps)
    rows = bundle_rows(str(source.resolve()), world, scores)
    record = detection_record(rows, bundle_file.resolve().as_posix())
    output.mkdir(parents=True, exist_ok=True)
    _publish({
        bundle_file: lambda: json.dumps(rows, indent=2) + "\n",
        detections_file: lambda: json.dumps(record, indent=2, sort_keys=True),
    })
    return {
        "outputs": {
            "detections": detections_file.resolve().as_posix(),
            "bundle_detections": bundle_file.resolve().as_posix(),
        },
        "details": {
            "weights_revision": weights_revision(weights_file.parent),
            "detection_count": len(boxes),
            "max_score": float(scores.max()) if len(scores) else None,
            "coordinate_convention": "nifti_ras",
            "box_mode": "cccwhd",
            "units": "mm",
            "classes": dict(CLASSES),
            "input_shape": list(image.shape),
            "resample_raw_input": resample_raw_input,
            "prepared_affine": affine.tolist(),
            "padded_shape": list(shape),
            "dtype": "float32",
            "device": str(mx.default_device()),
            "peak_memory_bytes": mx.get_peak_memory(),
        },
    }
