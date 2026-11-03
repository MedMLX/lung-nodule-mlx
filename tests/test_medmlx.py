"""Runner parity against recorded upstream MONAI inference, and argument rejection."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import NoReturn, cast

import numpy as np
import pytest
from _fixture_inputs import seeded_weights
from _nifti_inputs import write_mm_fixture
from _references import FIXTURES, exact, load, runner_atol
from medmlx_core.errors import AssetNotReadyError, InvalidInputError

from lung_nodule_mlx import medmlx, required_shapes
from lung_nodule_mlx._typing import (
    ArrayRuntime,
    FloatArray,
    Maps,
    RealArray,
)

WEIGHT_SEED = 20261007
INPUTS = load(FIXTURES / "runner_inputs.npz")
LPS_TO_RAS = np.array([-1, -1, 1, 1, 1, 1])


def write_volume(directory: Path, case: str = "oblique") -> Path:
    """Write a recorded CT volume with explicit millimeter spatial units."""
    path = directory / f"{case}.nii"
    write_mm_fixture(path, INPUTS[f"nifti::{case}"].tobytes())
    return path


@pytest.mark.parametrize("case", [str(case) for case in INPUTS["cases"]])
def test_prepare_image_matches_upstream(tmp_path: Path, case: str) -> None:
    """Compare pre-resampled image values and affine with upstream recordings."""
    image, affine = medmlx.prepare_image(write_volume(tmp_path, case), resample_raw_input=False)
    exact(image, cast(FloatArray, INPUTS[f"image::{case}"]), f"{case} image")
    exact(affine, INPUTS[f"affine::{case}"], f"{case} affine")


@pytest.fixture(scope="module")
def mx() -> ArrayRuntime:
    """Load the Metal runtime for seeded inference tests."""
    from medmlx_core.runtime import import_mlx

    return cast(ArrayRuntime, import_mlx())


@pytest.fixture(scope="module")
def weights_file(tmp_path_factory: pytest.TempPathFactory, mx: ArrayRuntime) -> Path:
    """Seeded full-graph weights in the converted safetensors format the runner loads."""
    path = tmp_path_factory.mktemp("weights") / "network.safetensors"
    mx.save_safetensors(str(path), seeded_weights(required_shapes(), WEIGHT_SEED, mx))
    return path


@pytest.mark.mlx
@pytest.mark.parametrize("raw", [False, True], ids=["pre-resampled", "raw"])
def test_run_matches_upstream_runner(tmp_path: Path, weights_file: Path, raw: bool) -> None:
    """Compare seeded runner outputs with recorded upstream detections."""
    source = write_volume(tmp_path)
    output = tmp_path / "out"
    output.mkdir()
    victim = tmp_path / "victim.json"
    victim.write_text("original", encoding="utf-8")
    (output / f".{os.getpid()}.{medmlx.DETECTIONS}").symlink_to(victim)
    result = medmlx.run(
        image_path=str(source),
        weights_path=str(weights_file),
        output_dir=str(output),
        resample_raw_input=raw,
    )
    assert victim.read_text(encoding="utf-8") == "original"
    bundle_file = Path(result["outputs"]["bundle_detections"])
    name = "runner_raw" if raw else "runner"
    expected = load(FIXTURES / f"{name}.npz")
    assert int(expected["weight_seed"]) == WEIGHT_SEED
    rows = json.loads(bundle_file.read_text())
    assert len(rows) == 1 and rows[0]["image"] == str(source.resolve())
    exact(np.asarray(rows[0]["label"], dtype=np.int64), expected["labels"], "runner labels")
    # FP32 convolution/norm and affine rounding differ from ATen; the contract
    # records the measured Metal errors for these seeded inputs.
    atol = runner_atol(name)
    for key, column in (("world_lps", "box"), ("world_scores", "label_scores")):
        np.testing.assert_allclose(
            np.asarray(rows[0][column], dtype=np.float64),
            cast(RealArray, expected[key]).astype(np.float64),
            rtol=0,
            atol=atol[key],
        )
    record = json.loads(Path(result["outputs"]["detections"]).read_text())
    # Only the world center's x and y change sign between patient LPS and NIfTI RAS.
    exact(
        np.asarray([item["box"] for item in record["detections"]]),
        np.asarray(rows[0]["box"]) * LPS_TO_RAS,
        "RAS record",
    )


@pytest.mark.mlx
@pytest.mark.parametrize("raw", [False, True], ids=["pre-resampled", "raw"])
def test_runner_maps_prepared_grid_to_upstream_world_boxes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, raw: bool
) -> None:
    """Replay recorded upstream head maps and compare the runner's world boxes."""
    from lung_nodule_mlx import mlx_detector, mlx_model

    expected = load(FIXTURES / ("runner_raw.npz" if raw else "runner.npz"))
    weights = tmp_path / "network.safetensors"
    weights.write_bytes(b"recorded head maps replace network execution")

    def no_weights(path: Path, mx: ArrayRuntime) -> dict[str, object]:
        return {}

    def no_network(weights: dict[str, object], *, mx: ArrayRuntime) -> None:
        pass

    def predict(image: FloatArray, network: None) -> tuple[Maps, tuple[int, ...]]:
        """Check prepared voxels and replay the recorded head maps."""
        # Raw Spacing float64 grid differences can produce one FP32 ULP.
        np.testing.assert_allclose(
            image, cast(FloatArray, expected["image"]), rtol=0, atol=1.2e-7 if raw else 0
        )
        return {
            key: [cast(FloatArray, expected[f"{key}_{level}"]).copy() for level in range(3)]
            for key in ("classification", "box_regression")
        }, tuple(expected["padded_shape"])

    monkeypatch.setattr(medmlx, "load_weights", no_weights)
    monkeypatch.setattr(mlx_model, "RetinaNet", no_network)
    monkeypatch.setattr(mlx_detector, "predict_maps", predict)
    options = {"resample_raw_input": True} if raw else {}  # Exercise the public default.
    result = medmlx.run(
        image_path=str(write_volume(tmp_path)),
        weights_path=str(weights),
        output_dir=str(tmp_path / "out"),
        **options,
    )
    rows = json.loads(Path(result["outputs"]["bundle_detections"]).read_text())
    exact(np.asarray(rows[0]["label"], dtype=np.int64), expected["labels"], "runner labels")
    # Float32 NumPy/ATen exp and affine multiplication: observed maximum 1.9074e-6 mm.
    np.testing.assert_allclose(
        rows[0]["box"], cast(FloatArray, expected["world_lps"]), rtol=0, atol=2e-6
    )
    # Observed sigmoid rounding bound, independent of learned-network execution.
    np.testing.assert_allclose(
        rows[0]["label_scores"], cast(FloatArray, expected["world_scores"]), rtol=0, atol=6e-8
    )


@pytest.mark.parametrize(
    "problem", ["missing_image", "non_bool_resample", "existing_output", "dangling_output"]
)
def test_run_rejects_bad_arguments_before_downloading_weights(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, problem: str
) -> None:
    """Each rejection category fails before any weight download."""

    def download(model_id: str) -> NoReturn:
        """Fail the test if weights are fetched before the arguments are checked."""
        raise AssertionError(f"downloaded {model_id} before checking arguments")

    monkeypatch.setattr(medmlx, "weights_dir", download)
    image, output, resample = write_volume(tmp_path), tmp_path / "out", False
    if problem == "missing_image":
        image = tmp_path / "absent.nii"
    elif problem == "non_bool_resample":
        resample = cast(bool, 1)
    elif problem == "existing_output":
        output.mkdir()
        (output / "detections.json").write_bytes(b"earlier output")
    else:
        output.mkdir()
        (output / "detections.json").symlink_to(tmp_path / "absent.json")
    with pytest.raises(InvalidInputError):
        medmlx.run(image_path=image, output_dir=output, resample_raw_input=resample)
    if problem == "dangling_output":
        assert (output / "detections.json").is_symlink()


@pytest.mark.mlx
def test_run_rejects_missing_and_mismatched_weights(
    tmp_path: Path, weights_file: Path, mx: ArrayRuntime
) -> None:
    """Reject missing, unreadable, incomplete and unconverted-layout checkpoints."""
    weights = mx.load(str(weights_file), format="safetensors")
    name = "feature_extractor.body.conv1.weight"
    (tmp_path / "broken.safetensors").write_bytes(b"not safetensors")
    partial = {key: value for key, value in weights.items() if key != name}
    mx.save_safetensors(str(tmp_path / "partial.safetensors"), partial)
    oidhw = {**weights, name: mx.transpose(weights[name], (0, 4, 1, 2, 3))}
    mx.save_safetensors(str(tmp_path / "oidhw.safetensors"), oidhw)
    source = write_volume(tmp_path)
    for label in ("absent", "broken", "partial", "oidhw"):
        with pytest.raises(AssetNotReadyError):
            medmlx.run(
                image_path=source,
                weights_path=tmp_path / f"{label}.safetensors",
                output_dir=tmp_path / "out",
            )
    assert not (tmp_path / "out").exists()
