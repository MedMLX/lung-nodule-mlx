"""Offline parity with the pinned upstream MONAI graph and detector."""

from __future__ import annotations

import json
from typing import cast

import numpy as np
import pytest
from _fixture_inputs import seeded_weights
from _references import exact, reference

from lung_nodule_mlx import (
    RetinaNet,
    anchors,
    boxes_to_world,
    decode,
    detections,
    numpy_topk,
    predict_maps,
    required_shapes,
    select_boxes,
)
from lung_nodule_mlx._typing import ArrayRuntime, FloatArray, Maps, RealArray


@pytest.mark.mlx
def test_retinanet_matches_monai() -> None:
    """Compare seeded Metal RetinaNet head maps with the recorded MONAI graph."""
    from medmlx_core.runtime import import_mlx

    mx = cast(ArrayRuntime, import_mlx())
    saved = reference("retinanet")
    weights = seeded_weights(required_shapes(), int(saved["weight_seed"]), mx)
    actual = RetinaNet(weights, mx=mx)(cast(FloatArray, saved["image"]))
    assert set(actual) == {"classification", "box_regression"}
    for key, values in actual.items():
        assert len(values) == 3, key
        for level, value in enumerate(values):
            expected = saved[f"{key}_{level}"]
            assert value.dtype == expected.dtype == np.float32, (key, level)
            # FP32 convolution/norm reductions differ between MLX and ATen. The
            # original seeded graph's observed max is 1.78e-6; bound at 2e-6.
            np.testing.assert_allclose(value, cast(RealArray, expected), rtol=0, atol=2e-6)


def test_released_parameter_names_and_shapes() -> None:
    """Compare released graph parameter names and shapes with recorded values."""
    saved = json.loads(reference("required_shapes")["shapes"].item())
    assert required_shapes() == {key: tuple(shape) for key, shape in saved.items()}


@pytest.mark.parametrize("level", [0, 1, 2])
def test_anchor_generation(level: int) -> None:
    """Match generated anchors with the recorded MONAI feature level."""
    saved = reference("anchors_decode")
    actual = anchors(tuple(saved["shape"]), tuple(saved["grid"]), level)
    exact(actual, saved[f"anchors_{level}"], f"level {level} anchors")


def test_box_decoding_and_log_scale_clamp() -> None:
    """Compare decoded boxes and the scale clamp with the recorded reference."""
    saved = reference("anchors_decode")
    # NumPy/ATen float32 exp differ by an ULP: max endpoint difference 3.82e-6.
    np.testing.assert_allclose(
        decode(cast(FloatArray, saved["regression"]), cast(FloatArray, saved["anchors_0"])),
        cast(RealArray, saved["decoded"]),
        rtol=0,
        atol=4e-6,
    )


@pytest.mark.parametrize(
    "n,k",
    [(0, 0), (8, 8), (31, 10), (128, 2), (1001, 1000), (1000, 1000), (10000, 1000), (100000, 1000)],
)
def test_topk_scores_match_recorded_upstream(n: int, k: int) -> None:
    """Match upstream top-k scores even when equal-score indices differ."""
    saved = reference(f"topk_{n}_{k}")
    scores = cast(FloatArray, saved["scores"])
    actual = scores[numpy_topk(scores, k)]
    indices = cast(np.ndarray[tuple[int, ...], np.dtype[np.int64]], saved["indices"])
    exact(actual, scores[indices], "top-k scores")


@pytest.mark.parametrize(
    "name",
    [f"nms_{kind}{i}_{empty}" for kind in ("", "four_") for i in range(3) for empty in range(2)]
    + ["nms_limits"],
)
def test_threshold_clipping_nms_and_detection_limits(name: str) -> None:
    """Compare MONAI selection, empty inputs and ties, allowing stable tied-box order."""
    saved = reference(name)
    boxes, logits = cast(FloatArray, saved["boxes"]), cast(FloatArray, saved["logits"])
    split = int(saved["split"]) if "split" in saved else None
    levels = [slice(None)] if split is None else [slice(None, split), slice(split, None)]
    actual = select_boxes(
        [boxes[s] for s in levels], [logits[s] for s in levels], tuple(saved["shape"])
    )
    for value, key in zip(actual, ("selected", "confidences", "labels"), strict=True):
        expected = saved[key].copy()
        if name == "nms_limits" and key == "confidences":
            # NumPy/ATen sigmoid differ by up to 2 float32 ULPs (observed 1.1921e-7).
            np.testing.assert_allclose(value, cast(RealArray, expected), rtol=0, atol=1.2e-7)
            continue
        if key == "selected" and name == "nms_2_0":
            expected[:2] = expected[1::-1]
        elif key == "selected" and name in ("nms_0_0", "nms_1_0"):
            expected[0] = saved["boxes"][0]
        exact(value, expected, f"{name} {key}")


def test_multilevel_head_maps_to_detections() -> None:
    """Compare decoded multilevel detections, scores, and labels with MONAI."""
    saved = reference("detections")
    maps = {
        key: [cast(FloatArray, saved[f"{key}_{level}"]) for level in range(3)]
        for key in ("classification", "box_regression")
    }
    actual = detections(maps, tuple(saved["padded_shape"]), tuple(saved["original_shape"]))
    for value, key in zip(actual, ("selected", "confidences", "labels"), strict=True):
        if key == "labels":
            exact(value, saved[key], key)
        else:
            # Decode exp: observed 4.7684e-7 voxel error; sigmoid: 5.9605e-8.
            bound = 5e-7 if key == "selected" else 6e-8
            np.testing.assert_allclose(value, cast(RealArray, saved[key]), rtol=0, atol=bound)


@pytest.mark.parametrize(
    "name", ["direct", "overlap", "symmetric", "seed12_overlap", "seed12_symmetric"]
)
def test_window_maps_match_source_padding_overlap_and_order(name: str) -> None:
    """Compare window order and combined maps with the recorded source."""
    saved = reference(f"windows_{name}")
    calls: list[int] = []

    def replay(image: FloatArray) -> Maps:
        """Verify each input patch and replay its recorded detector maps."""
        index = len(calls)
        exact(image, saved[f"patch_{index}::image"], f"{name} patch {index}")
        calls.append(index)
        return {
            key: [
                cast(FloatArray, saved[f"patch_{index}::{key}_{level}"]).copy()
                for level in range(3)
            ]
            for key in ("classification", "box_regression")
        }

    actual, padded = predict_maps(
        cast(FloatArray, saved["image"]),
        replay,
        roi=(32, 32, 32),
        force_windows=bool(saved["force_windows"]),
    )
    assert len(calls) == int(saved["patch_count"])
    assert padded == tuple(saved["padded_shape"])
    for key, values in actual.items():
        for level, value in enumerate(values):
            exact(value, saved[f"{key}_{level}"], f"{name} {key} {level}")


def test_oblique_box_interchange() -> None:
    """Match transformed box endpoints for the recorded oblique affine."""
    saved = reference("world_boxes")
    exact(
        boxes_to_world(cast(FloatArray, saved["boxes"]), cast(RealArray, saved["affine"])),
        saved["world"],
        "world boxes",
    )
