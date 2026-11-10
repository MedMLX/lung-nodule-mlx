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
    # The public deterministic rule is stronger than upstream's unspecified
    # ordering among ties, including ties crossing the partition boundary.
    np.testing.assert_array_equal(numpy_topk(scores, k), np.argsort(-scores, kind="stable")[:k])


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


@pytest.mark.parametrize("seed", [1, 17, 93])
def test_sparse_candidate_decode_preserves_dense_selection(seed: int) -> None:
    """Retain exact detection order with tied, empty and clipped candidates."""
    rng = np.random.default_rng(seed)
    shape = (48, 40, 24)
    maps: Maps = {"classification": [], "box_regression": []}
    boxes: list[FloatArray] = []
    logits: list[FloatArray] = []
    for level, grid in enumerate(((12, 10, 6), (6, 5, 3), (3, 2, 1))):
        cls = rng.integers(-8, 4, size=(1, 3, *grid)).astype(np.float32)
        reg = rng.normal(0, 0.5, size=(1, 18, *grid)).astype(np.float32)
        if level == 1:
            cls.fill(-100)
        maps["classification"].append(cls)
        maps["box_regression"].append(reg)
        boxes.append(
            decode(reg[0].transpose(1, 2, 3, 0).reshape(-1, 6), anchors(shape, grid, level))
        )
        logits.append(cls[0].transpose(1, 2, 3, 0).reshape(-1, 1))
    expected = select_boxes(boxes, logits, shape)
    actual = detections(maps, shape, shape)
    for result, baseline in zip(actual, expected, strict=True):
        np.testing.assert_array_equal(result, baseline)


@pytest.mark.parametrize("bad_delta", [np.finfo(np.float32).max, np.nan, np.inf])
def test_discarded_candidates_still_reject_nonfinite_decoded_boxes(bad_delta: float) -> None:
    maps: Maps = {
        "classification": [np.full((1, 3, 2, 2, 2), -100, dtype=np.float32)],
        "box_regression": [np.zeros((1, 18, 2, 2, 2), dtype=np.float32)],
    }
    maps["box_regression"][0][0, 0, 0, 0, 0] = bad_delta
    with (
        np.errstate(over="ignore", invalid="ignore"),
        pytest.raises(
            ValueError,
            match="Non-finite decoded lung boxes",
        ),
    ):
        detections(maps, (8, 8, 8), (8, 8, 8))


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


@pytest.mark.mlx
@pytest.mark.parametrize("stride", [(1, 1, 1), (2, 1, 2)])
def test_multi_slab_convolution_preserves_outputs(stride: tuple[int, int, int]) -> None:
    """Keep padding, striding and the short final slab equal under both storage policies."""
    from medmlx_core.runtime import import_mlx

    mx = cast(ArrayRuntime, import_mlx())
    rng = np.random.default_rng(563)
    values = mx.array(rng.normal(0, 1, (1, 67, 64, 48, 64)).astype(np.float32))
    weights = {
        "head.weight": mx.array(rng.normal(0, 0.01, (4, 3, 3, 3, 64)).astype(np.float32)),
        "head.bias": mx.array(rng.normal(0, 0.01, (4,)).astype(np.float32)),
    }
    compact = RetinaNet(weights, mx=mx).conv(values, "head", stride)
    retained = RetinaNet(weights, mx=mx, conv_output_policy="concatenate").conv(
        values, "head", stride
    )
    padded = mx.pad(values, [(0, 0), (1, 1), (1, 1), (1, 1), (0, 0)])
    direct = mx.conv3d(padded, weights["head.weight"], stride=stride) + weights["head.bias"]
    mx.eval(compact, retained, direct)
    observed = cast(FloatArray, np.asarray(compact))
    np.testing.assert_array_equal(observed, cast(FloatArray, np.asarray(retained)))
    # Native full-volume and slab kernels may choose different FP32 contractions.
    np.testing.assert_allclose(observed, cast(FloatArray, np.asarray(direct)), rtol=0, atol=2e-6)
