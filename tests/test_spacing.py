"""Offline parity with MONAI Spacingd and the pinned bundle's box transforms."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from _nifti_inputs import write_mm_fixture
from _references import FIXTURES
from medmlx_core.errors import InvalidInputError

from lung_nodule_mlx import boxes_to_world, medmlx, resample_image

with np.load(FIXTURES / "spacing.npz", allow_pickle=False) as recording:
    SAVED = dict(recording)
CASES = [str(name) for name in SAVED["cases"]]


@pytest.mark.parametrize("case", CASES)
def test_spacing_matches_monai(case: str) -> None:
    """Compare image resampling and affine with the recorded MONAI spacing."""
    actual, affine = resample_image(SAVED[f"oriented::{case}"], SAVED[f"source_affine::{case}"])
    expected = SAVED[f"sampled::{case}"]
    assert actual.dtype == expected.dtype == np.float32
    assert actual.shape == expected.shape
    # Float64 grid normalization and CPU BLAS round differently before FP32 output.
    np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-4)
    assert affine.dtype == np.float64
    np.testing.assert_allclose(affine, SAVED[f"affine::{case}"], rtol=0, atol=1e-12)


# MONAI Spacingd leaves these unchanged, so the unresampled default must accept them.
ON_GRID = {"target", "near_target"}


@pytest.mark.parametrize("resample", [True, False], ids=["resampled", "default"])
@pytest.mark.parametrize("case", CASES)
def test_prepare_image_matches_monai(tmp_path: Path, case: str, resample: bool) -> None:
    """Compare prepared CT voxels and affine with MONAI; default rejects off-grid input."""
    path = tmp_path / f"{case}.nii"
    write_mm_fixture(path, SAVED[f"nifti::{case}"].tobytes())
    if not resample and case not in ON_GRID:
        with pytest.raises(InvalidInputError, match="resample_raw_input=True"):
            medmlx.prepare_image(path)
        return
    actual, affine = medmlx.prepare_image(path, resample_raw_input=resample)
    assert actual.dtype == np.float32 and actual.flags.c_contiguous
    assert actual.shape == SAVED[f"prepared::{case}"].shape
    np.testing.assert_allclose(actual, SAVED[f"prepared::{case}"], rtol=0, atol=1.2e-7)
    np.testing.assert_allclose(affine, SAVED[f"affine::{case}"], rtol=0, atol=1e-12)


@pytest.mark.parametrize("case", CASES)
def test_world_boxes_match_monai(case: str) -> None:
    """Compare LPS world boxes, including empty results, with MONAI."""
    affine = SAVED[f"affine::{case}"].copy()
    affine[:2] *= -1
    actual = boxes_to_world(SAVED[f"clipped::{case}"], affine)
    assert actual.dtype == SAVED[f"world_lps::{case}"].dtype == np.float32
    np.testing.assert_allclose(actual, SAVED[f"world_lps::{case}"], rtol=0, atol=8e-6)
