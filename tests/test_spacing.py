"""Offline parity with MONAI Spacingd and the pinned bundle's box transforms."""

from __future__ import annotations

import gzip
from pathlib import Path
from typing import Protocol, cast

import nibabel as nib
import numpy as np
import pytest
from _nifti_inputs import write_mm_fixture
from _references import FIXTURES
from medmlx_core.errors import InvalidInputError

from lung_nodule_mlx import boxes_to_world, medmlx, resample_image

with np.load(
    FIXTURES.parent.parent.parent / "work/real-reference/spacing.npz", allow_pickle=False
) as recording:
    SAVED = dict(recording)
CASES = [str(name) for name in SAVED["cases"]]


class _ImageWriter(Protocol):
    """The part of ``nibabel.save`` used here, with a complete signature."""

    def save(self, img: object, filename: Path) -> None: ...


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


# MONAI Spacingd leaves these unchanged, so the unresampled mode must accept them.
ON_GRID = {"target", "near_target"}


@pytest.mark.parametrize("resample", [True, False], ids=["resampled", "unresampled"])
@pytest.mark.parametrize("case", CASES)
def test_prepare_image_matches_monai(tmp_path: Path, case: str, resample: bool) -> None:
    """Compare prepared CT voxels and affine with MONAI; unresampled mode rejects off-grid input."""
    path = tmp_path / f"{case}.nii"
    write_mm_fixture(path, SAVED[f"nifti::{case}"].tobytes())
    if not resample and case not in ON_GRID:
        with pytest.raises(InvalidInputError, match="resample_raw_input=True"):
            medmlx.prepare_image(path, resample_raw_input=False)
        return
    actual, affine = medmlx.prepare_image(path, resample_raw_input=resample)
    assert actual.dtype == np.float32 and actual.flags.c_contiguous
    assert actual.shape == SAVED[f"prepared::{case}"].shape
    np.testing.assert_allclose(actual, SAVED[f"prepared::{case}"], rtol=0, atol=1.2e-7)
    np.testing.assert_allclose(affine, SAVED[f"affine::{case}"], rtol=0, atol=1e-12)


@pytest.mark.parametrize("problem", ["truncated", "mgh"])
def test_prepare_image_rejects_damaged_and_non_nifti_files(tmp_path: Path, problem: str) -> None:
    """A cut-off .nii.gz and a valid MGH volume are input errors, not decoder errors."""
    if problem == "truncated":
        path = tmp_path / "ct.nii.gz"
        packed = gzip.compress(SAVED[f"nifti::{CASES[0]}"].tobytes())
        # The header still parses; the voxels end early.
        path.write_bytes(packed[: len(packed) // 2])
    else:
        path = tmp_path / "ct.mgz"
        volume = nib.MGHImage(SAVED[f"oriented::{CASES[0]}"], SAVED[f"source_affine::{CASES[0]}"])
        cast(_ImageWriter, nib).save(volume, path)
    with pytest.raises(InvalidInputError, match="NIfTI image"):
        medmlx.prepare_image(path)


@pytest.mark.parametrize("case", sorted(ON_GRID))
def test_unspecified_units_are_millimetres_as_in_monai(tmp_path: Path, case: str) -> None:
    """MONAI reads unknown units as mm; the unresampled path must too."""
    path = tmp_path / f"{case}.nii"
    write_mm_fixture(path, SAVED[f"nifti::{case}"].tobytes(), units="unknown")
    actual, affine = medmlx.prepare_image(path, resample_raw_input=False)
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
