"""Record MONAI spacing and intensity outputs on HCC-TACE CT crops."""

from __future__ import annotations

import json
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
from monai import __version__ as monai_version
from monai.data import MetaTensor
from monai.transforms import ScaleIntensityRange, Spacing

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "tests/fixtures/darwin-arm64/spacing.npz"
OUTPUT = ROOT / "work/real-reference/spacing.npz"


def main() -> None:
    """Run pinned MONAI transforms on CT voxels with controlled NIfTI geometries."""
    if monai_version != "1.6.0" or not torch.__version__.startswith("2.13."):
        raise ValueError("MONAI 1.6.0 and Torch 2.13 are required")
    ct = np.asarray(
        nib.load(Path.home() / ".cache/medmlx/test-data/hcc_063_ct.nii.gz").dataobj,
        dtype=np.float32,
    )
    with np.load(FIXTURE, allow_pickle=False) as archive:
        saved = {key: archive[key] for key in archive.files}
    metadata = json.loads(saved["metadata"].item())
    metadata["image_source"] = "HCC-TACE-Seg HCC_063 contrast CT"
    metadata.pop("seed", None)
    saved["metadata"] = np.asarray(json.dumps(metadata))
    for case in saved["cases"]:
        name = str(case)
        header = nib.Nifti1Header(saved[f"header::{name}"].tobytes())
        shape = tuple(int(v) for v in header.get_data_shape())
        slices = tuple(
            slice(start, start + length)
            for start, length in zip((100, 210, 55), shape, strict=True)
        )
        voxels = ct[slices].astype(header.get_data_dtype())
        image = nib.Nifti1Image(voxels, None, header)
        if name == "scaled":
            image.header.set_slope_inter(0.5, -200.0)
        saved[f"nifti::{name}"] = np.frombuffer(image.to_bytes(), dtype=np.uint8)
        canonical = nib.as_closest_canonical(nib.Nifti1Image.from_bytes(image.to_bytes()))
        oriented = np.asarray(canonical.dataobj, dtype=np.float32)
        saved[f"oriented::{name}"] = oriented
        saved[f"source_affine::{name}"] = canonical.affine
        source = MetaTensor(torch.from_numpy(oriented[None].copy()), affine=canonical.affine)
        result = Spacing(pixdim=(0.703125, 0.703125, 1.25), mode="bilinear")(source)
        sampled = result.as_tensor().numpy()[0]
        saved[f"sampled::{name}"] = sampled
        saved[f"affine::{name}"] = result.affine.numpy()
        prepared = ScaleIntensityRange(a_min=-1024, a_max=300, b_min=0, b_max=1, clip=True)(result)
        saved[f"prepared::{name}"] = prepared.as_tensor().numpy()[None]
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUTPUT, **saved)


if __name__ == "__main__":
    main()
