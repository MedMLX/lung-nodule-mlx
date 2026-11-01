"""Declare the known millimeter geometry of stored synthetic reference inputs."""

from pathlib import Path
from typing import cast

import nibabel as nib

from lung_nodule_mlx._typing import NiftiHeader


def write_mm_fixture(path: Path, payload: bytes) -> None:
    # Replace only header units; preserve voxel bytes, scaling and coded transforms.
    """Write a synthetic NIfTI fixture after setting its spatial units to mm."""
    header = nib.Nifti1Header(payload[:348])
    cast(NiftiHeader, header).set_xyzt_units("mm")
    path.write_bytes(header.binaryblock + payload[348:])
