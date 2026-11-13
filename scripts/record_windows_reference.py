"""Record pinned MONAI sliding-window maps on HCC-TACE CT crops."""

from __future__ import annotations

from pathlib import Path

import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as functional
from monai import __version__ as monai_version
from monai.inferers import SlidingWindowInferer

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "work/real-reference"


def main() -> None:
    """Record patch order and multiscale accumulation with a deterministic source predictor."""
    if monai_version != "1.6.0" or not torch.__version__.startswith("2.13."):
        raise ValueError("MONAI 1.6.0 and Torch 2.13 are required")
    ct = np.asarray(
        nib.load(Path.home() / ".cache/medmlx/test-data/hcc_063_ct.nii.gz").dataobj,
        dtype=np.float32,
    )
    OUTPUT.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    cases = {
        "direct": ((17, 19, 9), False, (100, 210, 55)),
        "overlap": ((47, 31, 17), True, (100, 210, 55)),
        "symmetric": ((17, 19, 9), True, (100, 210, 55)),
        "seed12_overlap": ((47, 31, 17), True, (120, 220, 60)),
        "seed12_symmetric": ((17, 19, 9), True, (120, 220, 60)),
    }
    for name, (shape, forced, origin) in cases.items():
        crop = tuple(slice(start, start + size) for start, size in zip(origin, shape, strict=True))
        image = np.clip((ct[crop] + np.float32(1024)) / np.float32(1324), 0, 1)[None, None]
        padded_shape = tuple(
            ((size + divisor - 1) // divisor) * divisor
            for size, divisor in zip(shape, (16, 16, 8), strict=True)
        )
        pad = tuple(
            value
            for size, target in reversed(list(zip(shape, padded_shape, strict=True)))
            for value in (0, target - size)
        )
        padded = functional.pad(torch.from_numpy(image.copy()), pad)
        saved = {"image": image, "padded_shape": np.asarray(padded_shape), "force_windows": forced}
        patches = []

        def predictor(window, saved=saved, patches=patches):
            maps = [
                functional.avg_pool3d(window, kernel_size=2**level, stride=2**level)
                for level in (1, 2, 3)
            ]
            fields = tuple(maps + [value * 0.5 for value in maps])
            index = len(patches)
            saved[f"patch_{index}::image"] = window.numpy().copy()
            for field, output in zip(
                ("classification", "box_regression"), (fields[:3], fields[3:]), strict=True
            ):
                for level, tensor in enumerate(output):
                    saved[f"patch_{index}::{field}_{level}"] = tensor.numpy().copy()
            patches.append(index)
            return fields

        with torch.inference_mode():
            if not forced:
                combined = predictor(padded)
            else:
                inferer = SlidingWindowInferer(
                    roi_size=(32, 32, 32),
                    sw_batch_size=1,
                    overlap=0.25,
                    mode="constant",
                    device="cpu",
                )
                combined = inferer(padded, predictor)
        saved["patch_count"] = len(patches)
        for field, output in zip(
            ("classification", "box_regression"), (combined[:3], combined[3:]), strict=True
        ):
            for level, tensor in enumerate(output):
                saved[f"{field}_{level}"] = tensor.numpy().copy()
        np.savez_compressed(OUTPUT / f"windows_{name}.npz", **saved)


if __name__ == "__main__":
    main()
