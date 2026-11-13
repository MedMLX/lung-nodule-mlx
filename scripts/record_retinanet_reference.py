"""Record pinned MONAI RetinaNet outputs on a small HCC-TACE CT crop."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
from monai import __version__ as monai_version
from monai.bundle import ConfigParser

ROOT = Path(__file__).resolve().parents[1]
BUNDLE_REVISION = "9d6622fda0e52be0a4155fed2e0afb9c17bcd80b"
sys.path.insert(0, str(ROOT / "tests"))
sys.path.insert(0, str(ROOT / "src"))
from _fixture_inputs import seeded_state  # noqa: E402

from lung_nodule_mlx import required_shapes  # noqa: E402


def main() -> None:
    """Run the bundle's source network with seeded weights and real image voxels."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument(
        "--ct", type=Path, default=Path.home() / ".cache/medmlx/test-data/hcc_063_ct.nii.gz"
    )
    args = parser.parse_args()
    if monai_version != "1.6.0" or not torch.__version__.startswith("2.13."):
        raise ValueError("MONAI 1.6.0 and Torch 2.13 are required")
    if args.bundle.name != BUNDLE_REVISION:
        raise ValueError("Supply the pinned lung-nodule bundle snapshot")
    config = ConfigParser()
    config.read_config(str(args.bundle / "configs/inference.json"))
    config["device"] = "$torch.device('cpu')"
    network = config.get_parsed_content("network_def").cpu().eval()
    seed = 20261007
    state = seeded_state(required_shapes(), seed)
    network.load_state_dict(
        {name: torch.from_numpy(value) for name, value in state.items()}, strict=True
    )
    ct = np.asarray(nib.load(args.ct).dataobj, dtype=np.float32)[100:109, 210:221, 55:60]
    image = np.clip((ct + np.float32(1024)) / np.float32(1324), 0, 1)[None, None]
    torch.set_num_threads(4)
    with torch.inference_mode():
        outputs = network(torch.from_numpy(image.copy()))
    saved = {
        "metadata": json.dumps(
            {
                "monai": monai_version,
                "torch": torch.__version__,
                "bundle_revision": BUNDLE_REVISION,
                "image_source": "HCC-TACE-Seg HCC_063 contrast CT",
            }
        ),
        "weight_seed": seed,
        "image": image,
    }
    for key, tensors in outputs.items():
        for level, tensor in enumerate(tensors):
            saved[f"{key}_{level}"] = tensor.numpy()
    destination = ROOT / "work/real-reference/retinanet.npz"
    destination.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(destination, **saved)


if __name__ == "__main__":
    main()
