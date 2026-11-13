"""Record the pinned MONAI lung detector on HCC-TACE CT crops."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
from monai import __version__ as monai_version
from monai.apps.detection.utils.detector_utils import preprocess_images
from monai.bundle import ConfigParser
from monai.data import MetaTensor
from monai.transforms import ScaleIntensityRange, Spacing

ROOT = Path(__file__).resolve().parents[1]
BUNDLE_REVISION = "9d6622fda0e52be0a4155fed2e0afb9c17bcd80b"
sys.path.insert(0, str(ROOT / "tests"))
sys.path.insert(0, str(ROOT / "src"))
from _fixture_inputs import seeded_state  # noqa: E402

from lung_nodule_mlx import required_shapes  # noqa: E402


def main() -> None:
    """Run MONAI 1.4 detector and bundle postprocessing on real CT input."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument(
        "--ct", type=Path, default=Path.home() / ".cache/medmlx/test-data/hcc_063_ct.nii.gz"
    )
    args = parser.parse_args()
    if monai_version != "1.4.0" or not torch.__version__.startswith("2.13."):
        raise ValueError("MONAI 1.4.0 and Torch 2.13 are required")
    if args.bundle.name != BUNDLE_REVISION:
        raise ValueError("Supply the pinned lung-nodule bundle snapshot")
    root = ROOT / "tests/fixtures/darwin-arm64"
    with np.load(root / "runner_inputs.npz", allow_pickle=False) as data:
        inputs = {key: data[key] for key in data.files}
    metadata = json.loads(inputs["metadata"].item())
    metadata["image_source"] = "HCC-TACE-Seg HCC_063 contrast CT"
    metadata.pop("seed", None)
    inputs["metadata"] = np.asarray(json.dumps(metadata))
    ct = np.asarray(nib.load(args.ct).dataobj, dtype=np.float32)
    for case in inputs["cases"]:
        name = str(case)
        header = nib.Nifti1Header(inputs[f"header::{name}"].tobytes())
        shape = tuple(int(v) for v in header.get_data_shape())
        crop = tuple(
            slice(start, start + size) for start, size in zip((100, 210, 55), shape, strict=True)
        )
        image = nib.Nifti1Image(ct[crop].astype(header.get_data_dtype()), None, header)
        if name == "scaled":
            image.header.set_slope_inter(0.5, -200.0)
        inputs[f"nifti::{name}"] = np.frombuffer(image.to_bytes(), dtype=np.uint8)
    canonical = nib.as_closest_canonical(
        nib.Nifti1Image.from_bytes(inputs["nifti::oblique"].tobytes())
    )
    data = np.asarray(canonical.dataobj, dtype=np.float32)
    spatial = MetaTensor(torch.from_numpy(data[None].copy()), affine=canonical.affine)
    spatial = Spacing(pixdim=(0.703125, 0.703125, 1.25), mode="bilinear")(spatial)
    image = ScaleIntensityRange(a_min=-1024, a_max=300, b_min=0, b_max=1, clip=True)(spatial)
    config = ConfigParser()
    config.read_config(str(args.bundle / "configs/inference.json"))
    config["device"] = "$torch.device('cpu')"
    network = config.get_parsed_content("network_def").cpu().eval()
    seed = 20261007
    weights = seeded_state(required_shapes(), seed)
    network.load_state_dict(
        {name: torch.from_numpy(value) for name, value in weights.items()}, strict=True
    )
    detector = config.get_parsed_content("detector").cpu().eval()
    detector.set_target_keys(box_key="box", label_key="label")
    detector.set_box_selector_parameters(
        score_thresh=0.02, topk_candidates_per_level=1000, nms_thresh=0.22, detections_per_img=300
    )
    torch.set_num_threads(4)
    with torch.inference_mode():
        batch = image.as_tensor()[None]
        padded, _ = preprocess_images(batch, 3, (16, 16, 8))
        maps = network(padded)
        result = detector(batch)[0]
        world = config.get_parsed_content("postprocessing")({**result, "image": image})
    saved = {
        "metadata": json.dumps(
            {
                "monai": monai_version,
                "torch": torch.__version__,
                "bundle_revision": BUNDLE_REVISION,
                "image_source": "HCC-TACE-Seg HCC_063 contrast CT",
            }
        ),
        "case": "oblique",
        "weight_seed": seed,
        "image": batch.numpy(),
        "affine": image.affine.numpy(),
        "input_shape": np.asarray(batch.shape),
        "padded_shape": np.asarray(padded.shape[2:]),
        "boxes": result["box"].numpy(),
        "scores": result["label_scores"].numpy(),
        "labels": result["label"].numpy(),
        "world_lps": world["box"].numpy(),
        "world_scores": world["label_scores"].numpy(),
    }
    for key, tensors in maps.items():
        for level, tensor in enumerate(tensors):
            saved[f"{key}_{level}"] = tensor.numpy()
    out = ROOT / "work/real-reference"
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / "runner_inputs.npz", **inputs)
    np.savez_compressed(out / "runner_raw.npz", **saved)


if __name__ == "__main__":
    main()
