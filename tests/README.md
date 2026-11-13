The image and detector references use the converted HCC-TACE-Seg HCC_063
contrast CT at `~/.cache/medmlx/test-data/hcc_063_ct.nii.gz`. Image-derived
inputs and outputs are recorded in `work/real-reference`, outside Git. The
tracked NIfTI headers specify controlled test geometries; they do not describe
the original patient's geometry.

Run `scripts/record_spacing_reference.py`, `scripts/record_windows_reference.py`,
and `scripts/record_retinanet_reference.py` with MONAI 1.6.0 and Torch 2.13.
The RetinaNet recorder takes `--bundle` pointing to the pinned
`9d6622fda0e52be0a4155fed2e0afb9c17bcd80b` bundle snapshot and also needs the
project's test dependencies. Run `scripts/record_runner_reference.py` with the
same bundle and Torch 2.13 but MONAI 1.4.0: that release has the bundle's anchor
convention. Each recorder uses the HCC-TACE cache path by default; the network
and runner recorders accept `--ct` to select another CT path.

After recording, run `python -m pytest tests/test_spacing.py
tests/test_equivalence.py tests/test_medmlx.py` in the project environment.
