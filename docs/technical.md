# Technical notes

## Backbone and head memory

The backbone and the prediction heads bound their temporaries to slabs of the
convolution output. Batch norm, ReLU and residual adds run as conv epilogues,
conv1 feeds a streamed MaxPool3d, the FPN top-down add happens per lateral slab,
and head GroupNorm reduces 1024-sample blocks chunk by chunk. Slab shapes depend
only on input, weight and stride, so every operation keeps the FP32 contraction
it had before.

Check on the converted HCC-TACE HCC_063 CT with the released weights, `run`
with default arguments, against the previous implementation on the same
machine: the four detections have identical boxes and scores (maximum absolute
difference 0.0), and peak MLX memory falls from 10.2 GB to 6.0 GB. Seeded-weight
parity with the recorded MONAI references (`tests/test_equivalence.py`,
`tests/test_medmlx.py`) passes at the existing tolerances.

The runner limits the MLX free cache to 512 MiB during inference, keeps a lower
limit the caller already set, and restores the caller's limit when inference
finishes or raises an error.

## CT resampling

For a diagonal map from output to input voxels, the resampler builds indices
and weights once per axis. Each voxel uses the original eight-corner order
and float64 accumulation before conversion to float32. Maps with off-diagonal
terms use the general trilinear path.

On an Apple M1 Max (64 GiB, macOS 27.2.1, NumPy 2.5.4, MLX 0.32.4), Perfkit
compared the resampling change against `44ca491` in six balanced rounds of fresh
processes. The LUNA16 scan `nodule_01_25mm.nii.gz` went from 512 × 512 × 147 to
512 × 512 × 293 voxels. Each process timed its first call and two warm
`prepare_image` calls or one warm `run` call. Full runs loaded cached released
weights, performed inference on Metal, and wrote both detection files. Memory
columns report the process peak across calls and the MLX allocation peak from
the warm full run.

| Workload | Warm p50 (s) | First-call p50 (s) | Peak process footprint (MiB) | Peak MLX allocation (MiB) |
| --- | ---: | ---: | ---: | ---: |
| `prepare_image` | 2.566 → 1.515 | 2.679 → 1.636 | 1336.1 → 1271.6 | CPU only |
| `run` | 28.510 → 27.454 | 28.617 → 27.559 | 8639.8 → 8579.9 | 5715.8 → 5715.8 |

The 97% paired intervals put the warm latency reduction at 40.3%–42.9% for
`prepare_image` and 3.5%–3.9% for `run`. A trial C-order conversion during
image loading gave 1.547 s warm latency and a 1274.4 MiB peak footprint on the
same preprocessing workload. Its 97% warm latency interval relative to the
separable patch was 1.1%–3.2% slower, an inconclusive difference under
Perfkit's 2% minimum-effect threshold.

Prepared voxels and affines matched bit for bit across two LUNA16 scans, an
oblique copy with shear, and nine recorded spacing fixtures. The 41 detections
across the three LUNA16 inputs and the recorded runner fixture had identical
boxes and scores in both RAS and LPS.
