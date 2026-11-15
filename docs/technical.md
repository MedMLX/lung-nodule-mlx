# Technical notes

## Backbone and head memory

The backbone and the prediction heads bound their temporaries to slabs of the
convolution output. Batch norm, ReLU and residual adds run as conv epilogues,
conv1 feeds a streamed MaxPool3d, the FPN top-down add happens per lateral slab,
and head GroupNorm reduces 1024-sample blocks chunk by chunk. Slab shapes depend
only on input, weight and stride, so every operation keeps the FP32 contraction
it had before.

Check on the converted HCC-TACE HCC_063 CT with the released weights, `detect`
with default arguments, against the previous implementation on the same
machine: the four detections have identical boxes and scores (maximum absolute
difference 0.0), and peak MLX memory falls from 10.2 GB to 6.0 GB. Seeded-weight
parity with the recorded MONAI references (`tests/test_equivalence.py`,
`tests/test_medmlx.py`) passes at the existing tolerances.
