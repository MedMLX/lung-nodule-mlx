"""FP32 MLX graph for MONAI lung_nodule_ct_detection 0.6.10.

The released ResNet50 FPN uses only stages 1 and 2. Spatial axes retain
MONAI's (i, j, k) order; only the channel dimension moves for Metal.
"""

from __future__ import annotations

from math import gcd
from typing import TYPE_CHECKING, cast

import numpy as np
from medmlx_core.layout import to_ncdhw, to_ndhwc

from lung_nodule_mlx._typing import ArrayRuntime, FloatArray, Maps

if TYPE_CHECKING:
    from mlx.core import array


def required_shapes() -> dict[str, tuple[int, ...]]:
    """Names and source-layout (PyTorch OIDHW) shapes of the released inference graph.

    Includes the batch-norm ``num_batches_tracked`` scalars (shape ``()``) that the
    checkpoint holds but inference does not use.
    """
    shapes: dict[str, tuple[int, ...]] = {}

    def conv(p: str, out: int, inp: int, k: int, bias: bool = False) -> None:
        """Record a convolution weight shape and optional bias shape."""
        shapes[p + ".weight"] = (out, inp, k, k, k)
        if bias:
            shapes[p + ".bias"] = (out,)

    def norm(p: str, n: int, batch: bool = True) -> None:
        """Record normalization parameters and optional running statistics."""
        for key in (
            ("weight", "bias", "running_mean", "running_var") if batch else ("weight", "bias")
        ):
            shapes[p + "." + key] = (n,)
        if batch:
            shapes[p + ".num_batches_tracked"] = ()

    body = "feature_extractor.body"
    conv(body + ".conv1", 64, 1, 7)
    norm(body + ".bn1", 64)
    inp = 64
    for stage, blocks, n in ((1, 3, 64), (2, 4, 128)):
        for block in range(blocks):
            p = f"{body}.layer{stage}.{block}"
            for i, out, channels, kernel in ((1, n, inp, 1), (2, n, n, 3), (3, n * 4, n, 1)):
                conv(f"{p}.conv{i}", out, channels, kernel)
                norm(f"{p}.bn{i}", out)
            if block == 0:
                conv(p + ".downsample.0", n * 4, inp, 1, True)
                norm(p + ".downsample.1", n * 4)
            inp = n * 4
    for level, n in enumerate((256, 512)):
        conv(f"feature_extractor.fpn.inner_blocks.{level}", 256, n, 1, True)
    for level in range(2):
        conv(f"feature_extractor.fpn.layer_blocks.{level}", 256, 256, 3, True)
    for head, last, n in (("classification", "cls_logits", 3), ("regression", "bbox_reg", 18)):
        for i in range(4):
            conv(f"{head}_head.conv.{i * 3}", 256, 256, 3, True)
            norm(f"{head}_head.conv.{i * 3 + 1}", 256, False)
        conv(f"{head}_head.{last}", n, 256, 3, True)
    return shapes


class RetinaNet:
    """One request's network, with evaluated spatial slabs bounding conv workspace.

    Activations are channels-last ``(N, D, H, W, C)`` (``D, H, W`` are the three
    spatial axes of the RAS-oriented volume); convolution weights are ``ODHWI``.
    FP32 arithmetic requires FP32 weights; the public runner validates them,
    while this class binds the supplied mapping without validation.
    """

    def __init__(self, weights: dict[str, array], *, mx: ArrayRuntime) -> None:
        """Bind native weights and the MLX runtime."""
        self.mx = mx
        self.weights = weights

    def conv(self, x: array, p: str, stride: tuple[int, int, int] = (1, 1, 1)) -> array:
        """Cubic-kernel 3D convolution of ``x`` with weights ``p + ".weight"``.

        Padding is ``kernel // 2`` per side and the bias, if present, is added.
        The output is computed in slabs along the first spatial axis.
        """
        mx = self.mx
        w = self.weights[p + ".weight"]
        k = w.shape[1]
        pad = k // 2
        # Bound the explicit conv3d lowering workspace to approximately 256 MiB.
        plane = ((x.shape[2] + 2 * pad - k) // stride[1] + 1) * (
            (x.shape[3] + 2 * pad - k) // stride[2] + 1
        )
        slab = max(1, (256 * 1024**2) // (4 * plane * k**3 * x.shape[-1]))
        length = (x.shape[1] + 2 * pad - k) // stride[0] + 1
        chunks: list[array] = []
        for start in range(0, length, slab):
            stop = min(start + slab, length)
            lo, hi = start * stride[0] - pad, (stop - 1) * stride[0] - pad + k
            crop = x[:, max(lo, 0) : min(hi, x.shape[1])]
            crop = mx.pad(
                crop,
                [(0, 0), (max(0, -lo), max(0, hi - x.shape[1])), (pad, pad), (pad, pad), (0, 0)],
            )
            y = mx.conv3d(crop, w, stride=stride)
            if p + ".bias" in self.weights:
                y = y + self.weights[p + ".bias"]
            mx.eval(y)
            chunks.append(y)
        out = mx.concatenate(chunks, axis=1) if len(chunks) > 1 else chunks[0]
        mx.eval(out)
        return out

    def batch_norm(self, x: array, p: str) -> array:
        """Inference-mode batch norm from the stored running statistics (eps 1e-5)."""
        w, mx = self.weights, self.mx
        scale = w[p + ".weight"] * mx.rsqrt(w[p + ".running_var"] + 1e-5)
        return (x - w[p + ".running_mean"]) * scale + w[p + ".bias"]

    def features(self, x: array) -> list[array]:
        """Run the ResNet50 stem and stages 1 and 2; return both stage outputs."""
        mx = self.mx
        p = "feature_extractor.body"
        x = mx.maximum(self.batch_norm(self.conv(x, p + ".conv1", (2, 2, 1)), p + ".bn1"), 0)
        # MaxPool3d(3, 2, 1), with negative infinity outside the source grid.
        x = mx.pad(x, [(0, 0), (1, 1), (1, 1), (1, 1), (0, 0)], constant_values=-float("inf"))
        shape = tuple((x.shape[i] - 3) // 2 + 1 for i in (1, 2, 3))
        pooled = None
        for i in range(3):
            for j in range(3):
                for k in range(3):
                    part = x[
                        :,
                        i : i + 2 * shape[0] : 2,
                        j : j + 2 * shape[1] : 2,
                        k : k + 2 * shape[2] : 2,
                    ]
                    pooled = part if pooled is None else mx.maximum(pooled, part)
        x = cast("array", pooled)
        mx.eval(x)
        stages: list[array] = []
        for stage, blocks in ((1, 3), (2, 4)):
            for block in range(blocks):
                q = f"{p}.layer{stage}.{block}"
                stride = (2, 2, 2) if stage == 2 and block == 0 else (1, 1, 1)
                residual = (
                    self.batch_norm(self.conv(x, q + ".downsample.0", stride), q + ".downsample.1")
                    if block == 0
                    else x
                )
                y = mx.maximum(self.batch_norm(self.conv(x, q + ".conv1"), q + ".bn1"), 0)
                y = mx.maximum(self.batch_norm(self.conv(y, q + ".conv2", stride), q + ".bn2"), 0)
                x = mx.maximum(
                    self.batch_norm(self.conv(y, q + ".conv3"), q + ".bn3") + residual, 0
                )
                mx.eval(x)
            stages.append(x)
        return stages

    def pyramid(self, stages: list[array]) -> list[array]:
        """Feature pyramid over the two stages; returns three maps, finest first.

        The third map is a stride-2 subsample of the coarsest one (the FPN's extra
        level).
        """
        mx = self.mx
        p = "feature_extractor.fpn"
        inner = self.conv(stages[1], p + ".inner_blocks.1")
        low = self.conv(inner, p + ".layer_blocks.1")
        lateral = self.conv(stages[0], p + ".inner_blocks.0")
        # Nearest-neighbor resize of the coarse map to the lateral grid: source index
        # floor(i * in / out) on each axis.
        for axis in (1, 2, 3):
            indices = mx.arange(lateral.shape[axis]) * inner.shape[axis] // lateral.shape[axis]
            inner = mx.take(inner, indices, axis=axis)
        high = self.conv(lateral + inner, p + ".layer_blocks.0")
        return [high, low, low[:, ::2, ::2, ::2]]

    def head(self, x: array, name: str) -> array:
        """Run one prediction head on a pyramid map.

        ``name`` is ``"classification"`` or ``"regression"``. Four blocks of
        conv, GroupNorm (8 groups of 32 channels, eps 1e-5, per-channel affine)
        and ReLU precede the output conv, which has 3 channels for classification
        (one logit per anchor) and 18 for regression (six deltas per anchor).
        """
        mx = self.mx
        for i in range(4):
            p = f"{name}_head.conv"
            x = self.conv(x, f"{p}.{i * 3}")
            shape = x.shape
            grouped = mx.reshape(
                mx.transpose(mx.reshape(x, (shape[0], -1, 8, 32)), (0, 2, 1, 3)),
                (shape[0], 8, -1),
            )
            # Full windows contain 50M samples per group. A single long reduction
            # drifts beyond the source FP32 tolerance; balance sums in <=1024 blocks.
            blocked = mx.reshape(grouped, (shape[0], 8, -1, gcd(grouped.shape[-1], 1024)))
            mean = mx.mean(mx.mean(blocked, axis=-1), axis=-1, keepdims=True)
            variance = mx.mean(
                mx.mean(mx.square(blocked - mean[..., None]), axis=-1), axis=-1, keepdims=True
            )
            normalized = (grouped - mean) * mx.rsqrt(variance + 1e-5)
            x = mx.reshape(
                mx.transpose(mx.reshape(normalized, (shape[0], 8, -1, 32)), (0, 2, 1, 3)), shape
            )
            x = mx.maximum(
                x * self.weights[f"{p}.{i * 3 + 1}.weight"] + self.weights[f"{p}.{i * 3 + 1}.bias"],
                0,
            )
            mx.eval(x)
        suffix = "cls_logits" if name == "classification" else "bbox_reg"
        return self.conv(x, f"{name}_head.{suffix}")

    def __call__(self, image: FloatArray) -> Maps:
        """Run the detector on a host volume and return host head maps.

        Args:
            image: Float32 ``(N, 1, X, Y, Z)`` volume (channels-first). Callers
                pad the spatial sizes to multiples of ``(16, 16, 8)``, as
                ``predict_maps`` does.

        Returns:
            Per-level float32 NCDHW arrays, finest level first, under
            ``"classification"`` (``C = 3`` logits) and ``"box_regression"``
            (``C = 18`` deltas). The supplied runtime selects the execution device.
        """
        mx = self.mx
        x = to_ndhwc(mx.array(np.asarray(image, dtype=np.float32)), mx)
        pyramid = self.pyramid(self.features(x))
        result: Maps = {"classification": [], "box_regression": []}
        for feature in pyramid:
            for name, key in (
                ("classification", "classification"),
                ("regression", "box_regression"),
            ):
                out = to_ncdhw(self.head(feature, name), mx)
                mx.eval(out)
                result[key].append(cast(FloatArray, np.asarray(out)).copy())
        return result
