"""FP32 MLX graph for MONAI lung_nodule_ct_detection 0.6.10.

The released ResNet50 FPN uses only stages 1 and 2. Spatial axes retain
MONAI's (i, j, k) order; only the channel dimension moves for Metal.
"""

from __future__ import annotations

import itertools
from collections.abc import Callable, Iterator
from math import gcd
from typing import TYPE_CHECKING, Literal, cast

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

    def __init__(
        self,
        weights: dict[str, array],
        *,
        mx: ArrayRuntime,
        conv_output_policy: Literal["bounded", "concatenate"] = "bounded",
    ) -> None:
        """Bind weights and choose bounded slab storage or faster concatenation.

        ``bounded`` fills one output, releasing each completed slab. ``concatenate``
        retains all slabs until concatenation and can use more memory for less copying.
        Both use identical convolution arithmetic and whole-volume normalization.
        """
        if conv_output_policy not in {"bounded", "concatenate"}:
            raise ValueError("conv_output_policy must be bounded or concatenate")
        self.mx = mx
        self.weights = weights
        self.conv_output_policy = conv_output_policy

    def _conv_slabs(
        self, x: array, p: str, stride: tuple[int, int, int]
    ) -> Iterator[tuple[int, int, array]]:
        """Yield ``(start, stop, y)`` slabs of the convolution along the first spatial axis.

        ``y`` holds output rows ``start:stop``, unevaluated, with the bias added.
        Slab boundaries depend only on the input shape, the weight shape and the
        stride: Metal may pick a different FP32 contraction for a different slab
        shape, so every consumer must use these slabs to reproduce the same values.
        """
        mx = self.mx
        w = self.weights[p + ".weight"]
        k = w.shape[1]
        pad = k // 2
        # Bound the explicit conv3d lowering workspace and each output slab to
        # approximately 256 MiB; the output bound applies to 1x1 expansion convs.
        plane = ((x.shape[2] + 2 * pad - k) // stride[1] + 1) * (
            (x.shape[3] + 2 * pad - k) // stride[2] + 1
        )
        budget = 256 * 1024**2 // (4 * plane)
        slab = max(1, min(budget // (k**3 * x.shape[-1]), budget // w.shape[0]))
        length = (x.shape[1] + 2 * pad - k) // stride[0] + 1
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
            yield start, stop, y

    def conv(
        self,
        x: array,
        p: str,
        stride: tuple[int, int, int] = (1, 1, 1),
        epilogue: Callable[[array, int, int], array] | None = None,
    ) -> array:
        """Cubic-kernel 3D convolution of ``x`` with weights ``p + ".weight"``.

        Padding is ``kernel // 2`` per side and the bias, if present, is added.
        The output is computed in slabs along the first spatial axis.
        ``epilogue(y, start, stop)``, if given, maps each slab (output rows
        ``start:stop``) before it is stored. It must be elementwise, such as batch
        norm, ReLU or a residual add, so that no full-volume temporary is built;
        per-element arithmetic is then the same as applying it to the whole output.
        """
        mx = self.mx
        w = self.weights[p + ".weight"]
        k = w.shape[1]
        pad = k // 2
        length = (x.shape[1] + 2 * pad - k) // stride[0] + 1
        chunks: list[array] = []
        out = None
        slabs = self._conv_slabs(x, p, stride)
        first = next(slabs)
        if first[1] < length and self.conv_output_policy == "bounded":
            spatial = tuple((x.shape[a + 1] + 2 * pad - k) // stride[a] + 1 for a in range(3))
            out = mx.zeros((x.shape[0], *spatial, w.shape[0]), dtype=x.dtype)
        for start, stop, y in itertools.chain((first,), slabs):
            if epilogue is not None:
                y = epilogue(y, start, stop)
            mx.eval(y)
            if out is None:
                chunks.append(y)
            else:
                # Evaluate before the next slab so old destination buffers can be donated.
                out[:, start:stop] = y
                mx.eval(out)
        if out is not None:
            return out
        out = mx.concatenate(chunks, axis=1) if len(chunks) > 1 else chunks[0]
        mx.eval(out)
        return out

    def batch_norm(self, x: array, p: str) -> array:
        """Inference-mode batch norm from the stored running statistics (eps 1e-5)."""
        w, mx = self.weights, self.mx
        scale = w[p + ".weight"] * mx.rsqrt(w[p + ".running_var"] + 1e-5)
        return (x - w[p + ".running_mean"]) * scale + w[p + ".bias"]

    def _bn_epilogue(
        self, p: str, *, relu: bool = True, residual: array | None = None
    ) -> Callable[[array, int, int], array]:
        """Conv epilogue for one slab: batch norm ``p``, the residual's rows, then ReLU.

        The operations and their order match the whole-volume graph they replace.
        """

        def apply(y: array, start: int, stop: int) -> array:
            y = self.batch_norm(y, p)
            if residual is not None:
                y = y + residual[:, start:stop]
            return self.mx.maximum(y, 0) if relu else y

        return apply

    def stem(self, x: array) -> array:
        """conv1, bn1, ReLU and MaxPool3d(3, 2, 1), one conv slab at a time.

        Each slab of conv1 output is normalized, rectified and pooled over the
        second and third spatial axes, then stored in a buffer whose first and
        last rows along the first axis are negative infinity; pooling that buffer
        along the first axis completes the 3x3x3 window. Max is exact, so this
        equals pooling the whole normalized volume with negative-infinity padding,
        while holding a quarter of the conv1 output instead of three full copies.
        """
        mx = self.mx
        p = "feature_extractor.body"
        k = self.weights[p + ".conv1.weight"].shape[1]
        stride = (2, 2, 1)
        length, height, width = (
            (x.shape[a + 1] + 2 * (k // 2) - k) // stride[a] + 1 for a in range(3)
        )
        pooled_height, pooled_width = (height - 1) // 2 + 1, (width - 1) // 2 + 1
        channels = self.weights[p + ".conv1.weight"].shape[0]
        shape = (x.shape[0], length + 2, pooled_height, pooled_width, channels)
        rows = mx.full(shape, -float("inf"), dtype=x.dtype)
        for start, stop, y in self._conv_slabs(x, p + ".conv1", stride):
            y = mx.maximum(self.batch_norm(y, p + ".bn1"), 0)
            y = mx.pad(y, [(0, 0), (0, 0), (1, 1), (1, 1), (0, 0)], constant_values=-float("inf"))
            plane = None
            for a in range(3):
                for b in range(3):
                    part = y[:, :, a : a + 2 * pooled_height : 2, b : b + 2 * pooled_width : 2]
                    plane = part if plane is None else mx.maximum(plane, part)
            # Evaluate before the next slab so the old buffer can be donated.
            rows[:, start + 1 : stop + 1] = cast("array", plane)
            mx.eval(rows)
        count = (length - 1) // 2 + 1
        pooled = mx.maximum(
            mx.maximum(rows[:, 0 : 2 * count : 2], rows[:, 1 : 1 + 2 * count : 2]),
            rows[:, 2 : 2 + 2 * count : 2],
        )
        mx.eval(pooled)
        return pooled

    def features(self, x: array) -> list[array]:
        """Run the ResNet50 stem and stages 1 and 2; return both stage outputs.

        Batch norm, ReLU and the residual add run as conv epilogues, slab by slab,
        so no full-volume normalization temporaries are built.
        """
        p = "feature_extractor.body"
        x = self.stem(x)
        stages: list[array] = []
        for stage, blocks in ((1, 3), (2, 4)):
            for block in range(blocks):
                q = f"{p}.layer{stage}.{block}"
                stride = (2, 2, 2) if stage == 2 and block == 0 else (1, 1, 1)
                residual = (
                    self.conv(
                        x,
                        q + ".downsample.0",
                        stride,
                        self._bn_epilogue(q + ".downsample.1", relu=False),
                    )
                    if block == 0
                    else x
                )
                y = self.conv(x, q + ".conv1", epilogue=self._bn_epilogue(q + ".bn1"))
                y = self.conv(y, q + ".conv2", stride, self._bn_epilogue(q + ".bn2"))
                x = self.conv(
                    y, q + ".conv3", epilogue=self._bn_epilogue(q + ".bn3", residual=residual)
                )
            stages.append(x)
        return stages

    def pyramid(self, stages: list[array]) -> list[array]:
        """Feature pyramid over the two stages; returns three maps, finest first.

        The third map is a stride-2 subsample of the coarsest one (the FPN's extra
        level). Empties ``stages`` so each stage output can be released once its
        lateral conv has run.
        """
        mx = self.mx
        p = "feature_extractor.fpn"
        fine, coarse = stages
        stages.clear()
        inner = self.conv(coarse, p + ".inner_blocks.1")
        del coarse
        low = self.conv(inner, p + ".layer_blocks.1")
        # Nearest-neighbor resize of the coarse map to the lateral grid: source index
        # floor(i * in / out) on each axis. The lateral 1x1 conv keeps the fine grid.
        indices = [
            mx.arange(fine.shape[axis]) * inner.shape[axis] // fine.shape[axis]
            for axis in (1, 2, 3)
        ]

        def add_top_down(lateral: array, start: int, stop: int) -> array:
            """Add the resized coarse map's rows ``start:stop`` to one lateral slab."""
            rows = mx.take(inner, indices[0][start:stop], axis=1)
            return lateral + mx.take(mx.take(rows, indices[1], axis=2), indices[2], axis=3)

        merged = self.conv(fine, p + ".inner_blocks.0", epilogue=add_top_down)
        del fine
        high = self.conv(merged, p + ".layer_blocks.0")
        return [high, low, low[:, ::2, ::2, ::2]]

    def group_norm_relu(self, x: array, p: str) -> array:
        """GroupNorm (8 groups, eps 1e-5) with affine parameters ``p``, then ReLU.

        Returns a new array filled chunk by chunk. Each group's mean and variance average
        1024-sample (or smaller, when the group size requires) contiguous blocks of
        its samples in source order (position-major, then the group's 32 channels),
        then average the block results: one long FP32 reduction over a full window's
        50M samples per group drifts beyond the source tolerance. Blocks are formed
        from depth chunks of whole blocks, at least 4 per group so the block
        reduction keeps the same Metal kernel as an unchunked pass. The arithmetic
        per sample matches a whole-volume pass, without its transposed full-volume
        copies.
        """
        mx = self.mx
        n, depth, height, width, channels = x.shape
        groups = 8
        size = channels // groups
        plane = height * width * size
        block = gcd(depth * plane, 1024)
        # Chunks start on block boundaries, hold at least 4 blocks and stay near 64 MiB.
        unit = block // gcd(block, plane)
        rows = max(-(-4 * block // plane), (64 * 1024**2) // (4 * height * width * channels))
        rows = -(-rows // unit) * unit
        spans = [(start, min(start + rows, depth)) for start in range(0, depth, rows)]
        if len(spans) > 1 and (spans[-1][1] - spans[-1][0]) * plane < 4 * block:
            spans[-2:] = [(spans[-2][0], depth)]

        def blocks(start: int, stop: int) -> array:
            """Samples of rows ``start:stop`` as ``(N, groups, blocks, block)``."""
            part = mx.reshape(x[:, start:stop], (n, -1, groups, size))
            return mx.reshape(mx.transpose(part, (0, 2, 1, 3)), (n, groups, -1, block))

        def block_mean(values: Callable[[int, int], array]) -> array:
            """Mean over blocks of per-block means; shape ``(N, groups, 1)``."""
            means: list[array] = []
            for start, stop in spans:
                means.append(mx.mean(values(start, stop), axis=-1))
                mx.eval(means[-1])
            return mx.mean(mx.concatenate(means, axis=-1), axis=-1, keepdims=True)

        mean = block_mean(blocks)
        variance = block_mean(lambda start, stop: mx.square(blocks(start, stop) - mean[..., None]))
        shift = mx.reshape(mean, (n, 1, groups, 1))
        scale = mx.reshape(mx.rsqrt(variance + 1e-5), (n, 1, groups, 1))
        weight, bias = self.weights[p + ".weight"], self.weights[p + ".bias"]
        out = mx.zeros(x.shape, dtype=x.dtype)
        for start, stop in spans:
            part = (mx.reshape(x[:, start:stop], (n, -1, groups, size)) - shift) * scale
            part = mx.reshape(part, (n, stop - start, height, width, channels))
            part = mx.maximum(part * weight + bias, 0)
            mx.eval(part)
            # Evaluate before the next chunk so old destination buffers can be donated.
            out[:, start:stop] = part
            mx.eval(out)
        return out

    def head(self, x: array, name: str) -> array:
        """Run one prediction head on a pyramid map.

        ``name`` is ``"classification"`` or ``"regression"``. Four blocks of
        conv, GroupNorm (8 groups of 32 channels, eps 1e-5, per-channel affine)
        and ReLU precede the output conv, which has 3 channels for classification
        (one logit per anchor) and 18 for regression (six deltas per anchor).
        """
        for i in range(4):
            p = f"{name}_head.conv"
            # Rebind before normalizing so the block input is released first.
            x = self.conv(x, f"{p}.{i * 3}")
            x = self.group_norm_relu(x, f"{p}.{i * 3 + 1}")
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
