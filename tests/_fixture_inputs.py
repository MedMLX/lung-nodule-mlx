"""Seeded fixture inputs shared by the reference recorder and equivalence tests.

This module contains no output oracle and imports neither implementation.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING

from lung_nodule_mlx._typing import ArrayRuntime

if TYPE_CHECKING:
    from mlx.core import array

import numpy as np

type SourceState = dict[str, np.ndarray[tuple[int, ...], np.dtype[np.float32 | np.int64]]]


def _seed_order(name: str) -> tuple[int, ...]:
    """Keep the original RNG draw order, independent of Torch registration order."""
    parts = name.split(".")
    suffix = {
        "weight": 0,
        "bias": 1,
        "running_mean": 2,
        "running_var": 3,
        "num_batches_tracked": 4,
    }[parts[-1]]
    if parts[:2] == ["feature_extractor", "body"]:
        if parts[2] in {"conv1", "bn1"}:
            return (0, 0, 0, int(parts[2] == "bn1"), suffix)
        stage, block = int(parts[2][-1]), int(parts[3])
        module = parts[4]
        group = (
            int(module[-1]) - 1
            if module.startswith("conv")
            else (int(module[-1]) + 2 if module.startswith("bn") else 6 + int(parts[5]))
        )
        return (0, stage, block, group, suffix)
    if parts[:2] == ["feature_extractor", "fpn"]:
        return (1, int(parts[3]), int(parts[2] == "layer_blocks"), suffix)
    head = int(parts[0] == "regression_head")
    return (2, head, int(parts[2]) if parts[1] == "conv" else 12, suffix)


def seeded_state(shapes: Mapping[str, tuple[int, ...]], seed: int) -> SourceState:
    """Source-layout FP32 weight inputs; no network or numerical output oracle."""
    rng = np.random.default_rng(seed)
    state: SourceState = {}
    for name in sorted(shapes, key=_seed_order):
        shape = shapes[name]
        if name.endswith("num_batches_tracked"):
            state[name] = np.asarray(0, dtype=np.int64)
        elif len(shape) == 5:
            scale = 0.4 / np.sqrt(np.prod(shape[1:]))
            state[name] = rng.normal(0, scale, shape).astype(np.float32)
        elif name.endswith("running_var") or name.endswith("weight"):
            state[name] = rng.uniform(0.8, 1.2, shape).astype(np.float32)
        else:
            state[name] = rng.normal(0, 0.03, shape).astype(np.float32)
    return state


def seeded_weights(
    shapes: Mapping[str, tuple[int, ...]], seed: int, mx: ArrayRuntime
) -> dict[str, array]:
    """Build the fixed released graph without storing its 84 MB of weights."""
    weights: dict[str, array] = {}
    for name, value in seeded_state(shapes, seed).items():
        if name.endswith("num_batches_tracked"):
            continue
        if value.ndim == 5:
            value = value.transpose(0, 2, 3, 4, 1)
        weights[name] = mx.array(np.ascontiguousarray(value))
    mx.eval(weights)
    return weights

