"""Read recorded upstream arrays without importing the reference framework or MLX."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from lung_nodule_mlx._typing import HostArray

FIXTURES = Path(__file__).with_name("fixtures") / "darwin-arm64"


def load(path: Path) -> dict[str, HostArray]:
    """Load a recorded upstream archive."""
    with np.load(path, allow_pickle=False) as saved:
        return dict(saved)


def reference(name: str) -> dict[str, HostArray]:
    """Load a named upstream reference archive."""
    return load(FIXTURES / f"{name}.npz")


def exact(actual: HostArray, expected: HostArray, name: str) -> None:
    """Require matching dtype, shape, and array values."""
    assert actual.dtype == expected.dtype, name
    assert actual.shape == expected.shape, name
    assert np.array_equal(actual, expected), name


def runner_atol(case: str) -> dict[str, float]:
    """Load the measured per-output absolute bounds for a recorded runner case."""
    contract = json.loads((FIXTURES / "runner_parity.json").read_text())
    return {name: float(rule["atol"]) for name, rule in contract["cases"][case].items()}
