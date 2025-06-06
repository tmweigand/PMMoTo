"""test_orientation.py"""

from typing import Any
from collections.abc import Iterable
from collections import Counter
from dataclasses import asdict, is_dataclass
import numpy as np
import pmmoto


faces = {
    (-1, 0, 0): {
        "opp": (1, 0, 0),
        "arg_order": np.array([0, 1, 2], dtype=np.uint8),
        "direction": 1,
    },
    (1, 0, 0): {
        "opp": (-1, 0, 0),
        "arg_order": np.array([0, 1, 2], dtype=np.uint8),
        "direction": -1,
    },
    (0, -1, 0): {
        "opp": (0, 1, 0),
        "arg_order": np.array([1, 0, 2], dtype=np.uint8),
        "direction": 1,
    },
    (0, 1, 0): {
        "opp": (0, -1, 0),
        "arg_order": np.array([1, 0, 2], dtype=np.uint8),
        "direction": -1,
    },
    (0, 0, -1): {
        "opp": (0, 0, 1),
        "arg_order": np.array([2, 0, 1], dtype=np.uint8),
        "direction": 1,
    },
    (0, 0, 1): {
        "opp": (0, 0, -1),
        "arg_order": np.array([2, 0, 1], dtype=np.uint8),
        "direction": -1,
    },
}


edges = {
    (-1, 0, -1): {
        "opp": (1, 0, 1),
        "faces": ((-1, 0, 0), (0, 0, -1)),
        "direction": (0, 2),
    },
    (-1, 0, 1): {
        "opp": (1, 0, -1),
        "faces": ((-1, 0, 0), (0, 0, 1)),
        "direction": (0, 2),
    },
    (-1, -1, 0): {
        "opp": (1, 1, 0),
        "faces": ((-1, 0, 0), (0, -1, 0)),
        "direction": (0, 1),
    },
    (-1, 1, 0): {
        "opp": (1, -1, 0),
        "faces": ((-1, 0, 0), (0, 1, 0)),
        "direction": (0, 1),
    },
    (1, 0, -1): {
        "opp": (-1, 0, 1),
        "faces": ((1, 0, 0), (0, 0, -1)),
        "direction": (0, 2),
    },
    (1, 0, 1): {
        "opp": (-1, 0, -1),
        "faces": ((1, 0, 0), (0, 0, 1)),
        "direction": (0, 2),
    },
    (1, -1, 0): {
        "opp": (-1, 1, 0),
        "faces": ((1, 0, 0), (0, -1, 0)),
        "direction": (0, 1),
    },
    (1, 1, 0): {
        "opp": (-1, -1, 0),
        "faces": ((1, 0, 0), (0, 1, 0)),
        "direction": (0, 1),
    },
    (0, -1, -1): {
        "opp": (0, 1, 1),
        "faces": ((0, -1, 0), (0, 0, -1)),
        "direction": (1, 2),
    },
    (0, -1, 1): {
        "opp": (0, 1, -1),
        "faces": ((0, -1, 0), (0, 0, 1)),
        "direction": (1, 2),
    },
    (0, 1, -1): {
        "opp": (0, -1, 1),
        "faces": ((0, 1, 0), (0, 0, -1)),
        "direction": (1, 2),
    },
    (0, 1, 1): {
        "opp": (0, -1, -1),
        "faces": ((0, 1, 0), (0, 0, 1)),
        "direction": (1, 2),
    },
}

corners = {
    (-1, -1, -1): {
        "opp": (1, 1, 1),
        "faces": ((-1, 0, 0), (0, -1, 0), (0, 0, -1)),
        "edges": ((-1, 0, -1), (-1, -1, 0), (0, -1, -1)),
    },
    (-1, -1, 1): {
        "opp": (1, 1, -1),
        "faces": ((-1, 0, 0), (0, -1, 0), (0, 0, 1)),
        "edges": ((-1, 0, 1), (-1, -1, 0), (0, -1, 1)),
    },
    (-1, 1, -1): {
        "opp": (1, -1, 1),
        "faces": ((-1, 0, 0), (0, 1, 0), (0, 0, -1)),
        "edges": ((-1, 0, -1), (-1, 1, 0), (0, 1, -1)),
    },
    (-1, 1, 1): {
        "opp": (1, -1, -1),
        "faces": ((-1, 0, 0), (0, 1, 0), (0, 0, 1)),
        "edges": ((-1, 0, 1), (-1, 1, 0), (0, 1, 1)),
    },
    (1, -1, -1): {
        "opp": (-1, 1, 1),
        "faces": ((1, 0, 0), (0, -1, 0), (0, 0, -1)),
        "edges": ((1, 0, -1), (1, -1, 0), (0, -1, -1)),
    },
    (1, -1, 1): {
        "opp": (-1, 1, -1),
        "faces": ((1, 0, 0), (0, -1, 0), (0, 0, 1)),
        "edges": ((1, 0, 1), (1, -1, 0), (0, -1, 1)),
    },
    (1, 1, -1): {
        "opp": (-1, -1, 1),
        "faces": ((1, 0, 0), (0, 1, 0), (0, 0, -1)),
        "edges": ((1, 0, -1), (1, 1, 0), (0, 1, -1)),
    },
    (1, 1, 1): {
        "opp": (-1, -1, -1),
        "faces": ((1, 0, 0), (0, 1, 0), (0, 0, 1)),
        "edges": ((1, 0, 1), (1, 1, 0), (0, 1, 1)),
    },
}


def dicts_equal(d1: Any, d2: Any) -> bool:
    """Recursively compares dictionaries or dataclass-like structures for equality."""
    # Convert dataclass instances or other objects with __dict__ to dict
    if is_dataclass(d1):
        d1 = asdict(d1)
    elif hasattr(d1, "__dict__"):
        d1 = d1.__dict__

    if is_dataclass(d2):
        d2 = asdict(d2)
    elif hasattr(d2, "__dict__"):
        d2 = d2.__dict__

    # Check both are now dictionaries
    if not isinstance(d1, dict) or not isinstance(d2, dict):
        return d1 == d2

    if d1.keys() != d2.keys():
        return False

    for k in d1:
        v1 = d1[k]
        v2 = d2[k]
        if isinstance(v1, np.ndarray) and isinstance(v2, np.ndarray):
            if not np.array_equal(v1, v2):
                return False
        elif isinstance(v1, dict) and isinstance(v2, dict):
            if not dicts_equal(v1, v2):
                return False
        elif isinstance(v1, Iterable) and not isinstance(v1, (str, bytes)):
            if Counter(v1) != Counter(v2):
                return False
        elif v1 != v2:
            return False
    return True


def test_3d_features_faces() -> None:
    """Test for 3D features."""
    features = pmmoto.core.orientation.FaceEdgeCornerMap(dim=3)
    pm_keys = sorted(features.faces.keys())
    ref_keys = sorted(faces.keys())
    assert (
        pm_keys == ref_keys
    ), f"Key mismatch:\nfeatures: {pm_keys}\nreference: {ref_keys}"

    # Compare values
    for key in pm_keys:
        assert dicts_equal(
            features.faces[key], faces[key]
        ), f"Value mismatch at {key}:\n{features.faces[key]}\n!=\n{faces[key]}"


def test_3d_features_edges() -> None:
    """Test for 3D features."""
    features = pmmoto.core.orientation.FaceEdgeCornerMap(dim=3)
    pm_keys = sorted(features.edges.keys())
    ref_keys = sorted(edges.keys())
    assert (
        pm_keys == ref_keys
    ), f"Key mismatch:\nfeatures: {pm_keys}\nreference: {ref_keys}"

    # Compare values
    for key in pm_keys:
        assert dicts_equal(
            features.edges[key], edges[key]
        ), f"Value mismatch at {key}:\n{features.edges[key]}\n!=\n{edges[key]}"


def test_3d_features_corners() -> None:
    """Test for 3D features."""
    features = pmmoto.core.orientation.FaceEdgeCornerMap(dim=3)
    pm_keys = sorted(features.corners.keys())
    ref_keys = sorted(corners.keys())
    assert (
        pm_keys == ref_keys
    ), f"Key mismatch:\nfeatures: {pm_keys}\nreference: {ref_keys}"

    # Compare values
    for key in pm_keys:
        assert dicts_equal(
            features.corners[key], corners[key]
        ), f"Value mismatch at {key}:\n{features.corners[key]}\n!=\n{corners[key]}"
