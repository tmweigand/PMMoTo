"""orientation.py

Defines orientation and feature indexing utilities for PMMoTo subdomains.
"""

import numpy as np

__all__ = ["get_boundary_id"]


def get_boundary_id(boundary_index):
    """Determine boundary ID from a boundary index.

    Args:
        boundary_index (list[int] or tuple[int]): Boundary index for [x, y, z],
        values -1, 0, 1.

    Returns:
        int: Encoded boundary ID.

    """
    params = [[0, 9, 18], [0, 3, 6], [0, 1, 2]]

    id_ = 0
    for n in range(0, 3):
        if boundary_index[n] < 0:
            id_ += params[n][0]
        elif boundary_index[n] > 0:
            id_ += params[n][1]
        else:
            id_ += params[n][2]

    return id_


def add_faces(boundary_features):
    """Add face indices for edges and corners in boundary_features.

    Since loop_info are by face, need to add face index for edges and corners in case
    edge and corner n_procs are < 0 but face is valid.

    Args:
        boundary_features (list[bool]): List of boundary feature flags.

    """
    for n in range(0, num_features):
        if boundary_features[n]:
            for nn in allFaces[n]:
                boundary_features[nn] = True


num_faces = 6
num_edges = 12
num_corners = 8
num_features = 26

faces = {
    (-1, 0, 0): {
        "opp": (1, 0, 0),
        "argOrder": np.array([0, 1, 2], dtype=np.uint8),
        "dir": 1,
    },
    (1, 0, 0): {
        "opp": (-1, 0, 0),
        "argOrder": np.array([0, 1, 2], dtype=np.uint8),
        "dir": -1,
    },
    (0, -1, 0): {
        "opp": (0, 1, 0),
        "argOrder": np.array([1, 0, 2], dtype=np.uint8),
        "dir": 1,
    },
    (0, 1, 0): {
        "opp": (0, -1, 0),
        "argOrder": np.array([1, 0, 2], dtype=np.uint8),
        "dir": -1,
    },
    (0, 0, -1): {
        "opp": (0, 0, 1),
        "argOrder": np.array([2, 0, 1], dtype=np.uint8),
        "dir": 1,
    },
    (0, 0, 1): {
        "opp": (0, 0, -1),
        "argOrder": np.array([2, 0, 1], dtype=np.uint8),
        "dir": -1,
    },
}

edges = {
    (-1, 0, -1): {
        "opp": (1, 0, 1),
        "faces": ((-1, 0, 0), (0, 0, -1)),
        "dir": (0, 2),
    },
    (-1, 0, 1): {
        "opp": (1, 0, -1),
        "faces": ((-1, 0, 0), (0, 0, 1)),
        "dir": (0, 2),
    },
    (-1, -1, 0): {
        "opp": (1, 1, 0),
        "faces": ((-1, 0, 0), (0, -1, 0)),
        "dir": (0, 1),
    },
    (-1, 1, 0): {
        "opp": (1, -1, 0),
        "faces": ((-1, 0, 0), (0, 1, 0)),
        "dir": (0, 1),
    },
    (1, 0, -1): {
        "opp": (-1, 0, 1),
        "faces": ((1, 0, 0), (0, 0, -1)),
        "dir": (0, 2),
    },
    (1, 0, 1): {
        "opp": (-1, 0, -1),
        "faces": ((1, 0, 0), (0, 0, 1)),
        "dir": (0, 2),
    },
    (1, -1, 0): {
        "opp": (-1, 1, 0),
        "faces": ((1, 0, 0), (0, -1, 0)),
        "dir": (0, 1),
    },
    (1, 1, 0): {
        "opp": (-1, -1, 0),
        "faces": ((1, 0, 0), (0, 1, 0)),
        "dir": (0, 1),
    },
    (0, -1, -1): {
        "opp": (0, 1, 1),
        "faces": ((0, -1, 0), (0, 0, -1)),
        "dir": (1, 2),
    },
    (0, -1, 1): {
        "opp": (0, 1, -1),
        "faces": ((0, -1, 0), (0, 0, 1)),
        "dir": (1, 2),
    },
    (0, 1, -1): {
        "opp": (0, -1, 1),
        "faces": ((0, 1, 0), (0, 0, -1)),
        "dir": (1, 2),
    },
    (0, 1, 1): {
        "opp": (0, -1, -1),
        "faces": ((0, 1, 0), (0, 0, 1)),
        "dir": (1, 2),
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


features = {}
features["faces"] = faces
features["edges"] = edges
features["corners"] = corners


def get_features():
    """Return a list of all feature IDs (faces, edges, corners).

    Returns:
        list[tuple[int, int, int]]: List of feature IDs.

    """
    features = [
        (-1, 0, 0),
        (1, 0, 0),
        (0, -1, 0),
        (0, 1, 0),
        (0, 0, -1),
        (0, 0, 1),
        (-1, 0, -1),
        (-1, 0, 1),
        (-1, -1, 0),
        (-1, 1, 0),
        (1, 0, -1),
        (1, 0, 1),
        (1, -1, 0),
        (1, 1, 0),
        (0, -1, -1),
        (0, -1, 1),
        (0, 1, -1),
        (0, 1, 1),
        (-1, -1, -1),
        (-1, -1, 1),
        (-1, 1, -1),
        (-1, 1, 1),
        (1, -1, -1),
        (1, -1, 1),
        (1, 1, -1),
        (1, 1, 1),
    ]

    return features


directions = {
    0: {"ID": [-1, -1, -1], "index": 0, "opp_index": 25},
    1: {"ID": [-1, -1, 0], "index": 1, "opp_index": 24},
    2: {"ID": [-1, -1, 1], "index": 2, "opp_index": 23},
    3: {"ID": [-1, 0, -1], "index": 3, "opp_index": 22},
    4: {"ID": [-1, 0, 0], "index": 4, "opp_index": 21},
    5: {"ID": [-1, 0, 1], "index": 5, "opp_index": 20},
    6: {"ID": [-1, 1, -1], "index": 6, "opp_index": 19},
    7: {"ID": [-1, 1, 0], "index": 7, "opp_index": 18},
    8: {"ID": [-1, 1, 1], "index": 8, "opp_index": 17},
    9: {"ID": [0, -1, -1], "index": 9, "opp_index": 16},
    10: {"ID": [0, -1, 0], "index": 10, "opp_index": 15},
    11: {"ID": [0, -1, 1], "index": 11, "opp_index": 14},
    12: {"ID": [0, 0, -1], "index": 12, "opp_index": 13},
    13: {"ID": [0, 0, 1], "index": 13, "opp_index": 12},
    14: {"ID": [0, 1, -1], "index": 14, "opp_index": 11},
    15: {"ID": [0, 1, 0], "index": 15, "opp_index": 10},
    16: {"ID": [0, 1, 1], "index": 16, "opp_index": 9},
    17: {"ID": [1, -1, -1], "index": 17, "opp_index": 8},
    18: {"ID": [1, -1, 0], "index": 18, "opp_index": 7},
    19: {"ID": [1, -1, 1], "index": 19, "opp_index": 6},
    20: {"ID": [1, 0, -1], "index": 20, "opp_index": 5},
    21: {"ID": [1, 0, 0], "index": 21, "opp_index": 4},
    22: {"ID": [1, 0, 1], "index": 22, "opp_index": 3},
    23: {"ID": [1, 1, -1], "index": 23, "opp_index": 2},
    24: {"ID": [1, 1, 0], "index": 24, "opp_index": 1},
    25: {"ID": [1, 1, 1], "index": 25, "opp_index": 0},
}

allFaces = [
    [0, 2, 6, 8, 18, 20, 24],  # 0
    [1, 2, 7, 8, 19, 20, 25],  # 1
    [2, 8, 20],  # 2
    [3, 5, 6, 8, 21, 23, 24],  # 3
    [4, 5, 7, 8, 22, 23, 25],  # 4
    [5, 8, 23],  # 5
    [6, 8, 24],  # 6
    [7, 8, 25],  # 7
    [8],  # 8
    [9, 11, 15, 17, 18, 20, 24],  # 9
    [10, 11, 16, 17, 19, 20, 25],  # 10
    [11, 17, 20],  # 11
    [12, 14, 15, 17, 21, 23, 24],  # 12
    [13, 14, 16, 17, 22, 23, 25],  # 13
    [14, 17, 23],  # 14
    [15, 17, 24],  # 15
    [16, 17, 25],  # 16
    [17],  # 17
    [18, 20, 24],  # 18
    [19, 20, 25],  # 19
    [20],  # 20
    [21, 23, 24],  # 21
    [22, 23, 25],  # 22
    [23],  # 23
    [24],  # 24
    [25],
]  # 25


def get_index_ordering(inlet, outlet):
    """Rearrange the loop_info ordering so the inlet and outlet faces are first.

    Args:
        inlet (tuple): Inlet flags for each face.
        outlet (tuple): Outlet flags for each face.

    Returns:
        list[int]: New ordering of axes.

    """
    order = [0, 1, 2]
    for n in range(0, 3):
        if inlet[n * 2] or outlet[n * 2] or inlet[n * 2 + 1] or outlet[n * 2 + 1]:
            order.remove(n)
            order.insert(0, n)

    return order
