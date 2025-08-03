"""octants.py

Octants are a 2x2x2 neighborhood for local calculations.
"""

import itertools

__all__ = ["generate_octant_mapping", "feature_octants", "get_neighbors_vertices"]


def generate_octant_mapping() -> dict[tuple[int, ...], int]:
    """Generate the octant id labels."""
    octants = list(itertools.product([-1, 1], repeat=3))
    octant_mapping = {}
    for octant_idx, oct in enumerate(octants):
        octant_mapping[oct] = octant_idx
    return octant_mapping


def feature_octants(feature: tuple[int, ...]) -> list[int]:
    """Determine available octants for input feature."""
    allowable_octants: list[int] = []
    octants = generate_octant_mapping()
    for signs, oct_idx in octants.items():
        match = True
        for f, s in zip(feature, signs):
            if f == 0:
                continue
            elif f == 1 and s != -1:
                match = False
                break
            elif f == -1 and s != 1:
                match = False
                break
        if match:
            allowable_octants.append(oct_idx)

    return allowable_octants


def get_neighbors_vertices(
    index: tuple[int, ...] = (0, 0, 0),
    boundary_face: None | tuple[int, ...] | list[int] = None,
):
    """Determine the vertices for neighborhoods.

    Given an index indicating the relative position (-1, 0, 1) along each axis,
    compute which neighbor vertices should be considered. This accounts for
    edge and corner cases by adjusting which ranges are iterated over.

    If boundary_face is specified, only the vertices on the face are provided.

    Args:
        index: The boundary index
        boundary_face: The boundary face index

    """
    if boundary_face is None:
        boundary_face = (0, 0, 0)

    # Can only extract on faces
    assert sum(abs(_id) for _id in boundary_face) < 2

    neighbor_vertices = []
    loop = [[-1, 1] for _ in index]

    for n, (ind, b_face) in enumerate(zip(index, boundary_face)):
        if ind == -1:
            loop[n][0] = 0
        if b_face == -1:
            loop[n][1] = 0
        elif ind == 1:
            loop[n][1] = 0
        if b_face == 1:
            loop[n][0] = 0

    for dx in range(loop[0][0], loop[0][1] + 1):
        for dy in range(loop[1][0], loop[1][1] + 1):
            for dz in range(loop[2][0], loop[2][1] + 1):
                idx = (dx + 1) * 9 + (dy + 1) * 3 + (dz + 1)
                neighbor_vertices.append(idx)

    return neighbor_vertices
