"""orientation.py

Defines orientation and feature indexing utilities for PMMoTo subdomains.
"""

from typing import Literal, Iterator
from dataclasses import dataclass
from itertools import product

__all__ = ["get_boundary_id"]


@dataclass(frozen=True)
class FaceInfo:
    """Metadata for a face in a subdomain feature map."""

    opp: tuple[int, ...]
    arg_order: tuple[int, ...]
    direction: int


@dataclass(frozen=True)
class EdgeInfo:
    """Metadata for an edge in a subdomain feature map."""

    opp: tuple[int, ...]
    faces: tuple[tuple[int, ...], ...]
    direction: tuple[int, ...]


@dataclass(frozen=True)
class CornerInfo:
    """Metadata for a corner in a subdomain feature map."""

    opp: tuple[int, ...]
    faces: tuple[tuple[int, ...], ...]
    edges: tuple[tuple[int, ...], ...]


class FaceEdgeCornerMap:
    """Generate face, edge, and corner maps for 2D and 3D voxel-based geometries."""

    def __init__(self, dim: Literal[3]) -> None:
        """Initialize the feature maps.

        Args:
            dim (Literal[2, 3]): Dimension of the space (2D or 3D).

        Raises:
            AssertionError: If dimension is not 2 or 3.

        """
        assert dim == 3  # Only 3D supported currently
        self.dim = dim
        self.num_faces = 6
        self.num_edges = 12
        self.num_corners = 8
        self.num_features = 26
        self.face_dirs = self._generate_face_dirs()
        self.faces = self._generate_faces()
        self.edges = self._generate_edges()
        self.corners = self._generate_corners()

    @property
    def all_features(
        self,
    ) -> Iterator[tuple[tuple[int, ...], FaceInfo | EdgeInfo | CornerInfo]]:
        yield from self.faces.items()
        yield from self.edges.items()
        yield from self.corners.items()

    def collect_feature_ids(self) -> list[tuple[int, ...]]:
        """Generate a list of feature ids"""
        feature_ids: list[tuple[int, ...]] = []
        for feature_id, _ in self.all_features:
            feature_ids.append(feature_id)
        return feature_ids

    def _generate_face_dirs(self) -> list[tuple[int, ...]]:
        """Generate unit directions for face normals in the specified dimension.

        Returns:
            list[tuple[int, ...]]: List of face directions.

        """
        return [
            d for d in product(*[[-1, 0, 1]] * self.dim) if sum(abs(x) for x in d) == 1
        ]

    def _generate_faces(self) -> dict[tuple[int, ...], FaceInfo]:
        """Generate face map: opposite direction, argument order, orientation.

        Returns:
            A dictionary mapping face direction tuples to FaceInfo objects.

        """
        faces: dict[tuple[int, ...], FaceInfo] = {}
        for d in self.face_dirs:
            idx = tuple(abs(i) for i in d).index(1)

            # Determine argument order based on primary axis
            if self.dim == 3:
                if idx == 0:
                    arg_order = (0, 1, 2)
                elif idx == 1:
                    arg_order = (1, 0, 2)
                else:
                    arg_order = (2, 0, 1)
            else:
                arg_order = tuple(range(self.dim))

            opp = tuple(-x for x in d)
            direction = -1 if any(x > 0 for x in d) else 1

            faces[d] = FaceInfo(opp=opp, arg_order=arg_order, direction=direction)

        return faces

    def _generate_edges(self) -> dict[tuple[int, ...], EdgeInfo]:
        """Generate edge map from combinations of face directions.

        Returns:
            A dictionary mapping edge directions to EdgeInfo metadata.

        """
        edges: dict[tuple[int, ...], EdgeInfo] = {}
        for d in product([-1, 0, 1], repeat=3):
            if sum(abs(x) for x in d) == 2 and d not in edges:
                faces = tuple(
                    fd
                    for fd in self.face_dirs
                    if all(fd[i] == 0 or d[i] == fd[i] for i in range(3))
                )
                direction = tuple(i for i, x in enumerate(d) if x != 0)
                edges[d] = EdgeInfo(
                    opp=tuple(-x for x in d), faces=faces, direction=direction
                )
        return edges

    def _generate_corners(self) -> dict[tuple[int, ...], CornerInfo]:
        """Generate corner map from combinations of face and edge directions in 3D.

        Returns:
            A dictionary mapping corner directions to CornerInfo metadata.

        """
        corners: dict[tuple[int, ...], CornerInfo] = {}
        for x in [-1, 1]:
            for y in [-1, 1]:
                for z in [-1, 1]:
                    corner = (x, y, z)
                    faces = ((x, 0, 0), (0, y, 0), (0, 0, z))
                    edges = ((x, y, 0), (x, 0, z), (0, y, z))
                    corners[corner] = CornerInfo(
                        opp=(-x, -y, -z), faces=faces, edges=edges
                    )
        return corners


def get_boundary_id(boundary_index: tuple[int, ...]) -> int:
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


# Create a shared singleton instance for 3D
FEATURE_MAP = FaceEdgeCornerMap(dim=3)
