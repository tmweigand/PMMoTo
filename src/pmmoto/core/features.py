"""features.py

This holds features of a cube: faces edges, and corners
"""

from __future__ import annotations
from typing import TYPE_CHECKING
import numpy as np
from numpy.typing import NDArray
from .boundary_types import BoundaryType
from .orientation import get_boundary_id
from .orientation import FEATURE_MAP

if TYPE_CHECKING:
    from .orientation import FaceInfo, EdgeInfo, CornerInfo


class Feature(object):
    """Base class for holding feature: {face, edge, corner} information.

    This is the main abstraction for handling boundary conditions
    and parallel communication.
    """

    def __init__(
        self,
        dim: int,
        feature_id: tuple[int, ...],
        neighbor_rank: int,
        boundary_type: BoundaryType,
        global_boundary: bool | None = None,
    ):
        """Initialize a Feature.

        Args:
            dim: dimension
            feature_id: Feature identifier (tuple).
            neighbor_rank: Neighboring process rank.
            boundary_type: Boundary type (e.g., "wall", "periodic").

            global_boundary: Whether this is a global boundary (bool).

        """
        self.dim = dim
        self.feature_id = feature_id
        self.neighbor_rank = neighbor_rank
        self.boundary_type = boundary_type
        self.global_boundary = global_boundary
        self.periodic = False
        self.periodic_correction: tuple[int, ...] = (0, 0, 0)
        self.extend: tuple[tuple[int, int], ...] = ((0, 0), (0, 0), (0, 0))
        self.own: NDArray[np.uint64] = np.zeros((self.dim, 2), dtype=np.uint64)
        self.neighbor: NDArray[np.uint64] = np.zeros((self.dim, 2), dtype=np.uint64)

    def convert_feature_id(self, index: tuple[int, ...] | None = None) -> int:
        """Convert the feature type to id.

        Returns:
            int: feature id

        """
        if index is None:
            index = self.feature_id

        return get_boundary_id(index)

    def is_periodic(self, boundary_type: BoundaryType) -> bool:
        """Determine if a feature is periodic

        Returns:
            bool: True if periodic

        """
        return boundary_type == BoundaryType.PERIODIC

    def get_voxels(
        self,
        voxels: tuple[int, ...],
        pad: tuple[tuple[int, int], ...] = ((0, 0), (0, 0), (0, 0)),
    ) -> None:
        """Determine the voxel indices for the feature"""
        padded = np.any(pad)
        for i in range(self.dim):
            f_id = self.feature_id[i]
            length = voxels[i]
            p_before, p_after = pad[i]
            if f_id == -1:
                self.own[i] = self.compute_lower_face(p_before)
                if padded:
                    self.neighbor[i] = self.compute_lower_neighbor(p_before)
            elif f_id == 1:
                self.own[i] = self.compute_upper_face(length, p_after)
                if padded:
                    self.neighbor[i] = self.compute_upper_neighbor(length, p_after)

            else:  # f_id == 0
                self.own[i] = [p_before, length - p_after]
                if padded:
                    self.neighbor[i] = [p_before, length - p_after]

    def compute_lower_face(self, lower_pad: int) -> list[int]:
        """Determine lower face voxels"""
        if lower_pad and self.boundary_type != BoundaryType.WALL:
            return [lower_pad, lower_pad * 2]
        elif lower_pad and self.boundary_type == BoundaryType.WALL:
            return [1, 2]
        else:
            return [0, 1]

    def compute_upper_face(
        self,
        length: int,
        upper_pad: int,
    ) -> list[int]:
        """Determine upper face voxels"""
        if upper_pad and self.boundary_type != BoundaryType.WALL:
            return [length - upper_pad * 2, length - upper_pad]
        elif upper_pad and self.boundary_type == BoundaryType.WALL:
            return [length - 2, length - 1]
        else:
            return [length - 1, length]

    def compute_lower_neighbor(self, lower_pad: int) -> list[int]:
        """Determine lower neighbor face voxels"""
        if self.boundary_type == BoundaryType.WALL:
            return [0, 1]
        else:
            return [0, lower_pad]

    def compute_upper_neighbor(
        self,
        length: int,
        upper_pad: int,
    ) -> list[int]:
        """Determine upper neighbor face voxels"""
        if self.boundary_type == BoundaryType.WALL:
            return [length - 1, length]
        else:
            return [length - upper_pad, length]


class Face(Feature):
    """Face information for a subdomain."""

    def __init__(
        self,
        feature_id: tuple[int, ...],
        neighbor_rank: int,
        boundary_type: BoundaryType,
        global_boundary: bool | None = None,
        inlet: bool | None = None,
        outlet: bool | None = None,
    ):
        """Initialize a Face feature.

        Args:
            feature_id: Feature identifier (tuple).
            neighbor_rank: Neighboring process rank.
            boundary_type: Boundary type.
            global_boundary: Whether this is a global boundary (bool).
            inlet: Whether this face is an inlet (bool).
            outlet: Whether this face is an outlet (bool).

        """
        assert feature_id in FEATURE_MAP.faces
        super().__init__(
            FEATURE_MAP.dim,
            feature_id,
            neighbor_rank,
            boundary_type,
            global_boundary,
        )
        self.info: FaceInfo = FEATURE_MAP.faces[feature_id]
        self.global_boundary = global_boundary
        self.periodic = self.is_periodic(boundary_type)
        self.inlet = inlet
        self.outlet = outlet
        self.periodic_correction = self.get_periodic_correction()
        self.forward = self.get_direction()
        self.slice = self.get_slice()

    def get_periodic_correction(self) -> tuple[int, ...]:
        """Determine spatial correction factor if periodic.

        Returns:
            A tuple of correction values for each spatial dimension.

        """
        _period_correction: list[int] = [0 for _ in range(FEATURE_MAP.dim)]
        if self.periodic:
            # Ensure .arg_order[0] and .direction are ints
            idx = int(self.info.arg_order[0])
            val = int(self.info.direction)
            _period_correction[idx] = val

        return tuple(_period_correction)

    def get_strides(self, strides: tuple[int, ...]) -> int:
        """Return the stride corresponding to the active dimension in the feature ID.

        This method is used to identify which stride (among the three dimensions)
        is relevant based on the nonzero entry in `self.feature_id`.

        Args:
            strides (Tuple[int, int, int]): A 3-element tuple representing the strides
                of a NumPy array, typically from `.strides`.

        Returns:
            int: The stride value corresponding to the active axis.

        """
        for feature_id, stride in zip(self.feature_id, strides):
            if feature_id != 0:
                return stride

        raise ValueError("All feature_id components are zero; no valid stride found.")

    def get_direction(self) -> bool:
        """Determine if the face is point in the forward direction.

            -1 for the non-negative feature id

        Returns:
            _type_: _description_

        """
        forward = True
        for feature_id in self.feature_id:
            if feature_id == 1:
                forward = False
        return forward

    def get_slice(
        self,
    ) -> tuple[int | slice, ...]:
        """Extract a 2D slice index for accessing a boundary face in a 3D array.

        The method inspects `self.feature_id`, which defines the orientation of a face
        in 3D space. It returns a tuple with one fixed index (either 0 or -1) to select
        a specific boundary face, and `slice(None)` for the remaining dimensions.

        Returns:
            Tuple[int | slice, ... ]:
                A 3-element tuple used to index a 2D face from a 3D array.

        """
        face_slice: list[int | slice] = [slice(None) for _ in range(3)]
        for dim, f_id in enumerate(self.feature_id):
            if f_id == -1:
                face_slice[dim] = 0
            elif f_id == 1:
                face_slice[dim] = -1

        return tuple(face_slice)

    def map_to_index(self) -> int:
        """Return the 1d-index for face."""
        for dim, f_id in enumerate(self.feature_id):
            if f_id < 0:
                return dim * 2
            elif f_id > 0:
                return dim * 2 + 1

        raise ValueError(
            "Feature ID does not correspond to a face (no nonzero entries)."
        )


class Edge(Feature):
    """Edge information for a subdomain.

    Need to distinguish between internal and external edges.
    There are 12 external corners. All others are termed internal.
    """

    def __init__(
        self,
        feature_id: tuple[int, ...],
        neighbor_rank: int,
        boundary_type: BoundaryType,
        global_boundary: bool | None = None,
    ):
        """Initialize an Edge feature.

        Args:
            feature_id: Feature identifier (tuple).
            neighbor_rank: Neighboring process rank.
            boundary_type: Boundary type.
            global_boundary: Whether this is a global boundary (bool).

        """
        assert feature_id in FEATURE_MAP.edges
        super().__init__(
            FEATURE_MAP.dim, feature_id, neighbor_rank, boundary_type, global_boundary
        )
        self.info: EdgeInfo = FEATURE_MAP.edges[feature_id]
        self.periodic = self.is_periodic(boundary_type)
        self.global_boundary = global_boundary
        self.periodic_correction = self.get_periodic_correction()

    def get_periodic_correction(self) -> tuple[int, ...]:
        """Determine spatial correction factor if periodic"""
        _period_correction = [0, 0, 0]
        for face in self.info.faces:
            if self.boundary_type == "periodic":
                _period_correction[FEATURE_MAP.faces[face].arg_order[0]] = (
                    FEATURE_MAP.faces[face].direction
                )

        return tuple(_period_correction)


class Corner(Feature):
    """Corner information for a subdomain.

    Need to distinguish between internal and external corners.
    There are 8 external corners. All others are termed internal.
    """

    def __init__(
        self,
        feature_id: tuple[int, ...],
        neighbor_rank: int,
        boundary_type: BoundaryType,
        global_boundary: bool | None = None,
    ):
        """Initialize a Corner feature.

        Args:
            feature_id: Feature identifier (tuple).
            neighbor_rank: Neighboring process rank.
            boundary_type: Boundary type.
            global_boundary: Whether this is a global boundary (bool).

        """
        super().__init__(
            FEATURE_MAP.dim, feature_id, neighbor_rank, boundary_type, global_boundary
        )
        self.info: CornerInfo = FEATURE_MAP.corners[feature_id]
        self.periodic = self.is_periodic(boundary_type)

        self.periodic_correction = self.get_periodic_correction()
        self.global_boundary = global_boundary

    def get_periodic_correction(self) -> tuple[int, ...]:
        """Determine spatial correction factor (shift) if periodic"""
        _period_correction = [0, 0, 0]
        for n_face in self.info.faces:
            if self.boundary_type == "periodic":
                _period_correction[FEATURE_MAP.faces[n_face].arg_order[0]] = (
                    FEATURE_MAP.faces[n_face].direction
                )

        return tuple(_period_correction)
