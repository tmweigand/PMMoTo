"""subdomain_features.py

Defines feature classes and utilities for handling subdomain faces, edges, and corners,
including boundary and periodicity information.
"""

from __future__ import annotations
from typing import TYPE_CHECKING, Iterator, Any
import numpy as np

from .boundary_types import BoundaryType, boundary_order
from .features import Face, Edge, Corner
from .orientation import FEATURE_MAP

if TYPE_CHECKING:
    from .subdomain import Subdomain
    from .subdomain_padded import PaddedSubdomain


class SubdomainFeatures:
    """Container for all features (faces, edges, corners) of a subdomain."""

    def __init__(
        self,
        subdomain: Subdomain | PaddedSubdomain,
        voxels: tuple[int, ...],
        pad: tuple[tuple[int, int], ...] = ((0, 0), (0, 0), (0, 0)),
    ):
        self.subdomain = subdomain
        self.pad = pad
        self.faces: dict[tuple[int, ...], Face] = {}
        self.edges: dict[tuple[int, ...], Edge] = {}
        self.corners: dict[tuple[int, ...], Corner] = {}
        self.collect_features()
        self.set_feature_voxels(voxels)

    @property
    def all_features(
        self,
    ) -> Iterator[tuple[tuple[int, ...], Face | Edge | Corner]]:
        """Iterates through features"""
        yield from self.faces.items()
        yield from self.edges.items()
        yield from self.corners.items()

    def set_feature_voxels(self, voxels: tuple[int, ...]) -> None:
        """Set the own and neighbor voxels for each feature"""
        for group in (self.faces, self.edges, self.corners):
            for feature in group.values():
                feature.get_voxels(voxels, self.pad)

    def get_features(self) -> dict[tuple[int, ...], Face | Edge | Corner]:
        """Return a dict of all features"""
        out_features: dict[tuple[int, ...], Face | Edge | Corner] = {}
        for group in (self.faces, self.edges, self.corners):
            for feature in group.values():
                out_features[feature.feature_id] = feature
        return out_features

    def collect_features(self) -> None:
        """Collect information for faces, edges, and corners for a subdomain."""
        self._collect_faces()
        self._collect_edges()
        self._collect_corners()

    def _collect_faces(self) -> None:
        """Collect Face"""
        ### Faces
        for feature in FEATURE_MAP.faces:
            face_dim = np.nonzero(feature)[0][0]
            if feature[face_dim] > 0:
                side = 1
            else:
                side = 0

            self.faces[feature] = Face(
                feature_id=feature,
                neighbor_rank=self.subdomain.neighbor_ranks[feature],
                boundary_type=self.get_boundary_type(
                    feature, self.subdomain.neighbor_ranks[feature]
                ),
                global_boundary=self.get_global_boundary(feature),
                inlet=self.subdomain.inlet[face_dim][side],
                outlet=self.subdomain.outlet[face_dim][side],
            )

    def _collect_edges(self) -> None:
        """Collect Edges"""
        for feature in FEATURE_MAP.edges:
            self.edges[feature] = Edge(
                feature_id=feature,
                neighbor_rank=self.subdomain.neighbor_ranks[feature],
                boundary_type=self.get_boundary_type(
                    feature, self.subdomain.neighbor_ranks[feature]
                ),
                global_boundary=self.get_global_boundary(feature),
            )

    def _collect_corners(self) -> None:
        """Collect Corners"""
        for feature in FEATURE_MAP.corners:
            self.corners[feature] = Corner(
                feature_id=feature,
                neighbor_rank=self.subdomain.neighbor_ranks[feature],
                boundary_type=self.get_boundary_type(
                    feature, self.subdomain.neighbor_ranks[feature]
                ),
                global_boundary=self.get_global_boundary(feature),
            )

    def get_global_boundary(self, feature_id: tuple[int, ...]) -> bool:
        """Determine if a feature is on the domain boundary.

        For a feature to be on a domain boundary, the subdomain index must contain
        either 0 or the number of subdomains.

        Returns:
            bool: If feature is on domain boundary

        """
        for ind, f_id, subdomains in zip(
            self.subdomain.index,
            feature_id,
            self.subdomain.domain.subdomains,
        ):
            ### Conditions for a domain boundary feature
            if (
                f_id == 0
                or ((ind == 0) and (f_id == -1))
                or ((ind == subdomains - 1) and (f_id == 1))
            ):
                continue
            else:
                return False

        return True

    def get_boundary_type(
        self, feature_id: tuple[int, ...], neighbor_rank: int
    ) -> BoundaryType:
        """Determine the boundary type for a feature.

        For a feature to be on a domain boundary, the subdomain index must contain
        either 0 or the number of subdomains.

        Returns:
            bool: If feature is on domain boundary

        """
        if not self.get_global_boundary(feature_id) or neighbor_rank == 0:
            return BoundaryType.INTERNAL

        boundary_type = []
        for ind, f_id, subdomains, boundary_types in zip(
            self.subdomain.index,
            feature_id,
            self.subdomain.domain.subdomains,
            self.subdomain.boundary_types,
        ):

            if (ind == 0) and (f_id == -1):
                boundary_type.append(boundary_types[0])
            elif (ind == subdomains - 1) and (f_id == 1):
                boundary_type.append(boundary_types[1])

        if not boundary_type:
            return BoundaryType.INTERNAL

        return boundary_order(boundary_type)

    def collect_periodic_features(self) -> list[tuple[int, ...]]:
        """Collect all periodic features from a features dictionary.

        Returns:
            list: List of feature_ids that are periodic.

        """
        periodic_features = []
        for group in (self.faces, self.edges, self.corners):
            for feature in group.values():
                if feature.periodic:
                    periodic_features.append(feature.feature_id)

        return periodic_features

    def collect_periodic_corrections(self) -> dict[tuple[int, ...], tuple[int, ...]]:
        """Collect periodic correction vectors for all periodic features.

        Args:
            features (dict): Dictionary with keys "faces", "edges", "corners".

        Returns:
            dict: Mapping from feature_id to periodic correction tuple.

        """
        per_corrections = {}
        for group in (self.faces, self.edges, self.corners):
            for feature in group.values():
                if feature.periodic:
                    per_corrections[feature.feature_id] = feature.periodic_correction

        return per_corrections

    def get_feature_member(self, feature_id: tuple[int, ...], member_name: str) -> Any:
        """Get the value of a member from the Face, Edge, or Corner with feature_id.

        Args:
            feature_id (tuple[int, ...]): The feature ID.
            member_name (str): Name of the member to retrieve.

        Returns:
            Any: Value of the requested member.

        Raises:
            KeyError: If feature_id is not found.
            AttributeError: If member_name does not exist.

        """
        feature_obj: Face | Edge | Corner
        if feature_id in self.faces:
            feature_obj = self.faces[feature_id]
        elif feature_id in self.edges:
            feature_obj = self.edges[feature_id]
        elif feature_id in self.corners:
            feature_obj = self.corners[feature_id]
        else:
            raise KeyError(f"Feature ID {feature_id} not found in features.")

        try:
            return getattr(feature_obj, member_name)
        except AttributeError:
            raise AttributeError(
                f"Feature {feature_id} ({type(feature_obj).__name__})"
                "has no member '{member_name}'."
            )
