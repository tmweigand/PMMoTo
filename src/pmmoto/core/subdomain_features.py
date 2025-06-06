"""subdomain_features.py

Defines feature classes and utilities for handling subdomain faces, edges, and corners,
including boundary and periodicity information.
"""

from __future__ import annotations
from typing import TYPE_CHECKING
import numpy as np

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

    def set_feature_voxels(self, voxels: tuple[int, ...]) -> None:
        """Set the own and neighbor voxels for each feature"""
        for group in (self.faces, self.edges, self.corners):
            for feature in group.values():
                feature.get_voxels(voxels, self.pad)

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
                boundary_type=self.subdomain.feature_boundary_types[feature],
                global_boundary=self.subdomain.global_boundary[feature],
                inlet=self.subdomain.inlet[face_dim][side],
                outlet=self.subdomain.outlet[face_dim][side],
            )

    def _collect_edges(self) -> None:
        """Collect Edges"""
        for feature in FEATURE_MAP.edges:
            self.edges[feature] = Edge(
                feature_id=feature,
                neighbor_rank=self.subdomain.neighbor_ranks[feature],
                boundary_type=self.subdomain.feature_boundary_types[feature],
                global_boundary=self.subdomain.global_boundary[feature],
            )

    def _collect_corners(self) -> None:
        """Collect Corners"""
        for feature in FEATURE_MAP.corners:
            self.corners[feature] = Corner(
                feature_id=feature,
                neighbor_rank=self.subdomain.neighbor_ranks[feature],
                boundary_type=self.subdomain.feature_boundary_types[feature],
                global_boundary=self.subdomain.global_boundary[feature],
            )

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


# def get_feature_voxels(
#     feature_id: tuple[int, ...],
#     voxels: tuple[int, ...],
#     boundary_type: Optional[str] = None,
#     pad: Optional[Tuple[Tuple[int, int], ...]] = None,
# ) -> dict[str, np.ndarray]:
#     """Compute voxel index ranges for a feature face and optional neighbor region."""
#     ndim = len(voxels)
#     pad_array = np.array(pad) if pad else np.zeros((ndim, 2), dtype=int)
#     padded = np.any(pad_array)

#     own = np.zeros((ndim, 2), dtype=np.uint64)
#     neighbor = np.zeros((ndim, 2), dtype=np.uint64) if padded else None

#     for i in range(ndim):
#         f_id = feature_id[i]
#         length = voxels[i]
#         p_before, p_after = pad_array[i]

#         if f_id == -1:
#             own[i] = compute_lower_face(length, p_before, boundary_type)
#             if padded:
#                 neighbor[i] = compute_lower_neighbor(length, p_before, boundary_type)

#         elif f_id == 1:
#             own[i] = compute_upper_face(length, p_after, boundary_type)
#             if padded:
#                 neighbor[i] = compute_upper_neighbor(length, p_after, boundary_type)

#         else:  # f_id == 0
#             own[i] = [p_before, length - p_after]
#             if padded:
#                 neighbor[i] = [p_before, length - p_after]

#     result = {"own": own}
#     if padded:
#         result["neighbor"] = neighbor
#     return result


# def get_feature_voxels(feature_id, voxels, boundary_type=None, pad=None):
#     """Compute feature-specific voxel ranges with optional padding.

#     This function generates voxel ranges for a given feature based on the `feature_id`,
#     `voxels`, and an optional `pad` parameter. It returns a dictionary with ranges for
#     "own" voxels and, if applicable, "neighbor" voxels.

#     Args:
#         feature_id (tuple[int]): Feature identifiers for each axis (-1, 0, or 1).
#         voxels (tuple[int]): Lengths of the voxel dimensions along each axis.
#         boundary_type (str, optional): Boundary type ("wall", "periodic", etc.).
#         pad (tuple[tuple[int, int]], optional): Padding for each axis as
#             ((before, after), ...).

#     Returns:
#         dict: Dictionary with keys:
#             - "own": np.ndarray of shape (3, 2) with voxel ranges for the feature's own
#                 voxels.
#             - "neighbor" (optional): np.ndarray of shape (3, 2) with voxel ranges for
#                 neighboring voxels if padding is specified.

#     Notes:
#         - If `pad` is None or contains only zeros, the function assumes no padding.
#         - The `feature_id` determines how voxel ranges are computed for each axis:
#           - -1: Specifies the start of the axis.
#           - 0: Includes the entire range (or adjusted range if padding exists).
#           - 1: Specifies the end of the axis.
#         - For wall boundary conditions, the walls are stored as neighbor.

#     """
#     # Determine if padding is active
#     padded = pad is not None and np.any(pad)

#     # Default pad values
#     if not padded:
#         pad = np.zeros((len(voxels), 2), dtype=int)

#     # Initialize loop dictionary
#     loop = {"own": np.zeros((len(voxels), 2), dtype=np.uint64)}
#     if padded:
#         loop["neighbor"] = np.zeros((len(voxels), 2), dtype=np.uint64)

#     for n, length in enumerate(voxels):

#         if feature_id[n] == -1:
#             lower_pad = pad[n][0]
#             if lower_pad != 0:
#                 if boundary_type == "wall":
#                     loop["own"][n] = [1, 2]
#                 else:
#                     loop["own"][n] = [lower_pad, lower_pad * 2]
#             else:
#                 loop["own"][n] = [0, 1]
#             if "neighbor" in loop:
#                 if boundary_type == "wall":
#                     loop["neighbor"][n] = [0, 1]
#                 else:
#                     loop["neighbor"][n] = [0, lower_pad]

#         elif feature_id[n] == 1:
#             upper_pad = pad[n][1]
#             if upper_pad != 0:
#                 if boundary_type == "wall":
#                     loop["own"][n] = [length - 2, length - 1]
#                 else:
#                     loop["own"][n] = [length - upper_pad * 2, length - upper_pad]
#             else:
#                 loop["own"][n] = [length - 1, length]
#             if "neighbor" in loop:
#                 if boundary_type == "wall":
#                     loop["neighbor"][n] = [length - 1, length]
#                 else:
#                     loop["neighbor"][n] = [length - upper_pad, length]

#         else:
#             if padded:
#                 loop["own"][n] = [pad[n][0], length - pad[n][1]]
#             else:
#                 loop["own"][n] = [0, length]
#             if "neighbor" in loop:
#                 loop["neighbor"][n] = [pad[n][0], length - pad[n][1]]

#     return loop


# def collect_periodic_features(features):
#     """Collect all periodic features from a features dictionary.

#     Args:
#         features (dict): Dictionary with keys "faces", "edges", "corners".

#     Returns:
#         list: List of feature_ids that are periodic.

#     """
#     periodic_features = []
#     feature_types = ["faces", "edges", "corners"]
#     for feature_type in feature_types:
#         for feature_id, feature in features[feature_type].items():
#             if feature.periodic:
#                 periodic_features.append(feature_id)

#     return periodic_features


# def collect_periodic_corrections(features):
#     """Collect periodic correction vectors for all periodic features.

#     Args:
#         features (dict): Dictionary with keys "faces", "edges", "corners".

#     Returns:
#         dict: Mapping from feature_id to periodic correction tuple.

#     """
#     periodic_corrections = {}
#     feature_types = ["faces", "edges", "corners"]
#     for feature_type in feature_types:
#         for feature_id, feature in features[feature_type].items():
#             if feature.periodic:
#                 periodic_corrections[feature_id] = feature.periodic_correction

#     return periodic_corrections
