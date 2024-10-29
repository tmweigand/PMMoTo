"""subdomain_features.py"""

import numpy as np
from . import orientation

__all__ = ["collect_features", "get_feature_voxels"]


class Feature(object):
    """
    Base class for holding features: {face, edge, corner} information for a subdomain.
    This is the main abstraction for handling boundary conditions and parallel communication
    """

    def __init__(self, ID, n_proc, boundary):
        self.ID = ID
        self.n_proc = n_proc
        self.boundary = boundary
        self.periodic = False
        self.periodic_correction = (0, 0, 0)
        self.global_boundary = False
        self.info = None
        self.opp_info = None
        self.extend = [[0, 0], [0, 0], [0, 0]]
        self.loop = None


class Face(Feature):
    """
    Face information for a subdomain
    """

    def __init__(self, ID, n_proc, boundary, boundary_type=None):
        super().__init__(ID, n_proc, boundary)
        self.info = orientation.faces[ID]
        self.global_boundary = self.get_global_boundary(boundary_type)
        self.periodic = self.is_periodic(boundary)
        self.opp_info = orientation.faces[ID]["opp"]
        self.periodic_correction = self.get_periodic_correction()

    def is_periodic(self, boundary_type) -> bool:
        """Determine if the face is a periodic face

        Returns:
            bool: True if periodic
        """
        return boundary_type == "periodic"

    def get_periodic_correction(self) -> tuple[int, ...]:
        """
        Determine spatial correction factor if periodic
        """
        _period_correction = [0, 0, 0]
        if self.periodic:
            _period_correction[self.info["argOrder"][0]] = self.info["dir"]

        return tuple(_period_correction)

    def get_global_boundary(self, boundary) -> bool:
        """
        Determine if the face is an external boundary
        """

        return boundary


class Edge(Feature):
    """
    Edge information for a subdomain
    Need to distinguish between internal and external edges.
    There are 12 external corners. All others are termed internal
    """

    def __init__(self, ID, n_proc, boundary, boundary_type):
        super().__init__(ID, n_proc, boundary)
        self.info = orientation.edges[ID]
        self.periodic = self.is_periodic(boundary_type)
        self.external_faces = self.collect_external_faces(boundary_type)
        self.global_boundary = self.get_global_boundary()
        if self.periodic:
            self.opp_info = self.set_opposite_feature(boundary_type)
        else:
            self.opp_info = orientation.edges[orientation.edges[ID]["opp"]]
        self.periodic_correction = self.get_periodic_correction(boundary_type)

    def set_opposite_feature(self, boundary_type):
        """
        Args:
            boundary_type (_type_): _description_
        """
        opp_feature_id = [0, 0, 0]
        for n in [0, 1, 2]:
            if self.ID[n] != 0 and boundary_type[n * 2] == 2:
                opp_feature_id[n] = -self.ID[n]
            elif self.ID[n] != 0 and boundary_type[n * 2] != 2:
                opp_feature_id[n] = self.ID[n]

        # return orientation.edges[orientation.edges[_id]["oppIndex"]]

    def is_periodic(self, boundary_type) -> bool:
        """Determine if an edge is a periodic edge

        Returns:
            bool: True if periodic
        """
        return boundary_type == "periodic"

    def collect_external_faces(self, boundary_type):
        """
        Determine if edges are on an external face with boundary type 0"""
        external_faces = []
        for face in orientation.edges[self.ID]["faces"]:
            if face in boundary_type:
                if boundary_type[face] == "end":
                    external_faces.append(face)

        return external_faces

    def get_periodic_correction(self, boundary_type) -> tuple[int, ...]:
        """
        Determine spatial correction factor if periodic
        """
        _period_correction = [0, 0, 0]
        for face in self.info["faces"]:
            if face in boundary_type:
                if boundary_type[face] == "periodic":
                    _period_correction[orientation.faces[face]["argOrder"][0]] = (
                        orientation.faces[face]["dir"]
                    )

        return tuple(_period_correction)

    def get_global_boundary(self) -> bool:
        """
        Determine if the edge is an external boundary
        """
        global_boundary = False
        if len(self.external_faces) == 2:
            global_boundary = True
        return global_boundary

    def get_extension(self, extend_domain, bounds):
        """
        Determine the span of the feature based on extend
        """
        _faces = orientation.edges[self.ID]["ID"]
        for n, f in enumerate(_faces):
            if f > 0:
                self.extend[n][0] = bounds[n][-1] - extend_domain[n]
                self.extend[n][1] = bounds[n][-1]
            elif f < 0:
                self.extend[n][0] = bounds[n][0]
                self.extend[n][1] = bounds[n][0] + extend_domain[n]
            else:
                self.extend[n][0] = 0
                self.extend[n][1] = 0


class Corner(Feature):
    """
    Corner information for a subdomain.
    Need to distinguish between internal and external corners.
    There are 8 external corners. All others are termed internal
    """

    def __init__(self, ID, n_proc, boundary, boundary_type, edges):
        super().__init__(ID, n_proc, boundary)
        self.info = orientation.corners[ID]
        self.periodic = self.is_periodic(boundary_type)
        self.external_faces = self.collect_external_faces(boundary_type)
        self.external_edges = self.collect_external_edges(edges)
        self.opp_info = orientation.corners[ID]["opp"]
        self.periodic_correction = self.get_periodic_correction(boundary_type)
        self.global_boundary = self.get_global_boundary()

    def is_periodic(self, boundary_type) -> bool:
        """Determine if a corner is a periodic corner

        Returns:
            bool: True if periodic
        """
        return boundary_type[self.ID] == "periodic"

    def get_periodic_correction(self, boundary_type) -> tuple[int, ...]:
        """
        Determine spatial correction factor (shift) if periodic
        """
        _period_correction = [0, 0, 0]
        for n_face in self.info["faces"]:
            if boundary_type[n_face] == 2:
                _period_correction[orientation.faces[n_face]["argOrder"][0]] = (
                    orientation.faces[n_face]["dir"]
                )

        return tuple(_period_correction)

    def collect_external_faces(self, boundary_type):
        """
        Determine if corners are on an external face with boundary type 0
        """
        external_faces = []
        for n_face in orientation.corners[self.ID]["faces"]:
            if boundary_type[n_face] == 0:
                external_faces.append(n_face)

        return external_faces

    def collect_external_edges(self, edges):
        """
        Determine if corners are on an external edge with boundary type 0
        """
        external_edges = []
        for edge in orientation.corners[self.ID]["edges"]:
            if edges[edge].boundary:
                external_edges.append(edge)

        return external_edges

    def get_global_boundary(self) -> bool:
        """
        Determine if the edge is an external boundary
        """
        global_boundary = False
        if len(self.external_faces) == 3 or len(self.external_edges) == 3:
            global_boundary = True
        return global_boundary

    def get_extension(self, extend_domain, bounds):
        """
        Determine the span of the feature based on extend
        """
        faces = orientation.corners[self.ID]["ID"]
        for n, f in enumerate(faces):
            if f > 0:
                self.extend[n][0] = bounds[n][-1] - extend_domain[n]
                self.extend[n][1] = bounds[n][-1]
            elif f < 0:
                self.extend[n][0] = bounds[n][0]
                self.extend[n][1] = bounds[n][0] + extend_domain[n]
            else:
                self.extend[n][0] = 0
                self.extend[n][1] = 0


def collect_features(neighbor_ranks, boundaries, boundary_types, voxels):
    """
    Collect information for faces, edges, and corners
    """

    faces = {}
    edges = {}
    corners = {}

    ### Faces
    for feature in orientation.faces.keys():
        if feature in boundary_types:
            faces[feature] = Face(
                feature,
                neighbor_ranks[feature],
                boundaries[feature],
                boundary_types[feature],
            )
        else:
            faces[feature] = Face(
                feature,
                neighbor_ranks[feature],
                boundaries[feature],
            )
        faces[feature].loop = get_feature_voxels(feature, voxels)

    ### Edges
    for feature in orientation.edges.keys():
        edges[feature] = Edge(
            feature, neighbor_ranks[feature], boundaries[feature], boundary_types
        )
        edges[feature].loop = get_feature_voxels(feature, voxels)

    ### Corners
    for feature in orientation.corners.keys():
        corners[feature] = Corner(
            feature,
            neighbor_ranks[feature],
            boundaries[feature],
            boundary_types,
            edges,
        )
        corners[feature].loop = get_feature_voxels(feature, voxels)

    data_out = {"faces": faces, "edges": edges, "corners": corners}

    return data_out


def set_padding(features, voxels, pad, reservoir_pad=0):
    """
    If the subdomain is padded, set the padding for the features
    """

    feature_types = ["faces", "edges", "corners"]
    for feature_type in feature_types:
        for feature_id, feature in features[feature_type].items():
            feature.loop = get_feature_voxels(feature_id, voxels, pad)

    return features


def get_feature_voxels(feature_id, voxels, pad=None):
    """_summary_

    Args:
        grid (_type_): _description_
        pad (_type_, optional): _description_. Defaults to None.

    Returns:
        _type_: _description_
    """
    if pad is None:
        pad = [[0, 0], [0, 0], [0, 0]]

    loop = np.zeros([3, 2], dtype=np.uint64)

    for n, length in enumerate(voxels):
        loop[n, 1] = length
        if feature_id[n] == -1:
            loop[n] = [0, pad[n][0] + 1]
        elif feature_id[n] == 1:
            loop[n] = [length - pad[n][1] - 1, length]
        else:
            loop[n] = [pad[n][0] + 1, length - pad[n][1] - 1]

    return loop
