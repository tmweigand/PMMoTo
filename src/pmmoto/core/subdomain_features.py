"""subdomain_features.py"""

from . import orientation

__all__ = ["collect_features"]


class Feature(object):
    """
    Base class for holding face,edge,corner information for a subdomain.
    Calling those features for lack of a better term.
    """

    def __init__(self, ID, n_proc, boundary):
        self.ID = ID
        self.n_proc = n_proc
        self.boundary = boundary
        self.periodic = False
        self.periodic_correction = (0, 0, 0)
        self.global_boundary = False
        self.info = None
        self.feature_id = None
        self.opp_info = None
        self.extend = [[0, 0], [0, 0], [0, 0]]


class Face(Feature):
    """
    Face information for a subdomain
    """

    def __init__(self, ID, n_proc, boundary, boundary_type):
        super().__init__(ID, n_proc, boundary)
        self.global_boundary = self.get_global_boundary(boundary_type)
        self.periodic = self.is_periodic(boundary_type)
        self.info = orientation.faces[ID]
        self.feature_id = orientation.get_boundary_id(self.info["ID"])
        self.opp_info = orientation.faces[orientation.faces[ID]["oppIndex"]]
        self.periodic_correction = self.get_periodic_correction()

    def is_periodic(self, boundary_type) -> bool:
        """Determine if the face is a periodic face

        Returns:
            bool: True if periodic
        """
        return boundary_type == 2

    def get_periodic_correction(self) -> tuple[int, ...]:
        """
        Determine spatial correction factor if periodic
        """
        _period_correction = [0, 0, 0]
        if self.periodic:
            _period_correction[self.info["argOrder"][0]] = self.info["dir"]

        return tuple(_period_correction)

    def get_global_boundary(self, boundary_type) -> bool:
        """
        Determine if the face is an external boundary
        """
        return boundary_type > -1


class Edge(Feature):
    """
    Edge information for a subdomain
    Need to distinguish between internal and external edges.
    There are 12 external corners. All others are termed internal
    """

    def __init__(self, ID, n_proc, boundary, boundary_type):
        super().__init__(ID, n_proc, boundary)
        self.periodic = self.is_periodic(boundary_type)
        self.external_faces = self.collect_external_faces(boundary_type)
        self.global_boundary = self.get_global_boundary()
        self.info = orientation.edges[ID]
        self.feature_id = orientation.get_boundary_id(self.info["ID"])
        self.opp_info = orientation.edges[orientation.edges[ID]["oppIndex"]]
        self.periodic_correction = self.get_periodic_correction(boundary_type)

    def is_periodic(self, boundary_type) -> bool:
        """Determine if an edge is a periodic edge

        Returns:
            bool: True if periodic
        """
        periodic_faces = [False, False, False]
        for n, n_face in enumerate(orientation.edges[self.ID]["faceIndex"]):
            if boundary_type[n_face] == 2:
                periodic_faces[n] = True

        return any(periodic_faces)

    def collect_external_faces(self, boundary_type):
        """
        Determine if edges are on an external face with boundary type 0"""
        external_faces = []
        for n_face in orientation.edges[self.ID]["faceIndex"]:
            if boundary_type[n_face] == 0:
                external_faces.append(n_face)

        return external_faces

    def get_periodic_correction(self, boundary_type) -> tuple[int, ...]:
        """
        Determine spatial correction factor if periodic
        """
        _period_correction = [0, 0, 0]
        for n, n_face in enumerate(self.info["faceIndex"]):
            if boundary_type[n_face] == 2:
                _period_correction[orientation.faces[n_face]["argOrder"][0]] = (
                    orientation.faces[n_face]["dir"]
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
        self.periodic = self.is_periodic(boundary_type)
        self.external_faces = self.collect_external_faces(boundary_type)
        self.external_edges = self.collect_external_edges(edges)
        self.info = orientation.corners[ID]
        self.feature_id = orientation.get_boundary_id(self.info["ID"])
        self.opp_info = orientation.corners[orientation.corners[ID]["oppIndex"]]
        self.periodic_correction = self.get_periodic_correction(boundary_type)
        self.global_boundary = self.get_global_boundary()

    def is_periodic(self, boundary_type) -> bool:
        """Determine if a corner is a periodic corner

        Returns:
            bool: True if periodic
        """
        periodic_faces = [False, False, False]
        for n, n_face in enumerate(orientation.corners[self.ID]["faceIndex"]):
            if boundary_type[n_face] == 2:
                periodic_faces[n] = True
        return any(periodic_faces)

    def get_periodic_correction(self, boundary_type) -> tuple[int, ...]:
        """
        Determine spatial correction factor (shift) if periodic
        """
        _period_correction = [0, 0, 0]
        for n_face in self.info["faceIndex"]:
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
        for n_face in orientation.corners[self.ID]["faceIndex"]:
            if boundary_type[n_face] == 0:
                external_faces.append(n_face)

        return external_faces

    def collect_external_edges(self, edges):
        """
        Determine if corners are on an external edge with boundary type 0
        """
        external_edges = []
        for edge in orientation.corners[self.ID]["edgeIndex"]:
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


def collect_features(neighbor_ranks, boundary, boundary_types):
    """
    Collect information for faces, edges, and corners
    """

    faces = {}
    edges = {}
    corners = {}

    ### Faces
    for n_face in range(0, orientation.num_faces):
        neighbor_proc = neighbor_ranks[orientation.faces[n_face]["ID"]]
        faces[n_face] = Face(n_face, neighbor_proc, boundary, boundary_types[n_face])

    ### Edges
    for n_edge in range(0, orientation.num_edges):
        neighbor_proc = neighbor_ranks[orientation.edges[n_edge]["ID"]]
        edges[n_edge] = Edge(n_edge, neighbor_proc, boundary, boundary_types)

    ### Corners
    for n_corner in range(0, orientation.num_corners):
        neighbor_proc = neighbor_ranks[orientation.corners[n_corner]["ID"]]
        corners[n_corner] = Corner(
            n_corner,
            neighbor_proc,
            boundary,
            boundary_types,
            edges,
        )

    data_out = {"faces": faces, "edges": edges, "corners": corners}

    return data_out
