"""subdomain_features.py"""

from . import orientation


class CubeFeature(object):
    """
    Base class for holding face,edge,corner information for a subdomain
    """

    def __init__(self, ID, n_proc, boundary, periodic):
        self.ID = ID
        self.n_proc = n_proc
        self.periodic = periodic
        self.boundary = boundary
        self.periodic_correction = (0, 0, 0)
        self.info = None
        self.feature_id = None
        self.opp_info = None
        self.extend = [[0, 0], [0, 0], [0, 0]]


class Face(CubeFeature):
    """
    Face information for a subdomain
    """

    def __init__(self, ID, n_proc, boundary, periodic):
        super().__init__(ID, n_proc, boundary, periodic)
        self.info = orientation.faces[ID]
        self.feature_id = orientation.get_boundary_id(self.info["ID"])
        self.opp_info = orientation.faces[orientation.faces[ID]["oppIndex"]]
        self.periodic_correction = self.get_periodic_correction()

    def get_periodic_correction(self) -> tuple[int, ...]:
        """
        Determine spatial correction factor if periodic
        """
        _period_correction = [0, 0, 0]
        if self.periodic:
            _period_correction[self.info["argOrder"][0]] = self.info["dir"]

        return tuple(_period_correction)


class Edge(CubeFeature):
    """
    Edge information for a subdomain
    """

    def __init__(self, ID, n_proc, boundary, periodic, periodic_faces, external_faces):
        super().__init__(ID, n_proc, boundary, periodic)
        self.external_faces = external_faces
        self.info = orientation.edges[ID]
        self.feature_id = orientation.get_boundary_id(self.info["ID"])
        self.opp_info = orientation.edges[orientation.edges[ID]["oppIndex"]]
        self.periodic_correction = self.get_periodic_correction(periodic_faces)
        self.global_boundary = self.get_global_boundary()

    def get_periodic_correction(self, periodic_faces) -> tuple[int, ...]:
        """
        Determine spatial correction factor if periodic
        """
        _period_correction = [0, 0, 0]
        for n, n_face in enumerate(self.info["faceIndex"]):
            if periodic_faces[n]:
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


class Corner(CubeFeature):
    """
    Corner information for a subdomain
    """

    def __init__(
        self,
        ID,
        n_proc,
        boundary,
        periodic,
        periodic_faces,
        external_faces,
        external_edges,
    ):
        super().__init__(ID, n_proc, boundary, periodic)
        self.external_faces = external_faces
        self.external_edges = external_edges
        self.info = orientation.corners[ID]
        self.feature_id = orientation.get_boundary_id(self.info["ID"])
        self.opp_info = orientation.corners[orientation.corners[ID]["oppIndex"]]
        self.periodic_correction = self.get_periodic_correction(periodic_faces)
        self.global_boundary = self.get_global_boundary()

    def get_periodic_correction(self, periodic_faces) -> tuple[int, ...]:
        """
        Determine spatial correction factor if periodic
        """
        _period_correction = [0, 0, 0]
        for n, n_face in enumerate(self.info["faceIndex"]):
            if periodic_faces[n]:
                _period_correction[orientation.faces[n_face]["argOrder"][0]] = (
                    orientation.faces[n_face]["dir"]
                )

        return tuple(_period_correction)

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
