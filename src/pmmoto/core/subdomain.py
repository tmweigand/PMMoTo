"""subdomains.py"""

from . import domain_discretization
from . import orientation
from . import subdomain_features

import numpy as np


class Subdomain(domain_discretization.DiscretizedDomain):
    """
    Parallelization is via decomposition of domain into subdomains
    """

    def __init__(
        self,
        rank: int,
        index: tuple[int, int, int],
        start: tuple[int, int, int],
        num_subdomains: int,
        domain_voxels: tuple[int, int, int],
        neighbor_ranks={},
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.rank = rank
        self.index = index
        self.num_subdomains = num_subdomains
        self.start = start
        self.domain_voxels = domain_voxels
        self.neighbor_ranks = neighbor_ranks
        self.periodic = self.periodic_check()
        self.boundary = self.boundary_check()
        self.features = subdomain_features.collect_features(
            self.neighbor_ranks, self.boundary, self.boundaries
        )

    def boundary_check(self) -> bool:
        """
        Determine if subdomain is on a boundary
        """
        boundary = False
        for n_bound in self.boundaries:
            if n_bound != -1:
                boundary = True

        return boundary

        # faces = [None] * orientation.num_faces
        # edges = [None] * orientation.num_edges
        # corners = [None] * orientation.num_corners

        # ### Faces
        # for n_face in range(0, orientation.num_faces):
        #     feature_index = orientation.faces[n_face]["ID"]
        #     neighbor_proc = self.neighbor_ranks[feature_index]

        #     _periodic = False
        #     if self.boundaries[n_face] == 2:
        #         _periodic = True

        #     faces[n_face] = subdomain_features.Face(
        #         n_face, neighbor_proc, self.boundary, _periodic
        #     )

        # ### Edges
        # for n_edge in range(0, orientation.num_edges):
        #     feature_index = orientation.edges[n_edge]["ID"]
        #     neighbor_proc = self.neighbor_ranks[feature_index]

        #     external_faces = []
        #     periodic_faces = [False, False, False]
        #     for n, n_face in enumerate(orientation.edges[n_edge]["faceIndex"]):

        #         if self.boundaries[n_face] == 2:
        #             periodic_faces[n] = True

        #         elif self.boundaries[n_face] == 0:
        #             external_faces.append(n_face)

        #     edges[n_edge] = subdomain_features.Edge(
        #         n_edge,
        #         neighbor_proc,
        #         self.boundary,
        #         any(periodic_faces),
        #         periodic_faces,
        #         external_faces,
        #     )

        # ### Corners
        # for n_corner in range(0, orientation.num_corners):
        #     feature_index = orientation.corners[n_corner]["ID"]
        #     neighbor_proc = self.neighbor_ranks[feature_index]

        #     ### Determine if Periodic Corner or Global Boundary Corner
        #     external_faces = []
        #     external_edges = []
        #     periodic_faces = [False, False, False]
        #     for n, n_face in enumerate(orientation.corners[n_corner]["faceIndex"]):
        #         if self.boundaries[n_face] == 2:
        #             periodic_faces[n] = True
        #         elif self.boundaries[n_face] == 0:
        #             external_faces.append(n_face)

        #     for edge in orientation.corners[n_corner]["edgeIndex"]:
        #         if edges[edge].boundary:
        #             external_edges.append(edge)

        #     corners[n_corner] = subdomain_features.Corner(
        #         n_corner,
        #         neighbor_proc,
        #         self.boundary,
        #         any(periodic_faces),
        #         periodic_faces,
        #         external_faces,
        #         external_edges,
        #     )

        # data_out = {"faces": faces, "edges": edges, "corners": corners}

        # return data_out
