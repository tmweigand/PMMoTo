"""subdomains.py"""

import numpy as np

from . import orientation
from ..core import domain_discretization
from ..core import subdomain_features


class Subdomain(domain_discretization.DiscretizedDomain):
    """
    Parallelization is via decomposition of domain into subdomains
    """

    def __init__(
        self,
        rank: int,
        decomposed_domain,
    ):
        self.rank = rank
        self.domain = decomposed_domain
        self.index = self.get_index(self.rank, self.domain.subdomains)
        self.voxels = self.get_voxels(
            self.index, self.domain.voxels, self.domain.subdomains
        )
        self.box = self.get_box(self.voxels)
        self.global_boundary = self.get_global_boundary()
        self.neighbor_ranks = self.domain.get_neighbor_ranks(self.index)
        self.boundary_types = self.get_boundary_types(
            self.global_boundary, self.neighbor_ranks
        )
        self.inlet = self.get_inlet()
        self.outlet = self.get_outlet()
        self.start = self.get_start()
        self.periodic = self.periodic_check()
        self.coords = self.get_coords(self.box, self.voxels, self.domain.resolution)
        self.features = subdomain_features.collect_features(
            self.neighbor_ranks,
            self.global_boundary,
            self.boundary_types,
            self.voxels,
            self.inlet,
            self.outlet,
        )

    @staticmethod
    def get_index(rank, subdomains) -> tuple[np.intp, ...]:
        """
        Determine the index of the subdomain
        """
        return np.unravel_index(rank, subdomains)

    @staticmethod
    def get_voxels(index, domain_voxels, subdomains) -> tuple[int, ...]:
        """
        Calculate number of voxels in each subdomain.
        This can be very bad when voxels ~= ranks or something like that
        """
        voxels = [0, 0, 0]
        for dim, ind in enumerate(index):
            sd_voxels, rem_sd_voxels = divmod(domain_voxels[dim], subdomains[dim])
            if ind == subdomains[dim] - 1:
                voxels[dim] = sd_voxels + rem_sd_voxels
            else:
                voxels[dim] = sd_voxels

        return tuple(voxels)

    def get_box(self, voxels):
        """
        Determine the bounding box for each subdomain.
        Note: subdomains are divided such that voxel spacing
        is constant
        """
        box = []
        for dim, ind in enumerate(self.index):
            length = voxels[dim] * self.domain.resolution[dim]
            lower = self.domain.box[dim][0] + length * ind
            if ind == self.domain.subdomains[dim] - 1:
                lower = self.domain.box[dim][1] - length
            upper = lower + length
            box.append((lower, upper))

        return tuple(box)

    def get_global_boundary(self):
        """
        Determine if the features are on the domain boundary.
        For a feature to be on a domain boundary, the following must be true:

            subdomain index must contain either 0 or number of subdomains

        """

        global_boundary = {}
        features = orientation.get_features()
        for feature in features:
            boundary = True
            for ind, f_id, subdomains in zip(
                self.index, feature, self.domain.subdomains
            ):
                ### Conditions for a domain boundary feature
                if (
                    f_id == 0
                    or ((ind == 0) and (f_id == -1))
                    or ((ind == subdomains - 1) and (f_id == 1))
                ):
                    continue
                else:
                    boundary = False

            if boundary:
                global_boundary[feature] = True
            else:
                global_boundary[feature] = False

        return global_boundary

    def get_boundary_types(self, global_boundary, neighbor_ranks):
        """
        Determines the boundary type for each feature in the computational domain.

        Args:
            global_boundary (dict): A dictionary mapping features to boolean values indicating
                                    whether they belong to the global boundary.
            neighbor_ranks (dict): A dictionary mapping features to the neighbor process rank.

        Returns:
            dict: A dictionary mapping each feature to its corresponding boundary type.
                Possible values: "end", "wall", "periodic", or "internal".

        Raises:
            Exception: If an unexpected boundary type is encountered.
        """

        boundary_type = {}
        features = orientation.get_features()
        for feature in features:
            if global_boundary.get(feature, False) or neighbor_ranks[feature] < 0:
                _boundary_type = []
                for ind, f_id, subdomains, boundary_types in zip(
                    self.index,
                    feature,
                    self.domain.subdomains,
                    self.domain.boundary_types,
                ):
                    if (ind == 0) and (f_id == -1):
                        _boundary_type.append(boundary_types[0])
                    elif (ind == subdomains - 1) and (f_id == 1):
                        _boundary_type.append(boundary_types[1])

                _boundary_type.sort()

                if _boundary_type[0] == 0:
                    boundary_type[feature] = "end"
                elif _boundary_type[0] == 1:
                    boundary_type[feature] = "wall"
                elif _boundary_type[0] == 2:
                    boundary_type[feature] = "periodic"
                else:
                    raise Exception
            else:
                boundary_type[feature] = "internal"

        return boundary_type

    def get_inlet(self):
        """
        Determine if subdomain is on inlet.
        Inlet requires the global boundary to be of type 0
        Also change orientation of how inlet is stored here
        """
        dims = len(self.index)
        inlet = np.zeros([dims * 2], dtype=np.uint8)
        for dim, ind in enumerate(self.index):
            if ind == 0 and self.domain.boundary_types[dim][0] == 0:
                inlet[dim * 2] = self.domain.inlet[dim][0]
            if (
                ind == self.domain.subdomains[dim] - 1
                and self.domain.boundary_types[dim][1] == 0
            ):
                inlet[dim * 2 + 1] = self.domain.inlet[dim][1]

        return inlet

    def get_outlet(self):
        """
        Determine if subdomain is on outlet
        Also change orientation of how outlet is stored here
        """
        dims = len(self.index)
        outlet = np.zeros([dims * 2], dtype=np.uint8)
        for dim, ind in enumerate(self.index):
            if ind == 0:
                outlet[dim * 2] = self.domain.outlet[dim][0]
            if ind == self.domain.subdomains[dim] - 1:
                outlet[dim * 2 + 1] = self.domain.outlet[dim][1]

        return outlet

    def get_start(self) -> tuple[int, ...]:
        """
        Determine the start of the subdomain. used for saving as vtk
        Start is the minimum voxel ID
        Args:
            sd_index (tuple[int, int, int]): subdomain index

        Returns:
            tuple[int,...]: start
        """
        _start = [0, 0, 0]

        for dim, _index in enumerate(self.index):
            sd_voxels, _ = divmod(self.domain.voxels[dim], self.domain.subdomains[dim])
            _start[dim] = sd_voxels * _index

        return tuple(_start)

    def set_wall_bcs(self, img):
        """
        If wall boundary conditions are specified, force solid on external boundaries.

        """
        feature_types = ["faces"]
        for feature_type in feature_types:
            for feature_id, feature in self.features[feature_type].items():
                if feature.boundary_type == "wall":
                    img[feature.slice] = 0

        return img

    def get_centroid(self):
        """
        Determine the centroid of a subdomain
        """
        centroid = np.zeros(3, dtype=np.double)
        for dim, side in enumerate(self.box):
            centroid[dim] = side[0] + 0.5 * (side[1] - side[0])

        return centroid

    def get_radius(self):
        """
        Determine the centroid of a subdomain
        """
        diameter = 0
        for dim, side in enumerate(self.box):
            length = side[1] - side[0]
            diameter += length * length

        return np.sqrt(diameter) / 2

    def get_origin(self) -> tuple[float, ...]:
        """
        Determine the domain origin from box

        Returns:
            tuple[float,...]: Domain origin
        """
        origin = [0, 0, 0]
        for n, box_dim in enumerate(self.box):
            origin[n] = box_dim[0]

        return tuple(origin)
