"""subdomains.py"""

import numpy as np
from . import domain_discretization
from . import orientation
from . import subdomain_features


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
        self.index = self.get_index(self.domain.subdomains)
        self.voxels = self.get_voxels(
            self.index, self.domain.voxels, self.domain.subdomains
        )
        self.box = self.get_box(
            self.index,
            self.voxels,
            self.domain.box,
            self.domain.resolution,
            self.domain.subdomains,
        )
        self.boundaries, self.boundary_types = self.get_boundaries(
            self.index, self.domain.boundary_types, self.domain.subdomains
        )
        self.inlet = self.get_inlet(
            self.index, self.domain.inlet, self.domain.subdomains
        )
        self.outlet = self.get_outlet(
            self.index, self.domain.outlet, self.domain.subdomains
        )
        self.start = self.get_start(
            self.index, self.domain.voxels, self.domain.subdomains
        )
        self.neighbor_ranks = self.domain.get_neighbor_ranks(self.index)
        self.periodic = self.periodic_check()
        self.boundary = self.boundary_check()
        self.coords = self.get_coords(self.box, self.voxels, self.domain.resolution)
        self.features = subdomain_features.collect_features(
            self.neighbor_ranks,
            self.boundaries,
            self.boundary_types,
            self.voxels,
            self.inlet,
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

    def get_voxels(
        self,
        index: tuple[np.intp, ...],
        domain_voxels: tuple[int, int, int],
        subdomains,
    ) -> tuple[int, ...]:
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

    def get_index(self, subdomains) -> tuple[np.intp, ...]:
        """
        Determine the index of the subdomain
        """
        print(self.rank, subdomains)
        return np.unravel_index(self.rank, subdomains)

    def get_box(
        self,
        index: tuple[np.intp, ...],
        voxels: tuple[int, int, int],
        domain_box,
        resolution: tuple[int, int, int],
        subdomains: tuple[int, int, int],
    ):
        """
        Determine the bounding box for each subdomain.
        Note: subdomains are divided such that voxel spacing
        is constant
        """
        box = []
        for dim, ind in enumerate(index):
            length = voxels[dim] * resolution[dim]
            lower = domain_box[dim][0] + length * ind
            if ind == subdomains[dim] - 1:
                lower = domain_box[dim][1] - length
            upper = lower + length
            box.append((lower, upper))

        return tuple(box)

    def get_boundaries(
        self,
        index: tuple[np.intp, ...],
        domain_boundaries,
        subdomains: tuple[int, int, int],
    ):
        """
        Determine the boundary types for each subdomain and feature in subdomain
        How to handle edges and corners? Now, assume  periodic -> end -> wall is the priority ranking
        to account for non-fully periodic domains and issues with padding and pass the edges and corners
        """
        boundaries = {}
        boundary_type = {}
        feature_types = ["faces", "edges", "corners"]
        for feature_type in feature_types:
            features = orientation.features[feature_type].keys()
            for feature in features:
                boundary = True
                _boundary_type = []
                for dim, (ind, f_id) in enumerate(zip(index, feature)):
                    if f_id == 0:
                        continue
                    elif (ind == 0) and (f_id == -1):
                        _boundary_type.append(domain_boundaries[dim][0])
                    elif (ind == subdomains[dim] - 1) and (f_id == 1):
                        _boundary_type.append(domain_boundaries[dim][1])
                    else:
                        boundary = False

                if boundary:
                    boundaries[feature] = True

                    if 2 in _boundary_type:
                        boundary_type[feature] = "periodic"
                    elif 0 in _boundary_type:
                        boundary_type[feature] = "end"
                    else:
                        boundary_type[feature] = "wall"
                else:
                    boundaries[feature] = False
                    boundary_type[feature] = "internal"

        return boundaries, boundary_type

    def get_inlet(
        self, index: tuple[np.intp, ...], domain_inlet, subdomains: tuple[int, int, int]
    ):
        """
        Determine if subdomain is on inlet
        Also change orientation of how inlet is stored here
        """
        dims = len(index)
        inlet = np.zeros([dims * 2], dtype=np.uint8)
        for dim, ind in enumerate(index):
            if ind == 0:
                inlet[dim * 2] = domain_inlet[dim][0]
            if ind == subdomains[dim] - 1:
                inlet[dim * 2 + 1] = domain_inlet[dim][1]

        return inlet

    def get_outlet(
        self,
        index: tuple[np.intp, ...],
        domain_outlet,
        subdomains: tuple[int, int, int],
    ):
        """
        Determine if subdomain is on outlet
        Also change orientation of how outlet is stored here
        """
        dims = len(index)
        outlet = np.zeros([dims * 2], dtype=np.uint8)
        for dim, ind in enumerate(index):
            if ind == 0:
                outlet[dim * 2] = domain_outlet[dim][0]
            if ind == subdomains[dim] - 1:
                outlet[dim * 2 + 1] = domain_outlet[dim][1]

        return outlet

    def get_start(
        self,
        index: tuple[int, int, int],
        domain_voxels,
        subdomains: tuple[int, int, int],
    ) -> tuple[int, ...]:
        """
        Determine the start of the subdomain. used for saving as vtk
        Start is the minimum voxel ID
        Args:
            sd_index (tuple[int, int, int]): subdomain index

        Returns:
            tuple[int,...]: start
        """
        _start = [0, 0, 0]

        for dim, _index in enumerate(index):
            sd_voxels, _ = divmod(domain_voxels[dim], subdomains[dim])
            _start[dim] = sd_voxels * _index

        return tuple(_start)
