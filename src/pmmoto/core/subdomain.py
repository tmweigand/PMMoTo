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
        self.index = self.get_index()
        self.voxels = self.get_voxels()
        self.box = self.get_box()
        self.global_boundary = self.get_global_boundary()
        self.boundary_types = self.get_boundary_types(self.global_boundary)
        self.inlet = self.get_inlet()
        self.outlet = self.get_outlet()
        self.start = self.get_start()
        self.neighbor_ranks = self.domain.get_neighbor_ranks(self.index)
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

    def get_index(self) -> tuple[np.intp, ...]:
        """
        Determine the index of the subdomain
        """
        return np.unravel_index(self.rank, self.domain.subdomains)

    def get_voxels(self) -> tuple[int, ...]:
        """
        Calculate number of voxels in each subdomain.
        This can be very bad when voxels ~= ranks or something like that
        """
        voxels = [0, 0, 0]
        for dim, ind in enumerate(self.index):
            sd_voxels, rem_sd_voxels = divmod(
                self.domain.voxels[dim], self.domain.subdomains[dim]
            )
            if ind == self.domain.subdomains[dim] - 1:
                voxels[dim] = sd_voxels + rem_sd_voxels
            else:
                voxels[dim] = sd_voxels

        return tuple(voxels)

    def get_box(self):
        """
        Determine the bounding box for each subdomain.
        Note: subdomains are divided such that voxel spacing
        is constant
        """
        box = []
        for dim, ind in enumerate(self.index):
            length = self.voxels[dim] * self.domain.resolution[dim]
            lower = self.domain.box[dim][0] + length * ind
            if ind == self.domain.subdomains[dim] - 1:
                lower = self.domain.box[dim][1] - length
            upper = lower + length
            box.append((lower, upper))

        return tuple(box)

    def get_global_boundary(self):
        """
        Determine if the features are on the domain boundary
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

    def get_boundary_types(self, global_boundary):
        """_summary_

        Returns:
            _type_: _description_
        """
        boundary_type = {}
        features = orientation.get_features()
        for feature in features:
            if global_boundary[feature]:
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
        Determine if subdomain is on inlet
        Also change orientation of how inlet is stored here
        """
        dims = len(self.index)
        inlet = np.zeros([dims * 2], dtype=np.uint8)
        for dim, ind in enumerate(self.index):
            if ind == 0:
                inlet[dim * 2] = self.domain.inlet[dim][0]
            if ind == self.domain.subdomains[dim] - 1:
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
