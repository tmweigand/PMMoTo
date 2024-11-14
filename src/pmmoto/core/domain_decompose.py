"""decomposed_domain.py"""

import numpy as np
from pmmoto.core import domain_discretization
from pmmoto.core import orientation
from pmmoto.core import subdomain


class DecomposedDomain(domain_discretization.DiscretizedDomain):
    """
    Collection of subdomains. Used to divide domain into subdomain
    and pass properties to subdomain
    """

    def __init__(self, subdomain_map: tuple[int, ...] = (1, 1, 1), **kwargs):
        super().__init__(**kwargs)
        self.subdomain_map = subdomain_map
        self.num_subdomains = np.prod(self.subdomain_map)
        self.map = self.gen_map()

    @classmethod
    def from_discretized_domain(cls, discretized_domain, subdomain_map):
        return cls(
            box=discretized_domain.box,
            boundaries=discretized_domain.boundaries,
            inlet=discretized_domain.inlet,
            outlet=discretized_domain.outlet,
            voxels=discretized_domain.voxels,
            subdomain_map=subdomain_map,
        )

    def initialize_subdomain(self, rank: int):
        """
        Initialize the subdomains and return subdomain on rank
        """
        sd_index = self.get_subdomain_index(rank)
        voxels = self.get_subdomain_voxels(sd_index)
        box = self.get_subdomain_box(sd_index, voxels)
        boundaries, boundary_types = self.get_subdomain_boundaries(sd_index)
        inlet = self.get_subdomain_inlet(sd_index)
        outlet = self.get_subdomain_outlet(sd_index)
        start = self.get_subdomain_start(sd_index)
        neighbor_ranks = self.get_neighbor_ranks(sd_index)
        _subdomain = subdomain.Subdomain(
            rank=rank,
            index=sd_index,
            num_subdomains=self.num_subdomains,
            start=start,
            domain_voxels=self.voxels,
            neighbor_ranks=neighbor_ranks,
            box=box,
            boundaries=boundaries,
            boundary_types=boundary_types,
            inlet=inlet,
            outlet=outlet,
            voxels=voxels,
        )

        return _subdomain

    def get_subdomain_voxels(self, index: tuple[np.intp, ...]) -> tuple[int, ...]:
        """
        Calculate number of voxels in each subdomain.
        This can be very bad when voxels ~= ranks or something like that
        """
        voxels = [0, 0, 0]
        for dim, ind in enumerate(index):
            sd_voxels, rem_sd_voxels = divmod(self.voxels[dim], self.subdomain_map[dim])
            if ind == self.subdomain_map[dim] - 1:
                voxels[dim] = sd_voxels + rem_sd_voxels
            else:
                voxels[dim] = sd_voxels

        return tuple(voxels)

    def get_subdomain_index(self, id: int) -> tuple[np.intp, ...]:
        """
        Determine the index of the subdomain
        """
        return np.unravel_index(id, self.subdomain_map)

    def get_subdomain_box(
        self, index: tuple[np.intp, ...], voxels: tuple[int, int, int]
    ):
        """
        Determine the bounding box for each subdomain.
        Note: subdomains are divided such that voxel spacing
        is constant
        """
        box = []
        for dim, ind in enumerate(index):
            length = voxels[dim] * self.resolution[dim]
            lower = self.box[dim][0] + length * ind
            if ind == self.subdomain_map[dim] - 1:
                lower = self.box[dim][1] - length
            upper = lower + length
            box.append((lower, upper))

        return tuple(box)

    def get_subdomain_boundaries(self, index: tuple[np.intp, ...]):
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
                        _boundary_type.append(self.boundaries[dim][0])
                    elif (ind == self.subdomain_map[dim] - 1) and (f_id == 1):
                        _boundary_type.append(self.boundaries[dim][1])
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

    def get_subdomain_inlet(self, index: tuple[np.intp, ...]):
        """
        Determine if subdomain is on inlet
        Also change orientation of how inlet is stored here
        """
        dims = len(index)
        inlet = np.zeros([dims * 2], dtype=np.uint8)
        for dim, ind in enumerate(index):
            if ind == 0:
                inlet[dim * 2] = self.inlet[dim][0]
            if ind == self.subdomain_map[dim] - 1:
                inlet[dim * 2 + 1] = self.inlet[dim][1]

        return inlet

    def get_subdomain_outlet(self, index: tuple[np.intp, ...]):
        """
        Determine if subdomain is on outlet
        Also change orientation of how outlet is stored here
        """
        dims = len(index)
        outlet = np.zeros([dims * 2], dtype=np.uint8)
        for dim, ind in enumerate(index):
            if ind == 0:
                outlet[dim * 2] = self.outlet[dim][0]
            if ind == self.subdomain_map[dim] - 1:
                outlet[dim * 2 + 1] = self.outlet[dim][1]

        return outlet

    def gen_map(self):
        """
        Generate process lookup map.
        -2: Wall Boundary Condition
        -1: No Assumption Boundary Condition
        >=0: proc_ID
        """
        _map = -np.ones([sd + 2 for sd in self.subdomain_map], dtype=np.int64)
        _map[1:-1, 1:-1, 1:-1] = np.arange(self.num_subdomains).reshape(
            self.subdomain_map
        )
        ### Set Boundaries of global SubDomain Map
        if self.boundaries[0][0] == 1:
            _map[0, :, :] = -2
        if self.boundaries[0][1] == 1:
            _map[-1, :, :] = -2
        if self.boundaries[1][0] == 1:
            _map[:, 0, :] = -2
        if self.boundaries[1][1] == 1:
            _map[:, -1, :] = -2
        if self.boundaries[2][0] == 1:
            _map[:, :, 0] = -2
        if self.boundaries[2][1] == 1:
            _map[:, :, -1] = -2
        if self.boundaries[0][0] == 2:
            _map[0, :, :] = _map[-2, :, :]
            _map[-1, :, :] = _map[1, :, :]
        if self.boundaries[1][0] == 2:
            _map[:, 0, :] = _map[:, -2, :]
            _map[:, -1, :] = _map[:, 1, :]
        if self.boundaries[2][0] == 2:
            _map[:, :, 0] = _map[:, :, -2]
            _map[:, :, -1] = _map[:, :, 1]

        return _map

    def get_subdomain_global_opposite_feature(self, boundary_types):
        """
        Determine the opposite feature for domain boundary features
        """
        pass

    def get_neighbor_ranks(self, sd_index: tuple[int, int, int]):
        """
        Determine the neighbor process rank
        """
        neighbor_ranks = {}
        feature_types = ["faces", "edges", "corners"]
        for feature_type in feature_types:
            for feature_index in orientation.features[feature_type].keys():
                neighbor_ranks[feature_index] = self._get_neighbor_ranks(
                    sd_index, feature_index
                )
        return neighbor_ranks

    def _get_neighbor_ranks(
        self,
        sd_index: tuple[int, int, int],
        feature_index: tuple[int, int, int],
    ) -> int:
        """_summary_

        Args:
            sd_index (tuple[int, int, int]): _description_
            feature_index (tuple[int, int, int]): _description_
        """
        index = []
        for n in range(self.dims):
            index.append(feature_index[n] + sd_index[n] + 1)

        rank = self.map[index[0], index[1], index[2]]

        return rank

    def get_subdomain_start(self, index: tuple[int, int, int]) -> tuple[int, ...]:
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
            sd_voxels, rem_sd_voxels = divmod(self.voxels[dim], self.subdomain_map[dim])
            _start[dim] = sd_voxels * _index

        return tuple(_start)
