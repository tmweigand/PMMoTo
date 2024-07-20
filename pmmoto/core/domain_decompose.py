"""decomposed_domain.py"""

import numpy as np
from pmmoto.core import subdomain
from pmmoto.core import domain_discretization


class DecomposedDomain(domain_discretization.DiscretizedDomain):
    """
    Collection of subdomains. Used to divide domain into subdomain
    and pass properties to subdomain
    """

    def __init__(
        self, rank: int = 0, subdomain_map: tuple[int, ...] = (1, 1, 1), **kwargs
    ):
        super().__init__(**kwargs)
        self.rank = rank
        self.subdomain_map = subdomain_map
        self.num_subdomains = np.prod(subdomain_map)
        self.map = self.gen_subdomain_map()
        self.subdomains = self.initialize_subdomain()

    def gen_subdomain_map(self):
        """
        Generate the mapping of the subdomains in the domain
        """
        return np.arange(self.num_subdomains).reshape(self.subdomain_map)

    def initialize_subdomain(self):
        """
        Initialize the subdomains
        """
        for sd_id in self.map.flatten():
            sd_index = self.get_subdomain_index(sd_id)
            if self.rank == sd_id:
                voxels = self.get_subdomain_voxels(sd_index)
                _subdomain = subdomain.Subdomain(
                    rank=sd_id,
                    index=sd_index,
                    box=self.get_subdomain_box(sd_index, voxels),
                    boundaries=self.get_subdomain_boundaries(sd_index),
                    inlet=self.get_subdomain_inlet(sd_index),
                    outlet=self.get_subdomain_outlet(sd_index),
                    num_voxels=voxels,
                )

        return _subdomain

    def get_subdomain_voxels(self, index: tuple[np.intp, ...]) -> tuple[int, ...]:
        """
        Calculate number of voxels in each subdomain
        """
        voxels = [0, 0, 0]
        for dim, ind in enumerate(index):
            sd_voxels, rem_sd_voxels = divmod(self.res[dim], self.subdomain_map[dim])
            if ind == self.subdomain_map[dim] - 1:
                voxels[dim] = sd_voxels + rem_sd_voxels
            else:
                voxels[dim] = sd_voxels

        return tuple(voxels)

    def get_subdomain_index(self, id: int) -> tuple[np.intp, ...]:
        """
        Determine the index of the subdomain
        """
        return np.unravel_index(id, self.map.shape)

    def get_subdomain_box(
        self, index: tuple[np.intp, ...], voxels: tuple[int, int, int]
    ):
        """
        Determine the bounding box for each subdomain.
        Note: subdomains are divided such that voxel spacing
        is constant
        """
        box = np.zeros([self.dims, 2])
        for dim, ind in enumerate(index):

            length = voxels[dim] * self.res[dim]
            box[dim, 0] = self.box[dim, 0] + length
            if ind == self.subdomain_map[dim] - 1:
                box[dim, 0] = self.box[dim, 1] - length
            box[dim, 1] = box[dim, 0] + length

        return box

    def get_subdomain_boundaries(self, index: tuple[np.intp, ...]):
        """
        Determine the boundary types for each subdomain
        """
        dims = len(index)
        boundaries = -np.ones([dims, 2], dtype=np.int8)
        for dim, ind in enumerate(index):
            if ind == 0:
                boundaries[dim, 0] = self.boundaries[dim][0]
            if ind == self.subdomain_map[dim] - 1:
                boundaries[dim, 1] = self.boundaries[dim][1]

        return boundaries

    def get_subdomain_inlet(self, index: tuple[np.intp, ...]):
        """
        Determine if subdomain is on inlet
        """
        dims = len(index)
        inlet = np.zeros([dims, 2], dtype=np.uint8)
        for dim, ind in enumerate(index):
            if ind == 0:
                inlet[dim, 0] = self.inlet[dim][0]
            if ind == self.subdomain_map[dim] - 1:
                inlet[dim, 1] = self.inlet[dim][1]

        return inlet

    def get_subdomain_outlet(self, index: tuple[np.intp, ...]):
        """
        Determine if subdomain is on outlet
        """
        dims = len(index)
        outlet = np.zeros([dims, 2], dtype=np.uint8)
        for dim, ind in enumerate(index):
            if ind == 0:
                outlet[dim, 0] = self.outlet[dim][0]
            if ind == self.subdomain_map[dim] - 1:
                outlet[dim, 1] = self.outlet[dim][1]

        return outlet
