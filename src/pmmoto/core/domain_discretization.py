"""domain_discretization.py"""

import numpy as np
from . import domain as pmmoto_domain


class DiscretizedDomain(pmmoto_domain.Domain):
    """
    Class for discretizing the domain
    """

    def __init__(self, voxels: tuple[int, ...] = (1, 1, 1), **kwargs):
        super().__init__(**kwargs)
        self.voxels = voxels
        self.resolution = self.get_voxel_size()

    @classmethod
    def from_domain(cls, domain, voxels):
        return cls(
            box=domain.box,
            boundaries=domain.boundaries,
            inlet=domain.inlet,
            outlet=domain.outlet,
            voxels=voxels,
        )

    def get_voxel_size(self) -> tuple[float, ...]:
        """
        Get domain length and voxel size
        """
        res = np.zeros([self.dims])
        for n in range(0, self.dims):
            assert self.voxels[n] > 1
            res[n] = self.length[n] / self.voxels[n]

        return tuple(res)

    def get_coords(self):
        """
        Determine the physical locations of voxel centroids
        """
        coords = []
        for n in range(0, self.dims):
            half = 0.5 * self.resolution[n]
            coords.append(
                np.linspace(
                    self.box[n][0] + half, self.box[n][1] - half, self.voxels[n]
                )
            )
        return coords
