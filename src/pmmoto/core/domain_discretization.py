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
        self.resolution = self.get_resolution()

    @classmethod
    def from_domain(cls, domain, voxels):
        return cls(
            box=domain.box,
            boundary_types=domain.boundary_types,
            inlet=domain.inlet,
            outlet=domain.outlet,
            voxels=voxels,
        )

    def get_resolution(self) -> tuple[float, ...]:
        """
        Get domain length and voxel size
        """
        res = np.zeros([self.dims])
        for n in range(0, self.dims):
            assert self.voxels[n] > 1
            res[n] = self.length[n] / self.voxels[n]

        return tuple(res)

    @staticmethod
    def get_coords(box, voxels, resolution):
        """
        Determine the physical locations of voxel centroids
        """
        coords = []
        for voxels, box, resolution in zip(voxels, box, resolution):
            half = 0.5 * resolution
            coords.append(np.linspace(box[0] + half, box[1] - half, voxels))
        return coords
