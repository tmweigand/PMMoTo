"""domain_discretization.py"""

import numpy as np
from . import domain as pmmoto_domain


class DiscretizedDomain(pmmoto_domain.Domain):
    """
    Class for discretizing the domain
    """

    def __init__(self, num_voxels: tuple[int, ...] = (1, 1, 1), **kwargs):
        super().__init__(**kwargs)
        self.num_voxels = num_voxels
        self.resolution = self.get_voxel_size()

    def get_voxel_size(self) -> tuple[float, ...]:
        """
        Get domain length and voxel size
        """
        res = np.zeros([self.dims])
        for n in range(0, self.dims):
            if self.num_voxels[n] < 1:
                print("Error")
            else:
                res[n] = self.length[n] / self.num_voxels[n]

        return tuple(res)
