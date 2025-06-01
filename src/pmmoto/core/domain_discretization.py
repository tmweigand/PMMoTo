"""domain_discretization.py

Defines the DiscretizedDomain class for discretizing a physical domain into voxels.
"""

import numpy as np
from . import domain as pmmoto_domain


class DiscretizedDomain(pmmoto_domain.Domain):
    """Discretize a physical domain into a voxel grid.

    This class extends the base Domain class to include voxelization and
    calculation of voxel resolution and coordinates.
    """

    def __init__(self, voxels: tuple[int, ...] = (1, 1, 1), **kwargs):
        """Initialize a discretized domain.

        Args:
            voxels (tuple[int, ...], optional): Number of voxels in each dimension.
            **kwargs: Additional arguments passed to the base Domain class.

        """
        super().__init__(**kwargs)
        self.voxels = voxels
        self.resolution = self.get_resolution()

    @classmethod
    def from_domain(cls, domain, voxels):
        """Create a DiscretizedDomain from an existing Domain and voxel counts.

        Args:
            domain (Domain): The base domain object.
            voxels (tuple[int, ...]): Number of voxels in each dimension.

        Returns:
            DiscretizedDomain: New discretized domain instance.

        """
        return cls(
            box=domain.box,
            boundary_types=domain.boundary_types,
            inlet=domain.inlet,
            outlet=domain.outlet,
            voxels=voxels,
        )

    def get_resolution(self) -> tuple[float, ...]:
        """Calculate the physical size of each voxel in every dimension.

        Returns:
            tuple[float, ...]: Resolution (voxel size) in each dimension.

        """
        res = np.zeros([self.dims])
        for n in range(0, self.dims):
            assert self.voxels[n] > 1
            res[n] = self.length[n] / self.voxels[n]
        return tuple(res)

    @staticmethod
    def get_coords(box, voxels, resolution):
        """Determine the physical locations of voxel centroids.

        Args:
            box (tuple[tuple[float, float], ...]): Physical bounds for each dimension.
            voxels (tuple[int, ...]): Number of voxels in each dimension.
            resolution (tuple[float, ...]): Voxel size in each dimension.

        Returns:
            list[np.ndarray]: List of arrays with centroid coordinates.

        """
        coords = []
        for voxels, box, resolution in zip(voxels, box, resolution):
            half = 0.5 * resolution
            coords.append(np.linspace(box[0] + half, box[1] - half, voxels))
        return coords
