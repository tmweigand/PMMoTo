"""domain_discretization.py

Defines the DiscretizedDomain class for discretizing a physical domain into voxels.
"""

from __future__ import annotations
from typing import Any, TYPE_CHECKING
from typing_extensions import Self
import numpy as np
from numpy.typing import NDArray
from . import domain as pmmoto_domain

if TYPE_CHECKING:
    from .domain import Domain


class DiscretizedDomain(pmmoto_domain.Domain):
    """Discretize a physical domain into a voxel grid.

    This class extends the base Domain class to include voxelization and
    calculation of voxel resolution and coordinates.
    """

    def __init__(self, voxels: tuple[int, ...] = (1, 1, 1), **kwargs: Any):
        """Initialize a discretized domain.

        Args:
            voxels (tuple[int, int, int], optional): Number of voxels in each dimension.
            **kwargs: Additional arguments passed to the base Domain class.

        """
        super().__init__(**kwargs)
        self.voxels = voxels
        self.resolution = self.get_resolution()

    @classmethod
    def from_domain(cls, domain: Domain, voxels: tuple[int, ...]) -> Self:
        """Create a DiscretizedDomain from an existing Domain and voxel counts.

        Args:
            domain (Domain): The base domain object.
            voxels (tuple[int, int, int]): Number of voxels in each dimension.

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
            tuple[float, float, float] Resolution (voxel size) in each dimension.

        """
        res = np.zeros([self.dims])
        for n in range(0, self.dims):
            assert self.voxels[n] > 1
            res[n] = self.length[n] / self.voxels[n]
        return tuple(res)

    @staticmethod
    def get_coords(
        box: tuple[tuple[float, float], ...],
        voxels: tuple[int, ...],
        resolution: tuple[float, ...],
    ) -> list[NDArray[np.float64]]:
        """Determine the physical locations of voxel centroids.

        Args:
            box (tuple[tuple[float, float], ...]): Physical bounds for each dimension.
            voxels (tuple[int,int, int]): Number of voxels in each dimension.
            resolution (tuple[float, float, float]): Voxel size in each dimension.

        Returns:
            list[np.ndarray]: List of arrays with centroid coordinates.

        """
        coords = []
        for _voxels, _box, _resolution in zip(voxels, box, resolution):
            half = 0.5 * _resolution
            coords.append(np.linspace(_box[0] + half, _box[1] - half, _voxels))
        return coords
