"""decomposed_domain.py

Defines the DecomposedDomain class for dividing a discretized domain into subdomains.
"""

from __future__ import annotations
from typing import Any, TYPE_CHECKING
from typing_extensions import Self
import numpy as np
from numpy.typing import NDArray
from .boundary_types import BoundaryType
from . import domain_discretization
from .orientation import FEATURE_MAP

if TYPE_CHECKING:
    from .domain_discretization import DiscretizedDomain


class DecomposedDomain(domain_discretization.DiscretizedDomain):
    """Collection of subdomains for domain decomposition.

    Used to divide the domain into subdomains and pass properties to each subdomain.
    """

    def __init__(self, subdomains: tuple[int, ...] = (1, 1, 1), **kwargs: Any):
        """Initialize a DecomposedDomain.

        Args:
            subdomains (tuple[int, int, int] optional): Number of subdomains
            **kwargs: Additional arguments passed to DiscretizedDomain.

        """
        super().__init__(**kwargs)
        self.subdomains = subdomains
        self.num_subdomains = np.prod(self.subdomains)
        self.map = self.gen_map()

    @classmethod
    def from_discretized_domain(
        cls, discretized_domain: DiscretizedDomain, subdomains: tuple[int, ...]
    ) -> Self:
        """Create a DecomposedDomain from an existing DiscretizedDomain and subdomains.

        Args:
            discretized_domain (DiscretizedDomain): The discretized domain object.
            subdomains: Number of subdomains in each spatial dimension.


        Returns:
            DecomposedDomain: New decomposed domain instance.

        """
        return cls(
            box=discretized_domain.box,
            boundary_types=discretized_domain.boundary_types,
            inlet=discretized_domain.inlet,
            outlet=discretized_domain.outlet,
            voxels=discretized_domain.voxels,
            subdomains=subdomains,
        )

    def gen_map(self) -> NDArray[np.int64]:
        """Generate process lookup map for subdomains.

        Map values:
            -2: Wall Boundary Condition
            -1: No Assumption Boundary Condition
            >=0: proc_ID

        Returns:
            np.ndarray: Map array with process IDs and boundary flags.

        """
        _map = -np.ones([sd + 2 for sd in self.subdomains], dtype=np.int64)
        _map[1:-1, 1:-1, 1:-1] = np.arange(self.num_subdomains).reshape(self.subdomains)

        if self.boundary_types[0][0] == BoundaryType.WALL:
            _map[0, :, :] = -2
        if self.boundary_types[0][1] == BoundaryType.WALL:
            _map[-1, :, :] = -2
        if self.boundary_types[1][0] == BoundaryType.WALL:
            _map[:, 0, :] = -2
        if self.boundary_types[1][1] == BoundaryType.WALL:
            _map[:, -1, :] = -2
        if self.boundary_types[2][0] == BoundaryType.WALL:
            _map[:, :, 0] = -2
        if self.boundary_types[2][1] == BoundaryType.WALL:
            _map[:, :, -1] = -2

        if self.boundary_types[0][0] == BoundaryType.PERIODIC:
            _map[0, :, :] = _map[-2, :, :]
            _map[-1, :, :] = _map[1, :, :]
        if self.boundary_types[1][0] == BoundaryType.PERIODIC:
            _map[:, 0, :] = _map[:, -2, :]
            _map[:, -1, :] = _map[:, 1, :]
        if self.boundary_types[2][0] == BoundaryType.PERIODIC:
            _map[:, :, 0] = _map[:, :, -2]
            _map[:, :, -1] = _map[:, :, 1]

        return _map

    def get_neighbor_ranks(
        self, sd_index: tuple[int, ...]
    ) -> dict[tuple[int, ...], int]:
        """Determine the neighbor process rank for each feature.

        Args:
            sd_index (tuple[int, int, int]): Index of the subdomain.

        Returns:
            dict: Mapping from feature index to neighbor rank.

        """
        neighbor_ranks: dict[tuple[int, ...], int] = {}

        feature_ids = FEATURE_MAP.collect_feature_ids()
        for _id in feature_ids:
            neighbor_ranks[_id] = self._get_neighbor_ranks(sd_index, _id)
        return neighbor_ranks

    def _get_neighbor_ranks(
        self,
        sd_index: tuple[int, ...],
        feature_index: tuple[int, ...],
    ) -> int:
        """Get the neighbor rank for a specific feature.

        Args:
            sd_index (tuple[int, int, int]): Index of the subdomain.
            feature_index (tuple[int, int, int]): Feature index (face, edge, or corner).

        Returns:
            int: Neighbor process rank or boundary flag.

        """
        index = []
        for n in range(self.dims):
            index.append(feature_index[n] + sd_index[n] + 1)

        rank = int(self.map[index[0], index[1], index[2]])

        return rank
