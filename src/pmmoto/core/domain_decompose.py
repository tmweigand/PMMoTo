"""decomposed_domain.py

Defines the DecomposedDomain class for dividing a discretized domain into subdomains.
"""

import numpy as np
from . import domain_discretization
from . import orientation


class DecomposedDomain(domain_discretization.DiscretizedDomain):
    """Collection of subdomains for domain decomposition.

    Used to divide the domain into subdomains and pass properties to each subdomain.
    """

    def __init__(self, subdomains: tuple[int, ...] = (1, 1, 1), **kwargs):
        """Initialize a DecomposedDomain.

        Args:
            subdomains (tuple[int, ...], optional): Number of subdomains
            **kwargs: Additional arguments passed to DiscretizedDomain.

        """
        super().__init__(**kwargs)
        self.subdomains = subdomains
        self.num_subdomains = np.prod(self.subdomains)
        self.map = self.gen_map()

    @classmethod
    def from_discretized_domain(cls, discretized_domain, subdomains):
        """Create a DecomposedDomain from an existing DiscretizedDomain and subdomains.

        Args:
            discretized_domain (DiscretizedDomain): The discretized domain object.
            subdomains (tuple[int, ...]): Number of subdomains in each dimension.

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

    def gen_map(self):
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

        if self.boundary_types[0][0] == 1:
            _map[0, :, :] = -2
        if self.boundary_types[0][1] == 1:
            _map[-1, :, :] = -2
        if self.boundary_types[1][0] == 1:
            _map[:, 0, :] = -2
        if self.boundary_types[1][1] == 1:
            _map[:, -1, :] = -2
        if self.boundary_types[2][0] == 1:
            _map[:, :, 0] = -2
        if self.boundary_types[2][1] == 1:
            _map[:, :, -1] = -2
        if self.boundary_types[0][0] == 2:
            _map[0, :, :] = _map[-2, :, :]
            _map[-1, :, :] = _map[1, :, :]
        if self.boundary_types[1][0] == 2:
            _map[:, 0, :] = _map[:, -2, :]
            _map[:, -1, :] = _map[:, 1, :]
        if self.boundary_types[2][0] == 2:
            _map[:, :, 0] = _map[:, :, -2]
            _map[:, :, -1] = _map[:, :, 1]

        return _map

    def get_neighbor_ranks(self, sd_index: tuple[np.intp, np.intp, np.intp]):
        """Determine the neighbor process rank for each feature.

        Args:
            sd_index (tuple[np.intp, np.intp, np.intp]): Index of the subdomain.

        Returns:
            dict: Mapping from feature index to neighbor rank.

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
        sd_index: tuple[np.intp, np.intp, np.intp],
        feature_index: tuple[int, int, int],
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
