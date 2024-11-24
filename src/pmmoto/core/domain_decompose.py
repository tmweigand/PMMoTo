"""decomposed_domain.py"""

import numpy as np
from . import domain_discretization


class DecomposedDomain(domain_discretization.DiscretizedDomain):
    """
    Collection of subdomains. Used to divide domain into subdomain
    and pass properties to subdomain
    """

    def __init__(self, subdomains: tuple[int, ...] = (1, 1, 1), **kwargs):
        super().__init__(**kwargs)
        self.subdomains = subdomains
        self.num_subdomains = np.prod(self.subdomains)
        self.map = self.gen_map()

    @classmethod
    def from_discretized_domain(cls, discretized_domain, subdomains):
        return cls(
            box=discretized_domain.box,
            boundary_types=discretized_domain.boundary_types,
            inlet=discretized_domain.inlet,
            outlet=discretized_domain.outlet,
            voxels=discretized_domain.voxels,
            subdomains=subdomains,
        )

    def gen_map(self):
        """
        Generate process lookup map.
        -2: Wall Boundary Condition
        -1: No Assumption Boundary Condition
        >=0: proc_ID
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
        """
        Determine the neighbor process rank
        """
        from . import orientation

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
