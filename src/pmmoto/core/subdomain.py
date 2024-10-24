"""subdomains.py"""

from . import domain_discretization
from . import orientation
from . import subdomain_features

import numpy as np


class Subdomain(domain_discretization.DiscretizedDomain):
    """
    Parallelization is via decomposition of domain into subdomains
    """

    def __init__(
        self,
        rank: int,
        index: tuple[int, int, int],
        start: tuple[int, int, int],
        num_subdomains: int,
        domain_voxels: tuple[int, int, int],
        neighbor_ranks={},
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.rank = rank
        self.index = index
        self.num_subdomains = num_subdomains
        self.start = start
        self.domain_voxels = domain_voxels
        self.neighbor_ranks = neighbor_ranks
        self.periodic = self.periodic_check()
        self.boundary = self.boundary_check()
        self.coords = self.get_coords()
        self.features = subdomain_features.collect_features(
            self.neighbor_ranks, self.boundary, self.boundaries, self.voxels
        )

    def boundary_check(self) -> bool:
        """
        Determine if subdomain is on a boundary
        """
        boundary = False
        for n_bound in self.boundaries:
            if n_bound != -1:
                boundary = True

        return boundary
