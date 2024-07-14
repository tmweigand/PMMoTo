"""subdomains.py"""
import numpy as np
from . import domain


class Subdomain(domain.Domain):
    """
    Parallelization is via decomposition of domain into subdomains
    """
    def __init__(
            self,
            rank = 0,
            index = np.array((0,0,0)),
            size_domain = np.array([(0,1),(0,1),(0,1)]),
            boundaries = ((-1,-1),(-1,-1),(-1,-1)),
            inlet = ((0,0),(0,0),(0,0)),
            outlet =((0,0),(0,0),(0,0))
            ):
        super().__init__(
            size_domain,
            boundaries,
            inlet,
            outlet,
        )
        self.rank = rank
        self.index = index
        self.periodic = self.periodic_check()
        self.boundary = self.boundary_check()
    

    def boundary_check(self):
        """
        Determine if subdomain is on a boundary
        """
        boundary = False
        for (m,p) in self.boundaries:
            if m != -1 or p != -1:
                boundary = True

        return boundary

