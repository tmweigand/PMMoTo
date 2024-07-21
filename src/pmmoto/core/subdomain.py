"""subdomains.py"""
import numpy as np
from . import domain_discretization


class Subdomain(domain_discretization.DiscretizedDomain):
    """
    Parallelization is via decomposition of domain into subdomains
    """
    def __init__(
            self,
            rank: int= 0,
            index: tuple[int,int,int] = (0,0,0),
            **kwargs
            ):
        super().__init__(**kwargs)
        self.rank = rank
        self.index = index
        self.periodic = self.periodic_check()
        self.boundary = self.boundary_check()
    

    def boundary_check(self) -> bool:
        """
        Determine if subdomain is on a boundary
        """
        boundary = False
        for (m,p) in self.boundaries:
            if m != -1 or p != -1:
                boundary = True

        return boundary

