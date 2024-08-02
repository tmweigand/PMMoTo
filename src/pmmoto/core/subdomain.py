"""subdomains.py"""

from . import domain_discretization


class Subdomain(domain_discretization.DiscretizedDomain):
    """
    Parallelization is via decomposition of domain into subdomains
    """

    def __init__(
        self, rank: int = 0, index: tuple[int, int, int] = (0, 0, 0), **kwargs
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
        for minus, plus in self.boundaries:
            if minus != -1 or plus != -1:
                boundary = True

        return boundary
