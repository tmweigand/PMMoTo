from typing import Literal
import numpy as np


class Domain:
    """
    Information for domain including:
        size_domain: Size of the domain in physical units
        boundaries:  0: No assumption made
                     1: Wall boundary condition
                     2: Periodic boundary condition
                        Opposing boundary must also be 2
        inlet: True/False boundary must be 0
        outlet: True/False boundary must be 0

    """

    def __init__(
        self,
        box: tuple[tuple[float, float], ...],
        boundaries: tuple[tuple[int, int], ...] = ((0, 0), (0, 0), (0, 0)),
        inlet: tuple[tuple[int, int], ...] = ((0, 0), (0, 0), (0, 0)),
        outlet: tuple[tuple[int, int], ...] = ((0, 0), (0, 0), (0, 0)),
    ):
        self.box = box
        self.boundaries = boundaries
        self.inlet = inlet
        self.outlet = outlet
        self.dims = 3
        self.periodic = self.periodic_check()
        self.length = self.get_length()

    def get_length(self) -> tuple[float, ...]:
        """
        Calculate the length of the domain
        """
        length = np.zeros([self.dims], dtype=np.float64)
        for n in range(0, self.dims):
            length[n] = self.box[n][1] - self.box[n][0]

        return tuple(length)

    def periodic_check(self) -> bool:
        """
        Check if any external boundary is periodic boundary
        """

        if len(self.boundaries) == self.dims * 2:
            boundaries = []
            for n in range(self.dims):
                boundaries.append([self.boundaries[n * 2], self.boundaries[n * 2 + 1]])
        else:
            boundaries = self.boundaries

        periodic = False
        for d_bound in boundaries:
            for n_bound in d_bound:
                if n_bound == 2:
                    periodic = True
        return periodic

    def update_domain_size(self):
        """
        Use data from io to set domain size and determine voxel size and coordinates
        """
        pass
