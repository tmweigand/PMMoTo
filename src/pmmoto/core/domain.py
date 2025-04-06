from typing import Literal
import numpy as np


class Domain:
    """
    Information for domain including:
        size_domain: Size of the domain in physical units
        boundary_types:  0: No assumption made
                         1: Wall boundary condition
                         2: Periodic boundary condition - Opposing face must also be 2!
        inlet: True/False boundary must be 0
        outlet: True/False boundary must be 0

    """

    def __init__(
        self,
        box: tuple[tuple[float, float], ...],
        boundary_types: tuple[tuple[int, int], ...] = ((0, 0), (0, 0), (0, 0)),
        inlet: tuple[tuple[int, int], ...] = ((0, 0), (0, 0), (0, 0)),
        outlet: tuple[tuple[int, int], ...] = ((0, 0), (0, 0), (0, 0)),
    ):
        # TODO: ADD input check
        self.box = box
        self.boundary_types = boundary_types
        self.inlet = inlet
        self.outlet = outlet
        self.dims = 3
        self.volume = self.get_volume()
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

    def get_volume(self):
        """
        Calculate the length of the domain
        """
        length = np.zeros([self.dims], dtype=np.float64)
        for n in range(0, self.dims):
            length[n] = self.box[n][1] - self.box[n][0]

        return np.prod(length)

    def periodic_check(self) -> bool:
        """
        Check if any external boundary is periodic boundary
        """
        periodic = False
        for d_bound in self.boundary_types:
            for n_bound in d_bound:
                if n_bound == 2:
                    periodic = True
        return periodic

    def get_origin(self) -> tuple[float, ...]:
        """
        Determine the domain origin from box

        Returns:
            tuple[float,...]: Domain origin
        """
        origin = [0, 0, 0]
        for n, box_dim in enumerate(self.box):
            origin[n] = box_dim[0]

        return tuple(origin)
