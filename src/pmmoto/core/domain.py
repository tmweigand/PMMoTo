"""domain.py

Defines the Domain class for representing the physical simulation domain in PMMoTo.
"""

from typing import Any
import numpy as np


from .boundary_types import BoundaryType


class Domain:
    """Represent a physical simulation domain.

    Attributes:
        box (tuple[tuple[float, float], ...]): Physical bounds for each dimension.
        boundary_types (tuple[tuple[BoundaryType, BoundaryType], ...]):
            Boundary types for each face.
            END: No assumption made
            WALL: Wall boundary condition
            PERIODIC: Periodic boundary condition (opposing face must also be 2)
        inlet (tuple[tuple[bool, bool], ...]): Inlet flags (must be 0 boundary type).
        outlet (tuple[tuple[bool, bool], ...]): Outlet flags (must be 0 boundary type).
        dims (int): Number of spatial dimensions (default 3).
        volume (float): Volume of the domain.
        periodic (bool): True if any boundary is periodic.
        length (tuple[float, ...]): Length of the domain in each dimension.

    """

    def __init__(
        self,
        box: tuple[tuple[float, float], ...],
        boundary_types: tuple[tuple[BoundaryType, BoundaryType], ...] = (
            (BoundaryType.END, BoundaryType.END),
            (BoundaryType.END, BoundaryType.END),
            (BoundaryType.END, BoundaryType.END),
        ),
        inlet: tuple[tuple[bool, bool], ...] = (
            (False, False),
            (False, False),
            (False, False),
        ),
        outlet: tuple[tuple[int, int], ...] = (
            (False, False),
            (False, False),
            (False, False),
        ),
    ):
        """Initialize a Domain.

        Args:
            box (tuple[tuple[float, float], ...]): Physical bounds for each dimension.
            boundary_types (tuple[tuple[int, int], ...], optional): Boundary types.
            inlet (tuple[tuple[int, int], ...], optional): Inlet flags for each face.
            outlet (tuple[tuple[int, int], ...], optional): Outlet flags for each face.

        """
        # TODO: ADD input check

        # Runtime type assertions for boundary_types
        assert isinstance(boundary_types, tuple), "boundary_types must be a tuple"
        for bt in boundary_types:
            assert (
                isinstance(bt, tuple) and len(bt) == 2
            ), "Each item must be a tuple of length 2"
            assert all(
                isinstance(b, BoundaryType) for b in bt
            ), "Each element must be a BoundaryType"

        self.box = box
        self.boundary_types = boundary_types
        self.inlet = inlet
        self.outlet = outlet
        self.dims = 3
        self.volume = self.get_volume()
        self.periodic = self.periodic_check()
        self.length = self.get_length()

    def get_length(self) -> tuple[float, ...]:
        """Calculate the length of the domain in each dimension.

        Returns:
            tuple[float, ...]: Length in each dimension.

        """
        length = np.zeros([self.dims], dtype=np.float64)
        for n in range(0, self.dims):
            length[n] = self.box[n][1] - self.box[n][0]
        return tuple(length)

    def get_volume(self) -> np.floating[Any]:
        """Calculate the volume of the domain.

        Returns:
            float: Volume of the domain.

        """
        length = np.zeros([self.dims], dtype=np.float64)
        for n in range(0, self.dims):
            length[n] = self.box[n][1] - self.box[n][0]
        return np.prod(length)

    def periodic_check(self) -> bool:
        """Check if any external boundary is a periodic boundary.

        Returns:
            bool: True if any boundary is periodic, False otherwise.

        """
        periodic = False
        for d_bound in self.boundary_types:
            for n_bound in d_bound:
                if n_bound == BoundaryType.PERIODIC:
                    periodic = True
        return periodic

    def get_origin(self) -> tuple[float, ...]:
        """Determine the domain origin from box.

        Returns:
            tuple[float, ...]: Domain origin.

        """
        origin: list[float] = [0.0] * len(self.box)
        for n, box_dim in enumerate(self.box):
            origin[n] = box_dim[0]
        return tuple(origin)
