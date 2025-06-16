"""lattice_packings.py

Defines classes for generating common lattice packings (SC, BCC, FCC) for PMMoTo.
"""

from __future__ import annotations
from typing import TypeVar
import math
import numpy as np
from numpy.typing import NDArray
from ..core.subdomain import Subdomain
from ..core.subdomain_padded import PaddedSubdomain
from ..core.subdomain_verlet import VerletSubdomain

T = TypeVar("T", bound=np.generic)


class Lattice:
    """Base class for generating common lattice packings.

    Note:
        This currently adds more objects than needed.

    """

    def __init__(
        self,
        subdomain: Subdomain | PaddedSubdomain | VerletSubdomain,
        lattice_constant: float,
    ):
        """Initialize a Lattice.

        Args:
            subdomain: Subdomain object.
            lattice_constant (float): Lattice constant (unit cell size).

        """
        self.subdomain = subdomain
        self.lattice_constant = lattice_constant

    def get_basis_vectors(self) -> NDArray[T]:
        """Return the basis vectors for the lattice.

        Returns:
            np.ndarray: Array of basis vectors.

        """
        return np.array([])

    def get_radius(self) -> float:
        """Return the sphere radius for the lattice.

        Returns:
            float: Sphere radius.

        """
        return 0.0

    def generate_lattice(self) -> NDArray[T]:
        """Generate a lattice for a given unit cell size and lattice type.

        Returns:
            np.ndarray: Array of shape (N, 4) of lattice points and sphere radius.

        """
        basis_vectors: NDArray[T] = self.get_basis_vectors()
        radius = self.get_radius()

        # Compute unit cell size dynamically
        repeats = [0, 0, 0]
        for dim, box in enumerate(self.subdomain.domain.box):
            repeats[dim] = int((box[1] - box[0]) // self.lattice_constant + 1)

        points = set()
        for i in range(repeats[0]):
            for j in range(repeats[1]):
                for k in range(repeats[2]):
                    cell_origin = np.array([i, j, k]) * self.lattice_constant
                    for basis in basis_vectors:
                        atom_pos = tuple(cell_origin + basis * self.lattice_constant)
                        points.add((*atom_pos, radius))
        return np.array(list(points))


class SimpleCubic(Lattice):
    """Simple Cubic (SC) lattice."""

    def __init__(
        self,
        subdomain: Subdomain | PaddedSubdomain | VerletSubdomain,
        lattice_constant: float,
    ):
        super().__init__(subdomain, lattice_constant)

    def get_basis_vectors(self) -> NDArray[T]:
        """Return basis vectors for SC lattice."""
        return np.array([[0, 0, 0]])

    def get_coordination_number(self) -> float:
        """Return coordination number for SC lattice."""
        return 6

    def get_packing_efficiency(self) -> float:
        """Return packing efficiency (percent) for SC lattice."""
        return 52  # percent

    def get_radius(self) -> float:
        """Return sphere radius for SC lattice."""
        return self.lattice_constant / 2.0


class BodyCenteredCubic(Lattice):
    """Body Centered Cubic (BCC) lattice."""

    def __init__(
        self,
        subdomain: Subdomain | PaddedSubdomain | VerletSubdomain,
        lattice_constant: float,
    ):
        super().__init__(subdomain, lattice_constant)

    def get_basis_vectors(self) -> NDArray[T]:
        """Return basis vectors for BCC lattice (corner and body center)."""
        return np.array([[0, 0, 0], [0.5, 0.5, 0.5]])

    def get_coordination_number(self) -> float:
        """Return coordination number for BCC lattice."""
        return 8.0

    def get_packing_efficiency(self) -> float:
        """Return packing efficiency (percent) for BCC lattice."""
        return 68.0  # percent

    def get_radius(self) -> float:
        """Return sphere radius for BCC lattice."""
        return self.lattice_constant * math.sqrt(3) / 4.0


class FaceCenteredCubic(Lattice):
    """Face Centered Cubic (FCC) lattice."""

    def __init__(
        self,
        subdomain: Subdomain | PaddedSubdomain | VerletSubdomain,
        lattice_constant: float,
    ):
        super().__init__(subdomain, lattice_constant)

    def get_basis_vectors(self) -> NDArray[T]:
        """Return basis vectors for FCC lattice."""
        return np.array(
            [
                [0, 0, 0],
                [0.5, 0.5, 0],
                [0.5, 0, 0.5],
                [0, 0.5, 0.5],
            ]
        )

    def get_coordination_number(self) -> float:
        """Return coordination number for FCC lattice."""
        return 12

    def get_packing_efficiency(self) -> float:
        """Return packing efficiency (percent) for FCC lattice."""
        return 74  # percent

    def get_radius(self) -> float:
        """Return sphere radius for FCC lattice."""
        return self.lattice_constant * math.sqrt(2) / 4.0
