"""subdomain_verlet.py"""

from __future__ import annotations
from typing import TYPE_CHECKING
import numpy as np
from numpy.typing import NDArray
from ..core import subdomain_padded

if TYPE_CHECKING:
    from .domain_decompose import DecomposedDomain


class VerletSubdomain(subdomain_padded.PaddedSubdomain):
    """Verlet subdomains divide a subdomain into smaller Verlet domains."""

    def __init__(
        self,
        rank: int,
        decomposed_domain: DecomposedDomain,
        verlet_domains: tuple[int, ...],
        pad: tuple[int, ...] = (0, 0, 0),
        reservoir_voxels: int = 0,
    ):
        super().__init__(rank, decomposed_domain, pad, reservoir_voxels)
        self.verlet_domains = verlet_domains
        self.num_verlet = np.prod(verlet_domains)
        self.verlet_voxels = self.get_verlet_voxels()
        self.verlet_loop = self.get_verlet_loop()
        self.centroids = self.get_verlet_centroid()
        self.max_diameters = self.get_maximum_diameter()
        self.verlet_box = self.get_verlet_box()

    def get_verlet_voxels(self) -> tuple[tuple[int, ...], ...]:
        """Determine the number of voxels for each verlet subdomain"""
        verlet_voxels = [
            self.get_voxels(
                self.get_index(n, self.verlet_domains), self.voxels, self.verlet_domains
            )
            for n in range(self.num_verlet)
        ]
        return tuple(verlet_voxels)

    def get_verlet_loop(self) -> NDArray[np.uint]:
        """Collect the loop information for each verlet subdomain"""
        loop = np.zeros([self.num_verlet, 3, 2], dtype=np.uint)
        for n, voxels in enumerate(self.verlet_voxels):
            index = self.get_index(n, self.verlet_domains)
            for dim, ind in enumerate(index):
                # Remainder voxels are added at end so account for start
                if ind == self.verlet_domains[dim] - 1:
                    _voxels, _ = divmod(self.voxels[dim], self.verlet_domains[dim])
                    loop[n][dim, 0] = ind * _voxels
                else:
                    loop[n][dim, 0] = ind * voxels[dim]

                loop[n][dim, 1] = loop[n][dim, 0] + voxels[dim]

        return loop

    def get_verlet_centroid(self) -> NDArray[np.double]:
        """Determine the center of the verlet domain"""
        centroid = np.zeros([self.num_verlet, 3], dtype=np.double)
        for n in range(self.num_verlet):
            for dim in range(self.domain.dims):
                length = self.domain.resolution[dim] * self.verlet_voxels[n][dim]
                centroid[n, dim] = (
                    self.coords[dim][self.verlet_loop[n][dim, 0]]
                    + length / 2.0
                    - self.domain.resolution[dim] / 2.0
                )

        return centroid

    def get_maximum_diameter(self) -> NDArray[np.double]:
        """Determine the maximum diameter of the verlet subdomain"""
        max_diameter = np.zeros([self.num_verlet], dtype=np.double)
        for n in range(self.num_verlet):
            diameter: float = 0.0
            for dim in range(self.domain.dims):
                length = self.domain.resolution[dim] * self.verlet_voxels[n][dim]
                diameter += length * length

            max_diameter[n] = np.sqrt(diameter)

        return max_diameter

    def get_verlet_box(self) -> dict[int, tuple[tuple[int, int], ...]]:
        """Determine the box of the verlet subdomain"""
        box: dict[int, tuple[tuple[int, int], ...]] = {}
        for n in range(self.num_verlet):
            bounds = []
            for dim in range(self.domain.dims):
                length = self.domain.resolution[dim] * self.verlet_voxels[n][dim]
                lower = (
                    self.coords[dim][self.verlet_loop[n][dim, 0]]
                    - self.domain.resolution[dim] / 2.0
                )

                upper = lower + length
                bounds.append((lower, upper))

            box[n] = tuple(bounds)

        return box
