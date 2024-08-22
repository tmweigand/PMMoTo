"""subdomain_padded.py"""

import numpy as np
from . import subdomain


class PaddedSubdomain(subdomain.Subdomain):
    """
    Padded subdomain to facilitate development of parallel algorithms
    """

    def __init__(
        self,
        subdomain,
        pad: tuple[int, int, int] = (0, 0, 0),
        reservoir_voxels=0,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.subdomain = subdomain
        self.pad = self.get_padding(pad)
        self.reservoir_pad = self.get_reservoir_padding(reservoir_voxels)
        self.start, self.index_own_nodes = self.update_start()
        self.voxels = self.get_voxels()
        self.box = self.get_padded_box()
        self.coords = self.get_coords()

    @classmethod
    def from_subdomain(cls, subdomain, pad, reservoir_voxels):
        return cls(
            rank=subdomain.rank,
            index=subdomain.index,
            neighbor_ranks=subdomain.neighbor_ranks,
            num_subdomains=subdomain.num_subdomains,
            domain_voxels=subdomain.domain_voxels,
            start=subdomain.start,
            box=subdomain.box,
            boundaries=subdomain.boundaries,
            inlet=subdomain.inlet,
            outlet=subdomain.outlet,
            voxels=subdomain.voxels,
            subdomain=subdomain,
            pad=pad,
            reservoir_voxels=reservoir_voxels,
        )

    def get_padding(self, pad: tuple[int, int, int]) -> tuple[tuple[int, int], ...]:
        """
        Add pad to boundaries of subdomain. Padding is only applied to the following boundaries
            -1: Internal subdomain boundary
             1: Wall boundary
             2: Periodic boundary

        Args:
            pad (tuple[int, int, int]): _description_

        Returns:
            tuple[int, int, int]: _description_
        """

        boundaries = []
        for n in range(self.dims):
            boundaries.append([self.boundaries[n * 2], self.boundaries[n * 2 + 1]])

        _pad = []
        for n, (minus, plus) in enumerate(boundaries):
            lower = 0
            if minus != 0:
                lower = pad[n]

            upper = 0
            if plus != 0:
                upper = pad[n]

            _pad.append((lower, upper))

        return tuple(_pad)

    def get_voxels(self) -> tuple[int, ...]:
        """
        Get the number of voxels with padding
        Uses voxels from subdomain class
        """
        voxels = []
        for n, (pad, r_pad) in enumerate(zip(self.pad, self.reservoir_pad)):
            voxels.append(self.voxels[n] + pad[0] + pad[1] + r_pad[0] + r_pad[1])

        return tuple(voxels)

    def get_padded_box(self) -> tuple[tuple[float, float], ...]:
        """
        Determine the box size of the padded domain
        """
        box = []
        for n, (pad, r_pad) in enumerate(zip(self.pad, self.reservoir_pad)):
            lower = self.box[n][0] - (pad[0] + r_pad[0]) * self.resolution[n]
            upper = self.box[n][1] + (pad[1] + r_pad[1]) * self.resolution[n]
            box.append((lower, upper))

        return tuple(box)

    def get_reservoir_padding(
        self, reservoir_voxels: int
    ) -> tuple[tuple[int, int], ...]:
        """
        Determine inlet/outlet info and pad grid but only inlet!
        """

        inlet = []
        for n in range(self.dims):
            inlet.append([self.inlet[n * 2], self.inlet[n * 2 + 1]])

        _pad = []
        for _inlet in inlet:
            lower = 0
            if _inlet[0]:
                lower = reservoir_voxels

            upper = 0
            if _inlet[1]:
                upper = reservoir_voxels

            _pad.append((lower, upper))

        return tuple(_pad)

    def update_start(self) -> tuple[int, ...]:
        """
        Update the start of the subdomain due to padding

        Returns:
            tuple[int,...]: start
        """
        _start = [0, 0, 0]
        index_own_nodes = np.zeros([6], dtype=np.int64)
        for dim, s in enumerate(self.start):
            _start[dim] = s - self.pad[dim][0] - self.reservoir_pad[dim][0]
            index_own_nodes[dim * 2] = s + self.pad[dim][0]
            index_own_nodes[dim * 2 + 1] = (
                index_own_nodes[dim * 2] + self.subdomain.voxels[dim]
            )

        return tuple(_start), index_own_nodes
