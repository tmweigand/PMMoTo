"""subdomain_padded.py

Defines the PaddedSubdomain class for handling subdomains with padding,
facilitating parallel algorithms in PMMoTo.
"""

from __future__ import annotations
from typing import Any, TYPE_CHECKING, TypeVar
import numpy as np
from numpy.typing import NDArray
from ..core import subdomain
from ..core import subdomain_features

if TYPE_CHECKING:
    from .domain_decompose import DecomposedDomain

T = TypeVar("T", bound=np.generic)


class PaddedSubdomain(subdomain.Subdomain):
    """Padded subdomain to facilitate development of parallel algorithms."""

    def __init__(
        self,
        rank: int,
        decomposed_domain: DecomposedDomain,
        pad: tuple[int, ...] = (0, 0, 0),
        reservoir_voxels: int = 0,
    ):
        """Initialize a PaddedSubdomain.

        Args:
            rank (int): Rank of the subdomain.
            decomposed_domain: Decomposed domain object.
            pad (tuple[int, int, int], optional): Padding for each dimension.
            reservoir_voxels (int, optional): Reservoir voxels to pad at inlet/outlet.

        """
        self.rank = rank
        self.domain = decomposed_domain
        self.index = self.get_index(self.rank, self.domain.subdomains)
        self.pad = self.get_padding(pad)
        self.inlet = self.get_inlet()
        self.outlet = self.get_outlet()

        self.reservoir_pad = self.get_reservoir_padding(reservoir_voxels)
        print(self.reservoir_pad)
        self.own_voxels = self.get_voxels(
            self.index, self.domain.voxels, self.domain.subdomains
        )
        self.own_box = self.get_box(self.own_voxels)
        self.voxels = self.get_padded_voxels()
        self.box = self.get_padded_box()
        self.global_boundary = self.get_global_boundary()

        self.neighbor_ranks = self.domain.get_neighbor_ranks(self.index)
        self.global_boundary = self.get_global_boundary()
        self.feature_boundary_types = self.get_boundary_types(
            self.global_boundary, self.neighbor_ranks
        )
        self.reservoir_pad = self.get_reservoir_padding(reservoir_voxels)

        self.start = self.get_start(
            self.index,
            self.domain.voxels,
            self.domain.subdomains,
            self.pad,
            self.reservoir_pad,
        )
        self.periodic = self.periodic_check()
        self.coords = self.get_coords(self.box, self.voxels, self.domain.resolution)
        self.features = subdomain_features.SubdomainFeatures(
            self, self.voxels, self.pad
        )

    def get_padding(self, pad: tuple[int, ...]) -> tuple[tuple[int, int], ...]:
        """Add pad to boundaries of subdomain.

        Padding is applied to all boundaries except 'end' and 'wall' boundary types,
        where pad is limited to 1. Padding must be equal on opposite features.

        Args:
            pad (tuple[int, ...]): Pad length for each dimension.

        Returns:
            tuple[tuple[int, int], ...]: Padding for each dimension.

        """
        dims = len(self.index)
        _pad: list[list[int]] = [[0, 0] for _ in range(dims)]

        for dim in range(dims):
            ind = self.index[dim]
            n_subdomains = self.domain.subdomains[dim]
            btypes = self.domain.boundary_types[dim]

            # Lower boundary
            if ind == 0:
                if btypes[0] == 0:  # 'end'
                    _pad[dim][0] = 0
                elif btypes[0] == 1:  # 'wall'
                    _pad[dim][0] = 1
                else:
                    _pad[dim][0] = pad[dim]
            else:
                _pad[dim][0] = pad[dim]

            # Upper boundary
            if ind == n_subdomains - 1:
                if btypes[1] == 0:  # 'end'
                    _pad[dim][1] = 0
                elif btypes[1] == 1:  # 'wall'
                    _pad[dim][1] = 1
                else:
                    _pad[dim][1] = pad[dim]
            else:
                _pad[dim][1] = pad[dim]

        return tuple((int(p[0]), int(p[1])) for p in _pad)

    def extend_padding(
        self, pad: tuple[int, ...]
    ) -> tuple[tuple[tuple[int, int], ...], dict[tuple[int, int, int], Any]]:
        """Extend pad to boundaries of subdomain.

        Padding is applied to all boundaries except 'end' and 'wall' boundary types,
        where pad is limited to 1. Padding must be equal on opposite features.

        Args:
            pad (tuple[int, ...]): Pad length for each dimension.

        Returns:
            tuple: (pad, loop) pad is the new padding; loop is a dict of extended loops.

        """
        _pad = [[0, 0], [0, 0], [0, 0]]
        for dim, ind in enumerate(self.index):
            if ind == 0 and self.domain.boundary_types[dim][0] in {0, 1}:
                _pad[dim][0] = 0
            else:
                _pad[dim][0] = pad[dim]
            if ind == self.domain.subdomains[dim] - 1 and self.domain.boundary_types[
                dim
            ][1] in {0, 1}:
                _pad[dim][1] = 0
            else:
                _pad[dim][1] = pad[dim]

        total_pad = [(s[0] + p[0], s[1] + p[1]) for s, p in zip(self.pad, _pad)]
        voxels = [v + p[0] + p[1] for v, p in zip(self.voxels, _pad)]

        extended_features = subdomain_features.SubdomainFeatures(
            self, voxels, tuple(total_pad)
        )

        return _pad, extended_features

    # def extend_padding(self, pad: tuple[int, ...]):
    #     """Extend pad to boundaries of subdomain.

    #     Padding is applied to all boundaries except 'end' and 'wall' boundary types,
    #     where pad is limited to 1. Padding must be equal on opposite features.

    #     Args:
    #         pad (tuple[int, ...]): Pad length for each dimension.

    #     Returns:
    #         tuple: (pad, loop) pad is the new padding; loop is a dict of extended loops.

    #     """
    #     _pad = [[0, 0], [0, 0], [0, 0]]
    #     for dim, ind in enumerate(self.index):
    #         if ind == 0 and self.domain.boundary_types[dim][0] in {0, 1}:
    #             _pad[dim][0] = 0
    #         else:
    #             _pad[dim][0] = pad[dim]
    #         if ind == self.domain.subdomains[dim] - 1 and self.domain.boundary_types[
    #             dim
    #         ][1] in {0, 1}:
    #             _pad[dim][1] = 0

    #         else:
    #             _pad[dim][1] = pad[dim]

    #     total_pad = [(s[0] + p[0], s[1] + p[1]) for s, p in zip(self.pad, _pad)]
    #     voxels = [v + p[0] + p[1] for v, p in zip(self.voxels, _pad)]
    #     pad = tuple(tuple(sublist) for sublist in _pad)

    #     # get loop for extended img
    #     loop = {}
    #     feature_types = ["faces", "edges", "corners"]
    #     for feature_type in feature_types:
    #         for feature_id, feature in self.features[feature_type].items():
    #             if feature.neighbor_rank > -1:
    #                 loop[feature_id] = subdomain_features.get_feature_voxels(
    #                     feature_id=feature_id,
    #                     voxels=voxels,
    #                     pad=total_pad,
    #                 )

    #     return pad, loop

    def get_padded_voxels(self) -> tuple[int, ...]:
        """Calculate number of voxels in each subdomain.

        This includes padding and reservoir.

        Returns:
            tuple[int, ...]: Number of voxels in each dimension.

        """
        voxels = [0, 0, 0]
        _voxels = self.get_voxels(
            self.index, self.domain.voxels, self.domain.subdomains
        )

        print(_voxels, self.pad, self.reservoir_pad)
        for n, (pad, r_pad) in enumerate(zip(self.pad, self.reservoir_pad)):
            voxels[n] = _voxels[n] + pad[0] + pad[1] + r_pad[0] + r_pad[1]

        return tuple(voxels)

    def get_padded_box(self) -> tuple[tuple[float, float], ...]:
        """Determine the bounding box for each subdomain.

        Note:
            Subdomains are divided such that voxel spacing is constant.

        Returns:
            tuple: Bounding box for each dimension.

        """
        box = []
        for dim, (ind, pad, r_pad) in enumerate(
            zip(self.index, self.pad, self.reservoir_pad)
        ):
            length = (
                self.voxels[dim] - (pad[0] + pad[1] + r_pad[0] + r_pad[1])
            ) * self.domain.resolution[dim]
            lower = self.domain.box[dim][0] + length * ind
            if ind == self.domain.subdomains[dim] - 1:
                lower = self.domain.box[dim][1] - length
            upper = lower + length
            box.append((lower, upper))

        for n, (pad, r_pad) in enumerate(zip(self.pad, self.reservoir_pad)):
            lower = box[n][0] - (pad[0] + r_pad[0]) * self.domain.resolution[n]
            upper = box[n][1] + (pad[1] + r_pad[1]) * self.domain.resolution[n]
            box[n] = (lower, upper)

        return tuple(box)

    def get_reservoir_padding(
        self, reservoir_voxels: int
    ) -> tuple[tuple[int, int], ...]:
        """Determine inlet/outlet info and pad image (only inlet).

        Args:
            reservoir_voxels (int): Number of reservoir voxels to pad.

        Returns:
            tuple[tuple[int, int], ...]: Reservoir padding for each dimension.

        """
        if reservoir_voxels == 0:
            return ((0, 0), (0, 0), (0, 0))

        pad: list[tuple[int, int]] = []
        for dim in range(3):
            before = reservoir_voxels if self.inlet[dim][0] else 0
            after = reservoir_voxels if self.inlet[dim][1] else 0
            pad.append((before, after))
        return pad

    @staticmethod
    def get_start(
        index: tuple[int, ...],
        domain_voxels: tuple[int, ...],
        subdomains: tuple[int, ...],
        pad: tuple[tuple[int, int], ...] = ((0, 0), (0, 0), (0, 0)),
        reservoir_pad: tuple[tuple[int, int], ...] = ((0, 0), (0, 0), (0, 0)),
    ) -> tuple[int, ...]:
        """Determine the start of the subdomain. used for saving as vtk
        Start is the minimum voxel ID
        Args:
            index (tuple[int, int, int]): subdomain index

        Returns:
            tuple[int,...]: start

        """
        _start = [0, 0, 0]

        for dim, _index in enumerate(index):
            sd_voxels, _ = divmod(domain_voxels[dim], subdomains[dim])
            _start[dim] = sd_voxels * _index

        start = [0, 0, 0]
        for dim, s in enumerate(_start):
            start[dim] = s - pad[dim][0] - reservoir_pad[dim][0]

        return tuple(start)

    def get_own_voxels(self) -> NDArray[np.int64]:
        """Determine the index for the voxels owned by this subdomain.

        Returns:
            np.ndarray: Array of indices for owned voxels.

        """
        own_voxels = np.zeros([6], dtype=np.int64)
        for dim, (pad, r_pad) in enumerate(zip(self.pad, self.reservoir_pad)):
            own_voxels[dim * 2] = pad[0] + r_pad[0]
            own_voxels[dim * 2 + 1] = self.voxels[dim] - pad[1] - r_pad[1]

        return own_voxels

    def update_reservoir(self, img: NDArray[T], value: T) -> NDArray[T]:
        """Enforce a constant value in reservoir regions.

        Args:
            img (np.ndarray): Image array to update.
            value: Value to set in reservoir regions.

        Returns:
            np.ndarray: Updated image array.

        """
        for dim, (start_pad, end_pad) in enumerate(self.reservoir_pad):
            if start_pad > 0:
                idx = [slice(None)] * img.ndim
                idx[dim] = slice(0, start_pad)
                img[tuple(idx)] = value

            if end_pad > 0:
                idx = [slice(None)] * img.ndim
                idx[dim] = slice(-end_pad, None)
                img[tuple(idx)] = value

        return img
