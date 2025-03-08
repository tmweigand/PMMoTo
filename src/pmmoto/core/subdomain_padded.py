"""subdomain_padded.py"""

import warnings
import numpy as np
from ..core import subdomain
from ..core import subdomain_features


class PaddedSubdomain(subdomain.Subdomain):
    """
    Padded subdomain to facilitate development of parallel algorithms
    """

    def __init__(
        self,
        rank: int,
        decomposed_domain,
        pad: tuple[int, int, int] = (0, 0, 0),
        reservoir_voxels=0,
    ):
        self.rank = rank
        self.domain = decomposed_domain
        self.index = self.get_index(self.rank, self.domain.subdomains)
        self.pad = self.get_padding(pad)
        self.inlet = self.get_inlet()

        self.reservoir_pad = self.get_reservoir_padding(reservoir_voxels)
        self.own_voxels = self.get_voxels(
            self.index, self.domain.voxels, self.domain.subdomains
        )
        self.own_box = self.get_box(self.own_voxels)
        self.voxels = self.get_padded_voxels()
        self.box = self.get_padded_box()
        self.global_boundary = self.get_global_boundary()

        self.neighbor_ranks = self.domain.get_neighbor_ranks(self.index)
        self.global_boundary = self.get_global_boundary()
        self.boundary_types = self.get_boundary_types(
            self.global_boundary, self.neighbor_ranks
        )
        self.reservoir_pad = self.get_reservoir_padding(reservoir_voxels)
        self.voxels = self.get_voxels()
        self.box = self.get_box()

        self.outlet = self.get_outlet()
        self.start = self.get_start()
        self.periodic = self.periodic_check()
        self.coords = self.get_coords(self.box, self.voxels, self.domain.resolution)
        self.features = subdomain_features.collect_features(
            self.neighbor_ranks,
            self.global_boundary,
            self.boundary_types,
            self.voxels,
            self.inlet,
            self.outlet,
            self.pad,
            self.reservoir_pad,
        )

    def get_padding(self, pad: tuple[int, ...]) -> tuple[tuple[int, int], ...]:
        """
        Add pad to boundaries of subdomain. Padding is applied to all boundaries
        except 'end' boundary type and 'wall' boundary type, where pad is limited to 1.
        Check is performed for wall boundary conditions if trying to extend an image which this function is used for.
        Padding must be equal on opposite feature!
        Args:
            pad (tuple[int, ...]): pad length for each dimension

        Returns:
             tuple[tuple[int, int], ...]: list of length == dim
        """

        _pad = [[0, 0], [0, 0], [0, 0]]
        for dim, ind in enumerate(self.index):
            if ind == 0 and self.domain.boundary_types[dim][0] == 0:
                _pad[dim][0] = 0
            elif ind == 0 and self.domain.boundary_types[dim][0] == 1:
                if hasattr(self, "pad") and self.pad[dim][0] == 1:
                    _pad[dim][0] = 0
                else:
                    _pad[dim][0] = 1
            else:
                _pad[dim][0] = pad[dim]
            if (
                ind == self.domain.subdomains[dim] - 1
                and self.domain.boundary_types[dim][1] == 0
            ):
                _pad[dim][1] = 0
            elif (
                ind == self.domain.subdomains[dim] - 1
                and self.domain.boundary_types[dim][1] == 1
            ):
                _pad[dim][1] = 1
            else:
                _pad[dim][1] = pad[dim]

        return tuple(tuple(sublist) for sublist in _pad)

    def extend_padding(self, pad: tuple[int, ...]):
        """
        Extend pad to boundaries of subdomain. Padding is applied to all boundaries
        except 'end' boundary type and 'wall' boundary type, where pad is limited to 1.
        Check is performed for wall boundary conditions if trying to extend an image which this function is used for.
        Padding must be equal on opposite feature!
        Args:
            pad (tuple[int, ...]): pad length for each dimension

        Returns:
             dict:
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
        pad = tuple(tuple(sublist) for sublist in _pad)

        # get loop for extended img
        loop = {}
        feature_types = ["faces", "edges", "corners"]
        for feature_type in feature_types:
            for feature_id, feature in self.features[feature_type].items():
                if feature.neighbor_rank > -1:
                    loop[feature_id] = subdomain_features.get_feature_voxels(
                        feature_id=feature_id,
                        voxels=voxels,
                        pad=total_pad,
                    )

        return pad, loop

    def get_padded_voxels(self) -> tuple[int, ...]:
        """
        Calculate number of voxels in each subdomain.
        This can be very bad when voxels ~= ranks or something like that
        """
        voxels = [0, 0, 0]
        _voxels = self.get_voxels(
            self.index, self.domain.voxels, self.domain.subdomains
        )

        for n, (pad, r_pad) in enumerate(zip(self.pad, self.reservoir_pad)):
            voxels[n] = _voxels[n] + pad[0] + pad[1] + r_pad[0] + r_pad[1]

        return tuple(voxels)

    def get_padded_box(self):
        """
        Determine the bounding box for each subdomain.
        Note: subdomains are divided such that voxel spacing is constant
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
        """
        Determine inlet/outlet info and pad img but only inlet!
        Convert to tuple of tuples - overly complicated
        """
        if reservoir_voxels == 0:
            return ((0, 0), (0, 0), (0, 0))

        _pad = [0, 0, 0, 0, 0, 0]
        for n, is_inlet in enumerate(self.inlet):
            if is_inlet:
                _pad[n] = reservoir_voxels

        return tuple((_pad[i], _pad[i + 1]) for i in range(0, len(_pad), 2))

    def get_start(self) -> tuple[int, ...]:
        """
        Determine the start of the subdomain. used for saving as vtk
        Start is the minimum voxel ID
        Args:
            sd_index (tuple[int, int, int]): subdomain index

        Returns:
            tuple[int,...]: start
        """
        _start = [0, 0, 0]

        for dim, _index in enumerate(self.index):
            sd_voxels, _ = divmod(self.domain.voxels[dim], self.domain.subdomains[dim])
            _start[dim] = sd_voxels * _index

        start = [0, 0, 0]
        for dim, s in enumerate(_start):
            start[dim] = s - self.pad[dim][0] - self.reservoir_pad[dim][0]

        return tuple(start)

    def get_own_voxels(self):
        """
        Determine the index for the voxels owned by this subdomain

        Returns:
            _type_: _description_
        """
        own_voxels = np.zeros([6], dtype=np.int64)
        for dim, (pad, r_pad) in enumerate(zip(self.pad, self.reservoir_pad)):
            own_voxels[dim * 2] = pad[0] + r_pad[0]
            own_voxels[dim * 2 + 1] = self.voxels[dim] - pad[1] - r_pad[1]

        return own_voxels

    def update_reservoir(self, img, value):
        """
        Enforce a constant value in reservoir
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
