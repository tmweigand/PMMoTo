"""subdomain_padded.py"""

import numpy as np
from . import subdomain
from . import subdomain_features


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
        self.index = self.get_index(self.domain.subdomains)
        self.pad = self.get_padding(
            pad, self.index, self.domain.subdomains, self.domain.boundary_types
        )
        self.inlet = self.get_inlet(
            self.index, self.domain.inlet, self.domain.subdomains
        )

        self.reservoir_pad = self.get_reservoir_padding(reservoir_voxels)
        self.voxels = self.get_voxels(
            self.index, self.domain.voxels, self.domain.subdomains
        )
        self.box = self.get_box(
            self.index,
            self.voxels,
            self.domain.box,
            self.domain.resolution,
            self.domain.subdomains,
        )
        self.boundaries, self.boundary_types = self.get_boundaries(
            self.index, self.domain.boundary_types, self.domain.subdomains
        )
        self.outlet = self.get_outlet(
            self.index, self.domain.outlet, self.domain.subdomains
        )
        self.start = self.get_start(
            self.index, self.domain.voxels, self.domain.subdomains
        )
        self.neighbor_ranks = self.domain.get_neighbor_ranks(self.index)
        self.periodic = self.periodic_check()
        self.boundary = self.boundary_check()
        self.coords = self.get_coords(self.box, self.voxels, self.domain.resolution)
        self.features = subdomain_features.collect_features(
            self.neighbor_ranks,
            self.boundaries,
            self.boundary_types,
            self.voxels,
            self.inlet,
        )
        self.features = subdomain_features.set_padding(
            self.features, self.voxels, self.pad, self.reservoir_pad
        )

    def get_padding(
        self,
        pad: tuple[int, int, int],
        index: tuple[np.intp, ...],
        subdomains,
        boundary_types,
    ) -> tuple[tuple[int, int], ...]:
        """
        Add pad to boundaries of subdomain. Padding is applied to all boundaries
        except 'end' boundary type.
        Args:
            pad (tuple[int, int, int]): _description_

        Returns:
            tuple[int, int, int]: _description_
        """
        _pad = [[0, 0], [0, 0], [0, 0]]
        for dim, ind in enumerate(index):
            if ind == 0 and boundary_types[dim][0] == "end":
                _pad[dim][0] = 0
            else:
                _pad[dim][0] = pad[dim]
            if ind == subdomains[dim] - 1 and boundary_types[dim][1] == "end":
                _pad[dim][1] = 0
            else:
                _pad[dim][1] = pad[dim]

        # feature_types = ["faces"]
        # for feature_type in feature_types:
        #     for feature_id, feature in self.features[feature_type].items():
        #         index = feature.info["argOrder"][0]
        #         if self.boundary_types[feature_id] != "end":
        #             if feature_id[index] < 0:
        #                 _pad[index][0] = pad[index]
        #             elif feature_id[index] > 0:
        #                 _pad[index][1] = pad[index]

        return tuple(tuple(sublist) for sublist in _pad)

    def get_voxels(
        self,
        index: tuple[np.intp, ...],
        domain_voxels: tuple[int, int, int],
        subdomains,
    ) -> tuple[int, ...]:
        """
        Calculate number of voxels in each subdomain.
        This can be very bad when voxels ~= ranks or something like that
        """
        voxels = [0, 0, 0]
        for dim, ind in enumerate(index):
            sd_voxels, rem_sd_voxels = divmod(domain_voxels[dim], subdomains[dim])
            if ind == subdomains[dim] - 1:
                voxels[dim] = sd_voxels + rem_sd_voxels
            else:
                voxels[dim] = sd_voxels

        for n, (pad, r_pad) in enumerate(zip(self.pad, self.reservoir_pad)):
            voxels[n] = voxels[n] + pad[0] + pad[1] + r_pad[0] + r_pad[1]

        return tuple(voxels)

    def get_box(
        self,
        index: tuple[np.intp, ...],
        voxels: tuple[int, int, int],
        domain_box,
        resolution: tuple[int, int, int],
        subdomains: tuple[int, int, int],
    ):
        """
        Determine the bounding box for each subdomain.
        Note: subdomains are divided such that voxel spacing
        is constant
        """
        box = []
        for dim, ind in enumerate(index):
            length = voxels[dim] * resolution[dim]
            lower = domain_box[dim][0] + length * ind
            if ind == subdomains[dim] - 1:
                lower = domain_box[dim][1] - length
            upper = lower + length
            box.append((lower, upper))

        for n, (pad, r_pad) in enumerate(zip(self.pad, self.reservoir_pad)):
            lower = box[n][0] - (pad[0] + r_pad[0]) * resolution[n]
            upper = box[n][1] + (pad[1] + r_pad[1]) * resolution[n]
            box[n] = (lower, upper)

        return tuple(box)

    def get_reservoir_padding(
        self, reservoir_voxels: int
    ) -> tuple[tuple[int, int], ...]:
        """
        Determine inlet/outlet info and pad grid but only inlet!
        Convert to tuple of tuples - overly complicated
        """

        _pad = [0, 0, 0, 0, 0, 0]
        for n, is_inlet in enumerate(self.inlet):
            if is_inlet:
                _pad[n] = reservoir_voxels

        return tuple((_pad[i], _pad[i + 1]) for i in range(0, len(_pad), 2))

    def get_start(
        self,
        index: tuple[int, int, int],
        domain_voxels,
        subdomains: tuple[int, int, int],
    ) -> tuple[int, ...]:
        """
        Determine the start of the subdomain. used for saving as vtk
        Start is the minimum voxel ID
        Args:
            sd_index (tuple[int, int, int]): subdomain index

        Returns:
            tuple[int,...]: start
        """
        _start = [0, 0, 0]

        for dim, _index in enumerate(index):
            sd_voxels, _ = divmod(domain_voxels[dim], subdomains[dim])
            _start[dim] = sd_voxels * _index

        start = [0, 0, 0]
        for dim, s in enumerate(_start):
            start[dim] = s - self.pad[dim][0] - self.reservoir_pad[dim][0]

        return tuple(start)

    def get_index_own_nodes(self):
        """ """
        index_own_nodes = np.zeros([6], dtype=np.int64)
        for dim, s in enumerate(self.start):
            index_own_nodes[dim * 2] = s + self.pad[dim][0]
            index_own_nodes[dim * 2 + 1] = (
                index_own_nodes[dim * 2] + self.voxels[dim]  ### MAYBE BUG HERE
            )

        return index_own_nodes
