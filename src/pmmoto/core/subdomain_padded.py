"""subdomain_padded.py"""

from . import subdomain


class PaddedSubdomain(subdomain.Subdomain):
    """
    Padded subdomain to facilitate development of parallel algorithms
    """

    def __init__(self, pad: tuple[int, int, int] = (0, 0, 0), **kwargs):
        super().__init__(**kwargs)
        self.pad = self.get_padding(pad)
        self.voxels = self.get_voxels()
        self.box = self.get_padded_box()
        self.coords = self.get_coords()

    @classmethod
    def from_subdomain(cls, subdomain, pad):
        return cls(
            rank=subdomain.rank,
            index=subdomain.index,
            box=subdomain.box,
            boundaries=subdomain.boundaries,
            inlet=subdomain.inlet,
            outlet=subdomain.outlet,
            voxels=subdomain.voxels,
            pad=pad,
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
        _pad = []
        for n, (minus, plus) in enumerate(self.boundaries):
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
        USes voxels from subdomain class
        """
        voxels = []
        for n, pad in enumerate(self.pad):
            voxels.append(self.voxels[n] + pad[0] + pad[1])

        return tuple(voxels)

    def get_padded_box(self) -> tuple[tuple[float, float], ...]:
        """
        Determine the box size of the padded domain
        """
        box = []
        for n, pad in enumerate(self.pad):
            lower = self.box[n][0] - pad[0] * self.resolution[n]
            upper = self.box[n][1] - pad[1] * self.resolution[n]
            box.append((lower, upper))

        return tuple(box)
