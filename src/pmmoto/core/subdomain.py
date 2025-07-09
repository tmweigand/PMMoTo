"""subdomains.py

Defines the Subdomain class for domain decomposition and parallelization in PMMoTo.
"""

from __future__ import annotations
from typing import Any, TYPE_CHECKING, TypeVar
import numpy as np
from numpy.typing import NDArray
from .boundary_types import BoundaryType
from . import domain_discretization
from . import subdomain_features

if TYPE_CHECKING:
    from .domain_decompose import DecomposedDomain

T = TypeVar("T", bound=np.generic)


class Subdomain(domain_discretization.DiscretizedDomain):
    """Decompose the domain into subdomains for parallelization."""

    def __init__(
        self,
        rank: int,
        decomposed_domain: DecomposedDomain,
    ):
        """Initialize a Subdomain.

        Args:
            rank (int): Rank of the subdomain.
            decomposed_domain: Decomposed domain object.

        """
        self.rank = rank
        self.domain = decomposed_domain
        self.index = self.get_index(self.rank, self.domain.subdomains)
        self.voxels = self.get_voxels(
            self.index, self.domain.voxels, self.domain.subdomains
        )
        self.box = self.get_box(self.voxels)
        self.global_boundary = self.get_global_boundary()
        self.boundary_types = self.get_boundary_types()
        self.neighbor_ranks = self.domain.get_neighbor_ranks(self.index)

        self.inlet = self.get_inlet()
        self.outlet = self.get_outlet()
        self.start = self.get_start(
            self.index, self.domain.voxels, self.domain.subdomains
        )
        self.periodic = self.periodic_check()
        self.coords = self.get_coords(self.box, self.voxels, self.domain.resolution)
        self.features = subdomain_features.SubdomainFeatures(self, self.voxels)

    @staticmethod
    def get_index(rank: int, subdomains: tuple[int, ...]) -> tuple[int, ...]:
        """Determine the index of the subdomain in the decomposition.

        Args:
            rank (int): The rank of the subdomain.
            subdomains (tuple[int, int, int]): Number of subdomains in each dimension.

        Returns:
            tuple[int, int, int]: Index of the subdomain in the decomposition grid.

        """
        return tuple(int(i) for i in np.unravel_index(rank, subdomains))

    @staticmethod
    def get_voxels(
        index: tuple[int, ...],
        domain_voxels: tuple[int, ...],
        subdomains: tuple[int, ...],
    ) -> tuple[int, ...]:
        """Calculate number of voxels in each subdomain.

        Args:
            index (tuple): Index of the subdomain.
            domain_voxels (tuple): Number of voxels in the full domain.
            subdomains (tuple): Number of subdomains in each dimension.

        Returns:
            tuple[int, int, int] Number of voxels in each dimension for this subdomain.

        """
        voxels = [0 for _ in range(len(index))]
        for dim, ind in enumerate(index):
            sd_voxels, rem_sd_voxels = divmod(domain_voxels[dim], subdomains[dim])
            if ind == subdomains[dim] - 1:
                voxels[dim] = sd_voxels + rem_sd_voxels
            else:
                voxels[dim] = sd_voxels

        return tuple(voxels)

    def get_box(self, voxels: tuple[int, ...]) -> tuple[tuple[float, float], ...]:
        """Determine the bounding box for each subdomain.

        Note:
            Subdomains are divided such that voxel spacing is constant.

        Args:
            voxels (tuple): Number of voxels in each dimension.

        Returns:
            tuple: Bounding box for each dimension.

        """
        box = []
        for dim, ind in enumerate(self.index):
            length = voxels[dim] * self.domain.resolution[dim]
            lower = self.domain.box[dim][0] + length * ind
            if ind == self.domain.subdomains[dim] - 1:
                lower = self.domain.box[dim][1] - length
            upper = lower + length
            box.append((lower, upper))

        return tuple(box)

    def get_length(self) -> tuple[float, ...]:
        """Calculate the length of the subdomain in each dimension.

        Returns:
            tuple[float, ...]: Length in each dimension.


        """
        length = np.zeros([self.domain.dims], dtype=np.float64)
        for n in range(0, self.domain.dims):
            length[n] = self.box[n][1] - self.box[n][0]

        return tuple(length)

    def get_global_boundary(self) -> tuple[tuple[bool, bool], ...]:
        """Determine if the subdomain is on the domain boundary.

        For a subdomain to be on a domain boundary, the subdomain index must contain
        either 0 or the number of subdomains.

        Returns:
            tuple[tuple[bool,bool],...]: Tuple of bools whether face is on boundary

        """
        global_boundary: list[tuple[bool, bool]] = []
        for ind, subdomains in zip(self.index, self.domain.subdomains):
            ### Conditions for a domain boundary feature
            if ind == 0:
                lower = True
            else:
                lower = False

            if ind == subdomains - 1:
                upper = True
            else:
                upper = False

            global_boundary.append((lower, upper))

        return tuple(global_boundary)

    def get_boundary_types(self) -> tuple[tuple[BoundaryType, BoundaryType], ...]:
        """Determine the boundary type for each subdomain.

        Returns:
            tuple[tuple[str,str],...]: Tuple specifying boundary type for each face

        """
        boundary_type: list[tuple[BoundaryType, BoundaryType]] = []
        for ind, subdomain, b_types in zip(
            self.index, self.domain.subdomains, self.domain.boundary_types
        ):
            if ind == 0:
                lower = b_types[0]
            else:
                lower = BoundaryType.INTERNAL

            if ind == subdomain - 1:
                upper = b_types[1]
            else:
                upper = BoundaryType.INTERNAL

            boundary_type.append((lower, upper))

        return tuple(boundary_type)

    def get_inlet(self) -> tuple[tuple[bool, bool], ...]:
        """Determine if the subdomain lies on the inlet boundaries.

        A subdomain is on an inlet if:
        - It is at the edge of the global domain (index is 0 or max in a dimension).
        - The corresponding global boundary type is `0` (denoting an inlet).

        Returns:
            tuple[tuple[bool, bool], ...]: bool if subdomain face is on inlet.

        """
        inlet: list[tuple[bool, bool]] = []

        for dim, ind in enumerate(self.index):
            lower = False
            upper = False

            # Lower Domain face
            if ind == 0 and self.domain.boundary_types[dim][0] == BoundaryType.END:
                lower = bool(self.domain.inlet[dim][0])

            # Upper Domain face
            if (
                ind == self.domain.subdomains[dim] - 1
                and self.domain.boundary_types[dim][1] == BoundaryType.END
            ):
                upper = bool(self.domain.inlet[dim][1])

            inlet.append((lower, upper))

        return tuple(inlet)

    def get_outlet(self) -> tuple[tuple[bool, bool], ...]:
        """Determine if the subdomain lies on the inlet boundaries.

        A subdomain is on an outlet if:
        - It is at the edge of the global domain (index is 0 or max in a dimension).
        - The corresponding global boundary type is `0` (denoting an outlet).

        Returns:
            tuple[tuple[bool, bool], ...]: bool if subdomain face is on outlet.

        """
        outlet: list[tuple[bool, bool]] = []

        for dim, ind in enumerate(self.index):
            lower = False
            upper = False

            # Lower face
            if ind == 0 and self.domain.boundary_types[dim][0] == BoundaryType.END:
                lower = bool(self.domain.outlet[dim][0])

            # Upper face
            if (
                ind == self.domain.subdomains[dim] - 1
                and self.domain.boundary_types[dim][1] == BoundaryType.END
            ):
                upper = bool(self.domain.outlet[dim][1])

            outlet.append((lower, upper))

        return tuple(outlet)

    @staticmethod
    def get_start(
        index: tuple[int, ...],
        domain_voxels: tuple[float, ...],
        subdomains: tuple[int, ...],
    ) -> tuple[int, ...]:
        """Determine the start of the subdomain.

        Args:
            index (tuple[int, int, int]): Subdomain index.
            domain_voxels (tuple[float, float, float]): Number of voxels per dimension
            subdomains (tuple[int, int, int]): Number of subdomains per dimension.

        Returns:
            tuple[int, int, int]: Start voxel indices - minimum voxel ID.

        """
        start_0 = int(domain_voxels[0] // subdomains[0]) * index[0]
        start_1 = int(domain_voxels[1] // subdomains[1]) * index[1]
        start_2 = int(domain_voxels[2] // subdomains[2]) * index[2]

        return (start_0, start_1, start_2)

    def set_wall_bcs(self, img: NDArray[T]) -> NDArray[T]:
        """Force solid on external boundaries if wall boundary conditions are specified.

        Args:
            img (np.ndarray): Image array to update.

        Returns:
            np.ndarray: Updated image array with wall boundaries set to solid.

        """
        for feature in self.features.faces.values():
            if feature.boundary_type == BoundaryType.WALL:
                img[feature.slice] = 0

        return img

    def get_centroid(self) -> NDArray[np.float64]:
        """Determine the centroid of a subdomain.

        Returns:
            np.ndarray: Centroid coordinates.

        """
        centroid = np.zeros(3, dtype=np.float64)
        for dim, side in enumerate(self.box):
            centroid[dim] = side[0] + 0.5 * (side[1] - side[0])

        return centroid

    def get_radius(self) -> Any:
        """Determine the radius of a subdomain.

        Returns:
            float: Radius of the subdomain.

        """
        diameter = 0.0
        for side in self.box:
            length = side[1] - side[0]
            diameter += length * length

        return np.sqrt(diameter) / 2

    def get_origin(self) -> tuple[float, ...]:
        """Determine the domain origin from box.

        Returns:
            tuple[float, ...]: Domain origin.

        """
        origin = [0.0 for _ in self.box]
        for n, box_dim in enumerate(self.box):
            origin[n] = box_dim[0]

        return tuple(origin)

    def get_img_index(self, coordinates: tuple[float, ...]) -> tuple[int, ...] | None:
        """Given coordinates, return the corresponding index in the img array.

        Args:
            coordinates (tuple[float, float, float]): The (x, y, z) coordinates.

        Returns:
            tuple[int, int, int] or None: The (i, j, k) index in the img array,
            or None if out of bounds.

        """
        indices = [0] * len(coordinates)
        for dim, coord in enumerate(coordinates):
            # Calculate the index based on the coordinate, origin, and resolution
            indices[dim] = int((coord - self.box[dim][0]) / self.domain.resolution[dim])

            # Ensure the index is within bounds
            if indices[dim] < 0 or indices[dim] >= self.voxels[dim]:
                return None

        return tuple(indices)

    def get_own_voxels(self) -> NDArray[np.integer[Any]]:
        """Determine the index for the voxels owned by this subdomain.

        Returns:
            np.ndarray: Array of indices for owned voxels.

        """
        own_voxels = np.zeros([6], dtype=np.int64)
        for dim, voxels in enumerate(self.voxels):
            own_voxels[dim * 2 + 1] = voxels

        return own_voxels

    def periodic_check(self) -> bool:
        """Determine if subdomain is periodic

        Returns: bool if and feature is periodic
        """
        for b_type in self.boundary_types:
            if b_type[0] == BoundaryType.PERIODIC or b_type[1] == BoundaryType.PERIODIC:
                return True

        return False

    def check_boundary_type(self, type: BoundaryType) -> bool:
        """Determine if boundary type on subdomain."""
        return any(type in inner for inner in self.boundary_types)
