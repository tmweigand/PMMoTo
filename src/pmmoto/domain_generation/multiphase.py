"""multiphase.py

Defines the Multiphase class for handling multiphase images and related calculations.
"""

from __future__ import annotations
from typing import TypeVar, TYPE_CHECKING, Generic
import numpy as np
from numpy.typing import NDArray
from ..core import communication
from ..core import utils
from ..core.subdomain_padded import PaddedSubdomain
from ..core.subdomain_verlet import VerletSubdomain
from .porousmedia import PorousMedia

if TYPE_CHECKING:
    from ..core.subdomain import Subdomain

T = TypeVar("T", bound=np.generic)


class Multiphase(Generic[T]):
    """Class for handling multiphase images and phase calculations."""

    def __init__(self, porous_media: PorousMedia, img: NDArray[T], num_phases: int):
        """Initialize a Multiphase object.

        Args:
            porous_media: PorousMedia object.
            img (np.ndarray): Multiphase image.
            num_phases (int): Number of phases.

        """
        self.porous_media: PorousMedia = porous_media
        self.pm_img = self.porous_media.img
        self.subdomain: Subdomain | PaddedSubdomain | VerletSubdomain = (
            porous_media.subdomain
        )
        self.num_phases: int = num_phases
        self.img: NDArray[T] = img
        self.fluids: list[int] = list(range(1, num_phases + 1))

    def update_img(self, img: NDArray[T]) -> None:
        """Update the multiphase image.

        Args:
            img (np.ndarray): New multiphase image.

        """
        if isinstance(self.subdomain, (PaddedSubdomain, VerletSubdomain)):
            img = self.subdomain.update_reservoir(img, img.dtype.type(1))
        else:
            raise TypeError("subdomain does not support update_reservoir")
        self.img = img

    def get_volume_fraction(self, phase: int, img: None | NDArray[T] = None) -> float:
        """Calculate the volume fraction of a given phase in a multiphase image.

        Args:
            phase (int): The phase ID to compute the volume fraction for.
            img (np.ndarray, optional): Image to use. Defaults to self.img.

        Returns:
            float: The volume fraction of the specified phase.

        """
        if img is None:
            img = self.img

        local_img = utils.own_img(self.subdomain, img)
        local_voxel_count = np.count_nonzero(local_img == phase)

        total_voxel_count: int = (
            communication.all_reduce(local_voxel_count)
            if self.subdomain.domain.num_subdomains > 1
            else local_voxel_count
        )

        total_voxels = np.prod(self.subdomain.domain.voxels)
        return float(total_voxel_count / total_voxels)

    def get_saturation(self, phase: int, img: None | NDArray[T] = None) -> float:
        """Calculate the saturation of a multiphase image.

        Args:
            phase (int): The phase ID to compute the saturation for.
            img (np.ndarray, optional): Image to use. Defaults to self.img.

        Returns:
            float: The saturation of the specified phase.

        """
        if img is None:
            img = self.img
        return self.get_volume_fraction(phase, img) / self.porous_media.porosity

    @staticmethod
    def get_probe_radius(
        pc: float, gamma: float = 1, contact_angle: float = 0
    ) -> float:
        """Return the probe radius.

        Args:
            pc (float): Capillary pressure.
            gamma (float, optional): Surface tension. Defaults to 1.
            contact_angle (float, optional): Contact angle in degrees. Defaults to 0.

        Returns:
            float: Probe radius.

        """
        if pc == 0:
            r_probe = 0
        else:
            r_probe = 2.0 * gamma / pc * np.cos(np.deg2rad(contact_angle))
        return r_probe

    @staticmethod
    def get_pc(radius: float, gamma: float = 1) -> float:
        """Return the capillary pressure given a surface tension and radius.

        Args:
            radius (float): Probe radius.
            gamma (float, optional): Surface tension. Defaults to 1.

        Returns:
            float: Capillary pressure.

        """
        return 2.0 * gamma / radius
