"""porousmedia.py

Defines the PorousMedia class for representing and analyzing porous media,
including porosity and distance transform calculations.
"""

from __future__ import annotations
from typing import TypeVar
import numpy as np
from numpy.typing import NDArray
from ..core import communication
from ..core import utils
from ..filters import distance
from ..core.subdomain import Subdomain
from ..core.subdomain_padded import PaddedSubdomain
from ..core.subdomain_verlet import VerletSubdomain

T = TypeVar("T", bound=np.generic)


class PorousMedia:
    """Porous media class for storing image data and computing properties."""

    def __init__(
        self,
        subdomain: Subdomain | PaddedSubdomain | VerletSubdomain,
        img: NDArray[np.uint8],
    ):
        """Initialize a PorousMedia object.

        Args:
            subdomain: Subdomain object.
            img (np.ndarray): Binary image of the porous medium.

        """
        self.subdomain: Subdomain | PaddedSubdomain | VerletSubdomain = subdomain
        self.img = img
        self._porosity: None | float = None
        self._distance: None | NDArray[np.float32] = None
        self._max_distance: None | float = None

    @property
    def porosity(self) -> float:
        """Get the porosity of the porous medium.

        Returns:
            float: Porosity value (fraction of pore voxels).

        """
        if self._porosity is None:
            self.set_porosity()
        assert self._porosity is not None

        return self._porosity

    def set_porosity(self) -> None:
        """Calculate and set the porosity of the porous medium."""
        own_img = utils.own_img(self.subdomain, self.img)
        local_pore_voxels = np.count_nonzero(own_img == 1)  # One is pore space

        if self.subdomain.domain.num_subdomains > 1:
            global_pore_voxels = communication.all_reduce(local_pore_voxels)
        else:
            global_pore_voxels = local_pore_voxels

        self._porosity = global_pore_voxels / np.prod(self.subdomain.domain.voxels)

    @property
    def distance(self) -> NDArray[np.float32]:
        """Get the Euclidean distance transform of the porous medium.

        Returns:
            np.ndarray: Distance transform array.

        """
        if self._distance is None:
            self.set_distance()
        assert self._distance is not None

        return self._distance

    @property
    def max_distance(self) -> float:
        """Get the maximum distance from the distance transform.

        Returns:
            float: Maximum distance value.

        """
        if self._max_distance is None:
            self._max_distance = utils.determine_maximum(self.distance)
        assert self._max_distance is not None

        return self._max_distance

    def set_distance(self) -> None:
        """Calculate and set the Euclidean distance transform."""
        assert isinstance(self.subdomain, (PaddedSubdomain, VerletSubdomain))
        _distance = distance.edt(img=self.img, subdomain=self.subdomain)
        _distance = communication.update_buffer(self.subdomain, _distance)
        self._distance = _distance


def gen_pm(
    subdomain: Subdomain | PaddedSubdomain | VerletSubdomain, img: NDArray[np.uint8]
) -> PorousMedia:
    """Initialize the PorousMedia class and set inlet/outlet/wall boundary conditions.

    Args:
        subdomain: Subdomain object.
        img (np.ndarray): Binary image of the porous medium.

    Returns:
        PorousMedia: Initialized porous media object.

    """
    pm = PorousMedia(subdomain=subdomain, img=img)
    return pm
