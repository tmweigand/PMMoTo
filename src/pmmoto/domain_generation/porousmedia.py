"""porousmedia.py

Defines the PorousMedia class for representing and analyzing porous media,
including porosity and distance transform calculations.
"""

import numpy as np

from ..core import communication
from ..core import utils
from ..filters import distance


class PorousMedia:
    """Porous media class for storing image data and computing properties."""

    def __init__(self, subdomain, img):
        """Initialize a PorousMedia object.

        Args:
            subdomain: Subdomain object.
            img (np.ndarray): Binary image of the porous medium.

        """
        self.subdomain = subdomain
        self.img = img
        self._porosity = None
        self._distance = None

    @property
    def porosity(self):
        """Get the porosity of the porous medium.

        Returns:
            float: Porosity value (fraction of pore voxels).

        """
        if self._porosity is None:
            self.set_porosity()
        return self._porosity

    def set_porosity(self):
        """Calculate and set the porosity of the porous medium."""
        own_img = utils.own_img(self.subdomain, self.img)
        local_pore_voxels = np.count_nonzero(own_img == 1)  # One is pore space

        if self.subdomain.domain.num_subdomains > 1:
            global_pore_voxels = communication.all_reduce(local_pore_voxels)
        else:
            global_pore_voxels = local_pore_voxels

        self._porosity = global_pore_voxels / np.prod(self.subdomain.domain.voxels)

    @property
    def distance(self):
        """Get the Euclidean distance transform of the porous medium.

        Returns:
            np.ndarray: Distance transform array.

        """
        if self._distance is None:
            self.set_distance()
        return self._distance

    def set_distance(self):
        """Calculate and set the Euclidean distance transform."""
        self._distance = distance.edt(img=self.img, subdomain=self.subdomain)


def gen_pm(subdomain, img):
    """Initialize the PorousMedia class and set inlet/outlet/wall boundary conditions.

    Args:
        subdomain: Subdomain object.
        img (np.ndarray): Binary image of the porous medium.

    Returns:
        PorousMedia: Initialized porous media object.

    """
    pm = PorousMedia(subdomain=subdomain, img=img)
    return pm
