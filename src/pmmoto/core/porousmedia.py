import numpy as np

from . import communication
from . import utils


class PorousMedia:
    """
    Porous media class
    """

    def __init__(self, subdomain, img):
        self.subdomain = subdomain
        self.img = img
        self._porosity = None

    @property
    def porosity(self):
        """
        Getter for porosity, calculates if None.
        """
        if self._porosity is None:
            self.set_porosity()
        return self._porosity

    def set_porosity(self):
        """
        Calculate the porosity of porous media
        """
        own_grid = utils.own_grid(self.img, self.subdomain.get_own_voxels())

        local_pore_voxels = np.count_nonzero(own_grid == 1)  # One is pore space

        if self.subdomain.domain.num_subdomains > 1:
            global_pore_voxels = communication.all_reduce(local_pore_voxels)
        else:
            global_pore_voxels = local_pore_voxels

        self._porosity = global_pore_voxels / np.prod(self.subdomain.domain.voxels)


def gen_pm(subdomain, img):
    """
    Initialize the porous media class and set inlet/outlet/wall bcs
    Gather loop_info for efficient looping
    """
    pm = PorousMedia(subdomain=subdomain, img=img)

    return pm
