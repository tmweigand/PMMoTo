"""multiphase.py"""

import numpy as np
from ..core import communication
from ..core import utils


class Multiphase:
    """Multiphase Class"""

    def __init__(self, porous_media, img, num_phases: int):
        if porous_media.img is None:
            raise ValueError("Error: The porous_media image (img) is None.")
        self.porous_media = porous_media
        self.pm_img = self.porous_media.img
        self.subdomain = porous_media.subdomain
        self.num_phases = num_phases
        self.img = img
        self.fluids = list(range(1, num_phases + 1))

    def update_img(self, img):
        """Update the multiphase img
        """
        self.img = img

    def get_volume_fraction(self, phase: int, img=None) -> float:
        """Calculate the volume fraction of a given phase in a multiphase image.

        Parameters
        ----------
            phase (int): The phase ID to compute the volume fraction for.

        Returns
        -------
            float: The volume fraction of the specified phase.

        """
        if img is None:
            img = self.img

        local_img = utils.own_img(self.subdomain, img)
        local_voxel_count = np.count_nonzero(local_img == phase)

        total_voxel_count = (
            communication.all_reduce(local_voxel_count)
            if self.subdomain.domain.num_subdomains > 1
            else local_voxel_count
        )

        total_voxels = np.prod(self.subdomain.domain.voxels)
        return total_voxel_count / total_voxels

    def get_saturation(self, phase: int, img=None) -> float:
        """Calculate the saturation of a multiphase image
        """
        if img is None:
            img = self.img
        return self.get_volume_fraction(phase, img) / self.porous_media.porosity

    @staticmethod
    def get_probe_radius(pc, gamma=1, contact_angle=0):
        """Return the probe radius given a capillary pressure, surface tension and contact_angle
        """
        if pc == 0:
            r_probe = 0
        else:
            r_probe = 2.0 * gamma / pc * np.cos(np.deg2rad(contact_angle))
        return r_probe

    @staticmethod
    def get_pc(radius, gamma=1):
        """Return the capillary pressure given a surface tension and radius
        """
        return 2.0 * gamma / radius
