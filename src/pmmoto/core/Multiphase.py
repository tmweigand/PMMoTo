"""multiphase.py"""

import numpy as np
from . import porousmedia

# from pmmoto.analysis import stats

__all__ = ["get_probe_radius", "get_pc", "initialize_multiphase"]


def get_probe_radius(pc, gamma):
    """
    Return the probe radius givena capillary pressure and surface tension
    """
    if pc == 0:
        r_probe = 0
    else:
        r_probe = 2.0 * gamma / pc
    return r_probe


def get_pc(radius, gamma):
    """
    Return the capillary pressure given a surface tension and radius
    """
    return 2.0 * gamma / radius


class Multiphase(porousmedia.PorousMedia):
    """Multiphase Class"""

    def __init__(self, pm_grid, loop_info, num_phases: int, **kwargs):
        super().__init__(**kwargs)
        self.num_phases = num_phases
        self.pm_grid = pm_grid
        self.loop_info = loop_info
        self.fluids = list(range(1, num_phases + 1))
        self.inlet = np.zeros(
            [self.num_phases + 1, self.subdomain.dims * 2], dtype=np.uint8
        )
        self.outlet = np.zeros(
            [self.num_phases + 1, self.subdomain.dims * 2], dtype=np.uint8
        )

    @classmethod
    def from_porous_media(cls, porous_media, num_phases):
        return cls(
            subdomain=porous_media.subdomain,
            grid=None,
            pm_grid=porous_media.grid,
            loop_info=porous_media.loop_info,
            num_phases=num_phases,
        )

    def set_inlet_outlet(self, inlets, outlets):
        """
        Determine Inlet/Outlet for each fluid phase
        """
        for fluid in self.fluids:
            for n in range(0, self.subdomain.dims):

                if self.subdomain.inlet[n * 2] and inlets[fluid - 1][n][0]:
                    self.inlet[fluid][n * 2] = True
                if self.subdomain.inlet[n * 2 + 1] and inlets[fluid - 1][n][1]:
                    self.inlet[fluid][n * 2 + 1] = True

                if self.subdomain.outlet[n * 2] and outlets[fluid - 1][n][0]:
                    self.outlet[fluid][n * 2] = True
                if self.subdomain.outlet[n * 2 + 1] and outlets[fluid - 1][n][1]:
                    self.outlet[fluid][n * 2 + 1] = True

    # def get_saturation(self, phase):
    #     """
    #     Calalcaute the saturation of a given phase
    #     """
    #     return stats.get_saturation(self.subdomain, self.grid, phase)


def initialize_multiphase(
    porous_media,
    num_phases,
    inlets,
    outlets,
):
    """
    Initialize the multiphase class and set inlet/outlet reservoirs
    """
    mp = Multiphase.from_porous_media(
        porous_media=porous_media,
        num_phases=num_phases,
    )
    mp.set_inlet_outlet(inlets, outlets)

    return mp
