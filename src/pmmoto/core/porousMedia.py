import numpy as np
from pmmoto.core import orientation
from pmmoto.analysis import stats


class PorousMedia:
    """
    Porous media class
    """

    def __init__(self, subdomain, grid):
        self.subdomain = subdomain
        self.grid = grid
        self.loop_info = np.zeros([orientation.num_faces + 1, 3, 2], dtype=np.int64)
        self.porosity = None

        # Set solid phase inlet/outlet to zeros
        _inlet = np.zeros([2, subdomain.dims * 2], dtype=np.uint8)
        _inlet[1, :] = subdomain.inlet.reshape([1, 6])
        self.inlet = _inlet

        _outlet = np.zeros([2, subdomain.dims * 2], dtype=np.uint8)
        _outlet[1, :] = subdomain.outlet.reshape([1, 6])
        self.outlet = _outlet

    def set_wall_bcs(self):
        """
        If wall boundary conditions are specified, force solid on external boundaries
        """
        if self.subdomain.boundaries[0] == 1:
            self.grid[0, :, :] = 0
        if self.subdomain.boundaries[1] == 1:
            self.grid[-1, :, :] = 0
        if self.subdomain.boundaries[2] == 1:
            self.grid[:, 0, :] = 0
        if self.subdomain.boundaries[3] == 1:
            self.grid[:, -1, :] = 0
        if self.subdomain.boundaries[4] == 1:
            self.grid[:, :, 0] = 0
        if self.subdomain.boundaries[5] == 1:
            self.grid[:, :, -1] = 0

    def get_porosity(self):
        """
        Calalcaute the porosity of porous media grid
        """
        self.porosity = 1.0 - stats.get_volume_fraction(self.subdomain, self.grid, 0)


def gen_pm(subdomain, grid, res_size=0):
    """
    Initialize the porousmedia class and set inlet/outlet/wall bcs
    Gather loop_info for efficient looping
    """
    pm = PorousMedia(subdomain=subdomain, grid=grid)
    pm.set_wall_bcs()
    pm.loop_info = orientation.get_loop_info(
        pm.grid, subdomain, subdomain.inlet, subdomain.outlet, res_size
    )

    return pm
