import numpy as np

from . import communication


class PorousMedia:
    """
    Porous media class
    """

    def __init__(self, subdomain, img):
        self.subdomain = subdomain
        self.img = img
        self.porosity = None
        self.set_wall_bcs()

        # Set solid phase inlet/outlet to zeros
        _inlet = np.zeros([2, 3 * 2], dtype=np.uint8)
        _inlet[1, :] = subdomain.inlet.reshape([1, 6])
        self.inlet = _inlet

        _outlet = np.zeros([2, 3 * 2], dtype=np.uint8)
        _outlet[1, :] = subdomain.outlet.reshape([1, 6])
        self.outlet = _outlet

    def set_wall_bcs(self):
        """
        If wall boundary conditions are specified, force solid on external boundaries.
        For wall boundary conditions, the walls are stores as neighbor.
        """
        feature_types = ["faces", "edges", "corners"]
        for feature_type in feature_types:
            for feature_id, feature in self.subdomain.features[feature_type].items():
                if feature.boundary_type == "wall":
                    self.img[
                        feature.loop["neighbor"][0][0] : feature.loop["neighbor"][0][1],
                        feature.loop["neighbor"][1][0] : feature.loop["neighbor"][1][1],
                        feature.loop["neighbor"][2][0] : feature.loop["neighbor"][2][1],
                    ] = 0

        self.img = communication.update_buffer(self.subdomain, self.img)

    # def get_porosity(self):
    #     """
    #     Calalcaute the porosity of porous media grid
    #     """
    #     self.porosity = 1.0 - stats.get_volume_fraction(self.subdomain, self.grid, 0)


def gen_pm(subdomain, img):
    """
    Initialize the porousmedia class and set inlet/outlet/wall bcs
    Gather loop_info for efficient looping
    """
    pm = PorousMedia(subdomain=subdomain, img=img)

    return pm
