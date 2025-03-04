"""equilibrium_distribution.py"""

import numpy as np
from .porosimetry import porosimetry


def drainage(capillary_pressure, multiphase):
    """
    This is a morphological approach to determining the equilibrium
    fluid distribution for a multiphase system
    Args:
        capillary_pressure (_type_): _description_
        multiphase (_type_): _description_

    Returns:
        _type_: _description_
    """
    radius = multiphase.get_probe_radius(capillary_pressure)

    img_temp = porosimetry(
        subdomain=multiphase.subdomain,
        img=multiphase.porous_media.img,
        radius=radius,
        inlet=True,
        mode="morph",
    )

    img_out = np.where(
        (multiphase.porous_media.img == 1) & (img_temp == 0), 2, img_temp
    )

    multiphase.update_img(img_out)
    w_saturation = multiphase.get_saturation(2)

    return img_out, w_saturation
