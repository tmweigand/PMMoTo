"""equilibrium_distribution.py"""

import numpy as np
import warnings
from .porosimetry import porosimetry
from . import connected_components


def drainage(multiphase, capillary_pressures):
    """
    This is a morphological approach to determining the equilibrium
    fluid distribution for a multiphase system.

    The updated img is stored in multiphase.img.
    """

    # Ensure capillary pressures are in a sorted list
    if isinstance(capillary_pressures, (int, float)):
        capillary_pressures = [capillary_pressures]
    elif isinstance(capillary_pressures, list):
        sorted_cp = sorted(capillary_pressures)
        if capillary_pressures != sorted_cp:
            warnings.warn(
                "The capillary pressure must be monotonically increasing. Sorting."
            )
            capillary_pressures = sorted_cp

    # Initialize saturation array
    w_saturation = np.zeros(len(capillary_pressures))

    # Perform drainage simulation
    for n, capillary_pressure in enumerate(capillary_pressures):

        # Compute morphological changes based on capillary pressure
        morph = porosimetry(
            subdomain=multiphase.subdomain,
            img=multiphase.pm_img,
            radius=multiphase.get_probe_radius(capillary_pressure),
            inlet=True,
            multiphase=multiphase,
            mode="hybrid",
        )

        # Identify wetting-phase connectivity
        w_connected = connected_components.outlet_connected_img(
            multiphase.subdomain, multiphase.img, 2
        )

        # Update the phase distribution

        multiphase.update_img(
            np.where((morph == 1) & (w_connected == 2), 1, multiphase.img)
        )

        # Store wetting phase saturation
        w_saturation[n] = multiphase.get_saturation(2)

    return w_saturation
