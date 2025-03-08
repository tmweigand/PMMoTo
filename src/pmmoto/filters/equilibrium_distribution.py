"""equilibrium_distribution.py"""

import numpy as np
import logging
from typing import Literal

from .porosimetry import porosimetry
from . import connected_components


logging.basicConfig(level=logging.INFO, format="%(message)s")


def drainage(
    multiphase,
    capillary_pressures,
    gamma=1,
    contact_angle=0,
    method: Literal["standard", "contact_angle"] = "standard",
):
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
            logging.warning(
                "The capillary pressure must be monotonically increasing. Sorting."
            )
            capillary_pressures = sorted_cp

    # Method Checks
    if method == "standard":
        if contact_angle != 0:
            raise ValueError("The standard approach requires a zero contact angle!")
        approach = _standard_method
    elif method == "contact_angle":
        if contact_angle == 0:
            logging.warning(
                "The contact angle is zero. This will yield same results as the standard approach."
            )
        approach = _contact_angle_method
    else:
        raise ValueError(f"{method} is not implemented. ")

    # Initialize saturation array
    w_saturation = np.zeros(len(capillary_pressures))

    # Perform drainage simulation
    for n, capillary_pressure in enumerate(capillary_pressures):

        # Apply the specified approach for the erosion and dilation step
        morph = approach(multiphase, capillary_pressure, gamma, contact_angle)

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

        if multiphase.subdomain.rank == 0:
            logging.info(
                "Wetting phase saturation at capillary pressure of %f: %f",
                capillary_pressure,
                w_saturation[n],
            )

    return w_saturation


def _standard_method(multiphase, capillary_pressure, gamma, contact_angle):
    """
    This method for drainage follows Hilpert and Miller 2001
    """

    # Compute morphological changes based on capillary pressure
    radius = multiphase.get_probe_radius(capillary_pressure, gamma)

    # Check if radius is larger than resolution
    if radius < min(multiphase.subdomain.domain.resolution):
        morph = np.zeros_like(multiphase.pm_img)
    else:
        morph = porosimetry(
            subdomain=multiphase.subdomain,
            porous_media=multiphase.porous_media,
            radius=radius,
            inlet=True,
            multiphase=multiphase,
            mode="hybrid",
        )

    return morph


def _contact_angle_method(multiphase, capillary_pressure, gamma, contact_angle):
    """
    This method for drainage follows Schulz and Wargo 2015.
    This paper seems to have many typos/ errors?
    ..For example Eqations 1 and 5 show dimeter but should be radii.
    """

    # Compute morphological changes based on capillary pressure
    erosion_radius = multiphase.get_probe_radius(
        capillary_pressure, gamma, contact_angle
    )
    dilation_radius = multiphase.get_probe_radius(capillary_pressure, gamma)

    radius = [erosion_radius, dilation_radius]

    # Check if radius is larger than resolution
    if min(radius) < min(multiphase.subdomain.domain.resolution):
        morph = np.zeros_like(multiphase.pm_img)
    else:
        morph = porosimetry(
            subdomain=multiphase.subdomain,
            porous_media=multiphase.porous_media,
            radius=radius,
            inlet=True,
            multiphase=multiphase,
            mode="hybrid",
        )

    return morph
