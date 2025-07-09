"""equilibrium_distribution.py

Morphological approaches for equilibrium fluid distributions in multiphase systems.
Implements drainage simulation methods for various capillary pressure models.
"""

from __future__ import annotations
from typing import Literal, TYPE_CHECKING, Callable, Any
import numpy as np
from numpy.typing import NDArray
import logging

from .porosimetry import porosimetry
from . import connected_components
from ..io.output import save_img
from ..core.subdomain_padded import PaddedSubdomain
from ..core.subdomain_verlet import VerletSubdomain

if TYPE_CHECKING:
    from pmmoto.domain_generation.multiphase import Multiphase

logging.basicConfig(level=logging.INFO, format="%(message)s")


def drainage(
    multiphase: Multiphase[np.uint8],
    capillary_pressures: int | float | list[float] | NDArray[np.floating[Any]],
    gamma: float = 1,
    contact_angle: float = 0,
    method: Literal["standard", "contact_angle", "extended_contact_angle"] = "standard",
    save: bool = False,
) -> NDArray[np.float64]:
    """Simulate morphological drainage for a multiphase system.

    This function determines the equilibrium fluid distribution for a multiphase system
    using a morphological approach. The updated image is stored in multiphase.img.

    Args:
        multiphase: Multiphase object with .img, .subdomain, .porous_media, etc.
        capillary_pressures (list or float): List of capillary pressures
                                             Must be in increasing order.
        gamma (float, optional): Surface tension parameter.
        contact_angle (float, optional): Contact angle in degrees.
        method (str, optional): Drainage method.
        save (bool, optional): If True, save intermediate results.

    Returns:
        np.ndarray: Wetting phase saturation at each capillary pressure.

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
    approach: Callable[[Multiphase[np.uint8], float, float, float], NDArray[np.uint8]]
    if method == "standard":
        if contact_angle != 0:
            raise ValueError("The standard approach requires a zero contact angle!")
        approach = _standard_method
        update_img = True
    elif method == "contact_angle":
        if contact_angle == 0:
            logging.warning(
                "The contact angle is zero."
                "This will yield same results as the standard approach."
            )
        approach = _contact_angle_method
        update_img = True
    elif method == "extended_contact_angle":
        if contact_angle == 0:
            logging.warning(
                "The contact angle is zero."
                "This will yield same results as the standard approach."
            )
        approach = _extended_contact_angle_method
        update_img = True
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
        if update_img:
            multiphase.update_img(
                np.where((morph == 1) & (w_connected == 2), 1, multiphase.img)
            )
            mp_img = multiphase.img
        else:
            mp_img = np.where((morph == 1) & (w_connected == 2), 1, multiphase.img)

        if save:
            file_out = (
                f"drainage_results/capillary_pressure_{capillary_pressure:.3f}".replace(
                    ".", "_"
                )
            )
            save_img(
                file_out,
                multiphase.subdomain,
                mp_img,
                additional_img={"morph": morph},
            )

        # Store wetting phase saturation
        w_saturation[n] = multiphase.get_saturation(2, mp_img)

        if multiphase.subdomain.rank == 0:
            logging.info(
                "Wetting phase saturation at capillary pressure of %f: %f",
                capillary_pressure,
                w_saturation[n],
            )

    return w_saturation


def _standard_method(
    multiphase: Multiphase[np.uint8],
    capillary_pressure: float,
    gamma: float,
    contact_angle: float,
) -> NDArray[np.uint8]:
    """Drainage method following Hilpert and Miller 2001.

    The radius (r) is defined as:
        r = 2 * gamma / p_c
    where gamma is the surface tension and p_c is the capillary pressure.

    Args:
        multiphase: Multiphase object.
        capillary_pressure (float): Capillary pressure.
        gamma (float): Surface tension.
        contact_angle (float): Contact angle (should be zero for this method).

    Returns:
        np.ndarray: Morphological result.

    """
    # Compute morphological changes based on capillary pressure
    radius = multiphase.get_probe_radius(capillary_pressure, gamma)

    # Check if radius is larger than resolution:
    #       min(multiphase.subdomain.domain.resolution)
    # or if the radius is < maximum distance
    if (
        radius < min(multiphase.subdomain.domain.resolution)
        or radius > multiphase.porous_media.max_distance
    ):
        morph = np.zeros_like(multiphase.pm_img, dtype=np.uint8)
    else:
        assert isinstance(multiphase.subdomain, (PaddedSubdomain, VerletSubdomain))
        morph = porosimetry(
            subdomain=multiphase.subdomain,
            porous_media=multiphase.porous_media,
            radius=radius,
            inlet=True,
            multiphase=multiphase,
            mode="hybrid",
        )

    return morph


def _contact_angle_method(
    multiphase: Multiphase[np.uint8],
    capillary_pressure: float,
    gamma: float,
    contact_angle: float,
) -> NDArray[np.uint8]:
    """Drainage method following Schulz and Becker 2007.

    The radius (r) is defined as:
        r = 2 * gamma * cos(theta) / p_c
    where gamma is the surface tension, theta is the contact angle,
    and p_c is the capillary pressure.

    Args:
        multiphase: Multiphase object.
        capillary_pressure (float): Capillary pressure.
        gamma (float): Surface tension.
        contact_angle (float): Contact angle in degrees.

    Returns:
        np.ndarray: Morphological result.

    """
    # Compute morphological changes based on capillary pressure
    radius = multiphase.get_probe_radius(capillary_pressure, gamma, contact_angle)

    # Check if radius is larger than resolution:
    #       min(multiphase.subdomain.domain.resolution)
    # or if the radius is < maximum distance
    if (
        radius < min(multiphase.subdomain.domain.resolution)
        or radius > multiphase.porous_media.max_distance
    ):
        morph = np.zeros_like(multiphase.pm_img)
    else:
        assert isinstance(multiphase.subdomain, (PaddedSubdomain, VerletSubdomain))
        morph = porosimetry(
            subdomain=multiphase.subdomain,
            porous_media=multiphase.porous_media,
            radius=radius,
            inlet=True,
            multiphase=multiphase,
            mode="hybrid",
        )

    return morph


def _extended_contact_angle_method(
    multiphase: Multiphase[np.uint8],
    capillary_pressure: float,
    gamma: float,
    contact_angle: float,
) -> NDArray[np.uint8]:
    """Drainage method from Schulz and Wargo 2015.

    In this method, the radius of the erosion step includes the contact angle,
    but the dilation does not.

    Args:
        multiphase: Multiphase object.
        capillary_pressure (float): Capillary pressure.
        gamma (float): Surface tension.
        contact_angle (float): Contact angle in degrees.

    Returns:
        np.ndarray: Morphological result.

    Note:
        This method does not yield good results.

    """
    # Compute morphological changes based on capillary pressure
    erosion_radius = multiphase.get_probe_radius(
        capillary_pressure, gamma, contact_angle
    )

    # Mutiply again by cosine of contact angle
    dilation_radius = multiphase.get_probe_radius(
        capillary_pressure, gamma, contact_angle
    ) * np.abs(np.cos(np.deg2rad(contact_angle)))

    radius = [erosion_radius, dilation_radius]

    # Check if radius is larger than resolution:
    #       min(multiphase.subdomain.domain.resolution)
    # or if the radius is < maximum distance
    if (
        min(radius) < min(multiphase.subdomain.domain.resolution)
        or max(radius) > multiphase.porous_media.max_distance
    ):
        morph = np.zeros_like(multiphase.pm_img)
    else:
        assert isinstance(multiphase.subdomain, (PaddedSubdomain, VerletSubdomain))
        morph = porosimetry(
            subdomain=multiphase.subdomain,
            porous_media=multiphase.porous_media,
            radius=radius,
            inlet=True,
            multiphase=multiphase,
            mode="hybrid",
        )

    return morph
