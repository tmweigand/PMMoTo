"""porosimetry.py"""

from typing import Literal
import numpy as np
from ..core import utils
from . import morphological_operators
from . import distance
from . import connected_components

__all__ = ["get_sizes", "porosimetry"]


def get_sizes(min_value, max_value, num_values, spacing="linear"):
    """
    Give list of pore sizes based on inputs provided
    """
    if min_value >= max_value:
        raise ValueError(
            f"Error: min_value {min_value} must be greater than max value {max_value}"
        )

    if num_values <= 0:
        raise ValueError(f"Error: num_values {num_values} must be greater than 0")

    if spacing == "linear":
        values = np.linspace(min_value, max_value, num_values)[
            ::-1
        ]  # returns in non-increasing order

    elif spacing == "log":
        if min_value < 1:
            raise ValueError(
                f"Error: min_value {min_value} must be greater than or equal to 1 for log spacing"
            )

        # convert min/max to log10 exponents
        log_min = np.log10(min_value)
        log_max = np.log10(max_value)
        values = np.logspace(log_min, log_max, num_values)[::-1]

    else:
        raise ValueError(f"spacing {spacing} can only be 'linear' or 'log'")

    return values


def porosimetry(
    subdomain,
    img,
    radius,
    inlet=False,
    multiphase=None,
    mode: Literal["hybrid", "distance", "morph"] = "hybrid",
):
    """
    Perform a morphological erosion followed by a morphological dilation.
    If inlet, the foreground voxels must be connected to the inlet.
    """

    # Erosion
    if mode == "morph":
        img_results = morphological_operators.subtraction(
            subdomain=subdomain, img=img, radius=radius, fft=True
        )
    elif mode in {"distance", "hybrid"}:
        edt = distance.edt(img=img, subdomain=subdomain)
        img_results = edt >= radius

    # Check inlet
    if inlet:
        img_results = connected_components.inlet_connected_img(subdomain, img_results)

    # Dilation
    if utils.phase_exists(img_results, 1):

        # Handle multiphase constraints if provided
        if (
            multiphase
            and hasattr(multiphase, "img")
            and utils.phase_exists(multiphase.img, 1)
        ):
            n_connected = connected_components.inlet_connected_img(
                subdomain, multiphase.img, 1
            )

            img_results = np.where(
                (img_results != 1) | (n_connected != 1), 0, img_results
            )

        if mode == "morph":
            img_results = morphological_operators.addition(
                subdomain=subdomain, img=img_results, radius=radius, fft=True
            )

        elif mode == "distance":
            edt_inverse = distance.edt(
                img=np.logical_not(img_results), subdomain=subdomain
            )
            img_results = edt_inverse < radius

        elif mode == "hybrid":
            img_results = morphological_operators.addition(
                subdomain=subdomain, img=img_results, radius=radius, fft=False
            )

    return img_results.astype(np.double)


def pore_size_distribution(
    subdomain,
    img,
    radii,
    inlet=False,
    mode: Literal["hybrid", "dt", "morph"] = "hybrid",
):
    """
    Generates a img where values are equal to the radius of the largest sphere that can be centered at given voxel.
    Calls porosimetry function with single size and returns img_results.
    """
    if not isinstance(radii, list):
        radii = [radii]

    img_results = np.zeros_like(img, dtype=np.double)
    for radius in radii:
        img_temp = porosimetry(
            subdomain=subdomain, img=img, radius=radius, inlet=inlet, mode=mode
        )

        if np.any(img_temp):
            img_results[np.logical_and(img_results == 0, img_temp == 1)] = radius

    return img_results
