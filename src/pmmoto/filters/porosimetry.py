"""porosimetry.py"""

from typing import Literal
import numpy as np
import pmmoto

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
        # print(values)

    else:
        raise ValueError(f"spacing {spacing} can only be 'linear' or 'log'")

    return values


def get_radii(p_c, gamma):
    """
    Given list of capillary pressures return a list of radii.
    p_c = list of capillary pressures
    gamma = surface tension
    """
    if not isinstance(p_c, list):
        p_c = [p_c]

    radii = np.zeros_like(p_c, dtype=float)
    for i, p in enumerate(p_c):
        diam = (2 * gamma) / p
        r = diam / 2
        radii[i] = r

    return radii


def porosimetry(
    subdomain, img, radius, mode: Literal["hybrid", "dt", "morph"] = "hybrid"
):
    """
    sd = subdomain
    img = image
    radii = number
    mode = must be either "hybrid", "dt", or "morph"
    """

    img_results = np.zeros_like(img, dtype=np.double)

    if mode == "morph":
        img_results = pmmoto.filters.morphological_operators.subtraction(
            subdomain=subdomain, img=img, radius=radius, fft=True
        )

        img_results = pmmoto.filters.morphological_operators.addition(
            subdomain=subdomain, img=img_results, radius=radius, fft=True
        )

    elif mode == "dt":  # Close, but need to fix to match morph
        edt = pmmoto.filters.distance.edt(img=img, subdomain=subdomain)
        img_results = edt >= radius

        if np.any(img_results):
            edt_inverse = pmmoto.filters.distance.edt(
                img=~img_results, subdomain=subdomain
            )
            img_results = edt_inverse < radius

    elif mode == "hybrid":
        edt = pmmoto.filters.distance.edt(img=img, subdomain=subdomain)
        img_results = edt >= radius

        if np.any(img_results):
            img_results = pmmoto.filters.morphological_operators.addition(
                subdomain=subdomain, img=img_results, radius=radius, fft=False
            )
    else:
        raise Exception("Unrecognized mode" + mode)

    return img_results.astype(np.double)


def pore_size_distribution(
    subdomain, img, radii, mode: Literal["hybrid", "dt", "morph"] = "hybrid"
):
    """
    Calls porosimetry function with single size and returns img_results.
    img
    """
    if not isinstance(radii, list):
        radii = [radii]

    img_results = np.zeros_like(img, dtype=np.double)
    for radius in radii:
        img_temp = porosimetry(subdomain=subdomain, img=img, radius=radius, mode=mode)

        if np.any(img_temp):
            # what about multiphase systems
            img_results[np.logical_and(img_results == 0, img_temp == 1)] = radius

    return img_results
