"""porosimetry.py"""

from typing import Literal
import numpy as np
import pmmoto

__all__ = ["get_sizes", "porosimetry", "extract_subsection"]


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


def extract_subsection(img, shape):
    """
    Extract a defined subsection
    Parameters
    ----------
    im : ndarray
        Image from which to extract the subsection
    shape : array_like
        Can either specify the size of the extracted section or the fractional
        size of the image to extact.

    Returns
    -------
    image : ndarray
        An ndarray of size given by the ``shape`` argument, taken from the
        center of the image.

    """
    shape = np.array(shape)
    if shape[0] < 1:
        shape = np.array(img.shape) * shape
    center = np.array(img.shape) / 2
    s_img = []
    for dim in range(img.ndim):
        r = shape[dim] / 2
        lower_img = np.amax((center[dim] - r, 0))
        upper_img = np.amin((center[dim] + r, img.shape[dim]))
        s_img.append(slice(int(lower_img), int(upper_img)))
    return img[tuple(s_img)]


def porosimetry(
    subdomain, img, sizes, mode: Literal["hybrid", "dt", "morph"] = "hybrid"
):
    """
    sd = subdomain
    img = image
    sizes = list of pore sizes from get_sizes() function
    mode = must be either "hybrid", "dt", or "morph"
    """
    if not isinstance(sizes, list):
        sizes = [sizes]

    img_results = np.zeros_like(img, dtype=np.double)

    if mode == "morph":
        for radius in sizes:
            img_temp = pmmoto.filters.morphological_operators.subtraction(
                subdomain=subdomain, img=img, radius=radius, fft=True
            )

            img_temp = pmmoto.filters.morphological_operators.addition(
                subdomain=subdomain, img=img_temp, radius=radius, fft=True
            )

            if np.any(img_temp):
                # what about multiphase systems
                img_results[np.logical_and(img_results == 0, img_temp == 1)] = radius

    elif mode == "dt":  # Close, but need to fix to match morph
        edt = pmmoto.filters.distance.edt(img=img, subdomain=subdomain)
        for radius in sizes:
            img_temp = edt >= radius

            if np.any(img_temp):
                edt_inverse = pmmoto.filters.distance.edt(
                    img=~img_temp, subdomain=subdomain
                )
                img_temp = edt_inverse < radius
                img_results[np.logical_and(img_results == 0, img_temp == 1)] = radius

    elif mode == "hybrid":
        edt = pmmoto.filters.distance.edt(img=img, subdomain=subdomain)
        for radius in sizes:
            img_temp = edt >= radius

            if np.any(img_temp):
                img_temp = pmmoto.filters.morphological_operators.addition(
                    subdomain=subdomain, img=img_temp, radius=radius, fft=False
                )
                img_results[np.logical_and(img_results == 0, img_temp == 1)] = radius
    else:
        raise Exception("Unrecognized mode" + mode)

    return img_results
