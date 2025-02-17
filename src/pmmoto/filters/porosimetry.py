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


def porosimetry(sd, img, sizes, mode: Literal["hybrid", "dt", "mio"] = "hybrid"):
    """

    sd = subdomain
    img = image
    sizes = list of pore sizes from get_sizes() function
    mode = must be either "hybrid", "dt", or "mio"
    """

    if mode == "mio":
        imgresults = np.zeros(np.shape(img))
        print(imgresults)
        for radius in sizes:
            imgtemp = pmmoto.filters.morphological_operators.subtraction(
                subdomain=sd, img=img, radius=radius, fft=False
            )

            imgtemp = pmmoto.filters.morphological_operators.addition(
                subdomain=sd, img=img, radius=radius, fft=False
            )

        if np.any(imgtemp):
            imgresults[(imgresults == 0) * imgtemp] = radius

        imgresults = extract_subsection(imgresults, shape=img.shape)

    elif mode == "dt":
        imgresults = np.zeros(np.shape(img))
        for radius in sizes:
            dt = pmmoto.filters.distance.edt(img=img, subdomain=sd)
            imgtemp = dt >= radius

            if np.any(imgtemp):
                imgtemp = pmmoto.filters.distance.edt(~imgtemp) < radius
                imgresults[(imgresults == 0) * imgtemp] = radius

    elif mode == "hybrid":
        imgresults = np.zeros(np.shape(img))
        for radius in sizes:
            dt = pmmoto.filters.distance.edt(img=img, subdomain=sd)
            imgtemp = dt >= radius

            if np.any(imgtemp):
                imgtemp = pmmoto.filters.morphological_operators.addition(
                    subdomain=sd, img=img, radius=radius, fft=False
                )
        imgresults[(imgresults == 0) * imgtemp] = radius
    else:
        raise Exception("Unrecognized mode" + mode)

    return imgresults
