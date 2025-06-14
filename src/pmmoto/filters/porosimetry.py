"""porosimetry.py"""

from typing import Literal, Dict
import numpy as np
import matplotlib.pyplot as plt
from ..core import utils
from ..io import io_utils
from . import morphological_operators
from . import distance
from . import connected_components

__all__ = [
    "get_sizes",
    "porosimetry",
    "pore_size_distribution",
    "plot_pore_size_distribution",
]


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
    porous_media,
    radius,
    inlet=False,
    multiphase=None,
    mode: Literal["hybrid", "distance", "morph"] = "hybrid",
):
    """
    Perform a morphological erosion followed by a morphological dilation.
    If inlet, the foreground voxels must be connected to the inlet.

    Additionally, allow for different radii to specified for the erosion and dilation.
    To do this, provide a list where the first entry is the erosion radius and the
    second entry is the dilation radius.
    """

    if isinstance(radius, (int, float)):
        erosion_radius = radius
        dilation_radius = radius
    elif isinstance(radius, list):
        erosion_radius = radius[0]
        dilation_radius = radius[1]
    else:
        raise ValueError(
            f"Radius {radius} must either be a number or a list of length 2"
        )

    morphological_operators.check_radii(subdomain, [erosion_radius, dilation_radius])

    # Erosion
    if mode == "morph":
        img_results = morphological_operators.subtraction(
            subdomain=subdomain, img=porous_media.img, radius=erosion_radius, fft=True
        )
    elif mode in {"distance", "hybrid"}:
        edt = porous_media.distance
        img_results = edt >= erosion_radius

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
                subdomain=subdomain, img=img_results, radius=dilation_radius, fft=True
            )

        elif mode == "distance":
            edt_inverse = distance.edt(
                img=np.logical_not(img_results), subdomain=subdomain
            )
            img_results = edt_inverse < dilation_radius

        elif mode == "hybrid":
            img_results = morphological_operators.addition(
                subdomain=subdomain, img=img_results, radius=dilation_radius, fft=False
            )

    return img_results.astype(np.double)


def pore_size_distribution(
    subdomain,
    porous_media,
    radii=None,
    inlet=False,
    mode: Literal["hybrid", "distance", "morph"] = "hybrid",
):
    """
    Generates a img where values are equal to the radius of the largest sphere that can be centered at given voxel.
    Calls porosimetry function with single size and returns img_results.
    """
    if radii is not None:
        if isinstance(radii, (int, float)):
            radii = [radii]
        elif isinstance(radii, list):
            radii.sort(reverse=True)
        elif isinstance(radii, np.ndarray):
            radii = np.sort(radii)[::-1]
    else:
        edt = porous_media.distance
        global_max_edt = utils.determine_maximum(edt)
        radii = get_sizes(
            np.min(subdomain.domain.resolution), global_max_edt, 50, "linear"
        )

    morphological_operators.check_radii(subdomain, radii)

    img_results = np.zeros_like(porous_media.img, dtype=np.double)
    for radius in radii:
        img_temp = porosimetry(
            subdomain=subdomain,
            porous_media=porous_media,
            radius=radius,
            inlet=inlet,
            mode=mode,
        )

        if np.any(img_temp):
            img_results = np.where(
                (img_results == 0) & (img_temp == 1), radius, img_results
            )

    return img_results


def plot_pore_size_distribution(
    file_name: str,
    pore_size_counts: Dict[float, float],
    plot_type: Literal["cdf", "pdf"] = "pdf",
):
    """
    Plots pore size distribution.
    plot_type: (string) choose between cumulative distribution function, or probability density function
    """
    io_utils.check_file_path(file_name)
    out_file = file_name + "pore_size_distribution.png"

    radii = np.array(list(pore_size_counts.keys()))
    counts = np.array(list(pore_size_counts.values()))
    total_counts = np.sum(counts)

    plt.figure(figsize=(8, 6))
    plt.title("Pore Size Distribution")
    plt.xlabel("Pore Size Radius")

    if plot_type == "cdf":
        # Cumulatively sum pore_size_counts for pore_size_cdf
        pore_size_cdf = np.cumsum(counts / total_counts)
        reversed_radii = radii[::-1]
        plt.plot(
            reversed_radii,
            pore_size_cdf,
            linestyle="-",
            linewidth=2,
            color="darkorchid",
        )
        plt.ylabel("Cumulative Distribution Function")
        plt.savefig(out_file)
    else:
        plt.plot(
            radii, counts / total_counts, linestyle="-", linewidth=2, color="royalblue"
        )
        plt.ylabel("Probability Density Function")
        plt.savefig(out_file)
