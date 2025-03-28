"""porosimetry.py"""

from typing import Literal
import numpy as np
import matplotlib.pyplot as plt
from ..core import utils
from . import morphological_operators
from . import distance
from . import connected_components

__all__ = ["get_sizes", "porosimetry", "pore_size_distribution"]


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
    plot: Literal["cdf", "pdf"] = None,
):
    """
    Generates a img where values are equal to the radius of the largest sphere that can be centered at given voxel.
    Calls porosimetry function with single size and returns img_results.
    """
    if radii:
        if not isinstance(radii, list):
            radii = [radii]
    else:
        edt = porous_media.distance
        global_max_edt = utils.determine_maximum(edt)
        radii = get_sizes(
            np.min(subdomain.domain.resolution), global_max_edt, 50, "linear"
        )
        print(global_max_edt, radii)

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
            img_results[np.logical_and(img_results == 0, img_temp == 1)] = radius

    if plot:
        _plot_pore_size_distribution(subdomain, porous_media, radii, img_results, plot)

    return img_results


def _plot_pore_size_distribution(
    subdomain, porous_media, radii, img, plot_type: Literal["cdf", "pdf"] = "cdf"
):
    """
    Plots pore size distribution.

    plot_type: (string) choose between cumulative distribution function, or probability density function
    """

    # Will normalize by pore voxel count
    pore_voxel_count = porous_media.porosity * np.prod(subdomain.domain.voxels)
    print(pore_voxel_count)
    # Initialize pore size distributions
    pore_size_counts = np.zeros(len(radii))
    for i, radius in enumerate(radii):
        matches = np.where(img == radius)
        sum_matches = len(matches[0])
        print(i, radius, sum_matches)
        normalized_matches = sum_matches / pore_voxel_count
        pore_size_counts[i] = normalized_matches
    print(pore_size_counts)

    PLOT = True
    if PLOT:
        plt.figure(figsize=(8, 6))
        plt.title("Pore Size Distribution of Inkbottle Test Case")
        plt.xlabel("Pore Size Radius")
        if plot_type == "cdf":
            # Cumulatively sum pore_size_counts for pore_size_cdf
            pore_size_cdf = np.cumsum(pore_size_counts)
            reversed_radii = radii[::-1]
            plt.plot(
                reversed_radii,
                pore_size_cdf,
                linestyle="-",
                linewidth=2,
                color="darkorchid",
            )
            plt.ylabel("Cumulative Distribution Function")
            plt.savefig("../../pore_size_distribution_CDF_inkbottle.png")
        else:
            plt.plot(
                radii, pore_size_counts, linestyle="-", linewidth=2, color="royalblue"
            )
            plt.ylabel("Probability Density Function")
            plt.savefig("../../pore_size_distribution_PDF_inkbottle.png")
