"""porosimetry.py

Functions for pore size analysis and morphological porosimetry in PMMoTo.
"""

from __future__ import annotations
from typing import Literal, Any, TYPE_CHECKING
import numpy as np
from numpy.typing import NDArray
import matplotlib.pyplot as plt
from ..core import utils
from ..io import io_utils
from . import morphological_operators
from . import distance
from . import connected_components

if TYPE_CHECKING:
    from ..core.subdomain_padded import PaddedSubdomain
    from ..core.subdomain_verlet import VerletSubdomain
    from ..domain_generation.porousmedia import PorousMedia
    from ..domain_generation.multiphase import Multiphase

__all__ = [
    "get_sizes",
    "porosimetry",
    "pore_size_distribution",
    "plot_pore_size_distribution",
]


def get_sizes(
    min_value: float,
    max_value: float,
    num_values: int,
    spacing: Literal["linear", "log"] = "linear",
) -> NDArray[np.floating[Any]]:
    """Generate a list of pore sizes based on input parameters.

    Args:
        min_value (float): Minimum pore size.
        max_value (float): Maximum pore size.
        num_values (int): Number of values to generate.
        spacing (str, optional): "linear" or "log" spacing. Defaults to "linear".

    Returns:
        np.ndarray: Array of pore sizes in non-increasing order.

    Raises:
        ValueError: If input parameters are invalid.

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
                f"Error: min_value {min_value} must be greater than or equal to 1"
            )

        # convert min/max to log10 exponents
        log_min = np.log10(min_value)
        log_max = np.log10(max_value)
        values = np.logspace(log_min, log_max, num_values)[::-1]

    else:
        raise ValueError(f"spacing {spacing} can only be 'linear' or 'log'")

    return values


def porosimetry(
    subdomain: PaddedSubdomain | VerletSubdomain,
    porous_media: PorousMedia,
    radius: float | list[float],
    inlet: bool = False,
    multiphase: None | Multiphase[np.uint8] = None,
    mode: Literal["hybrid", "distance", "morph"] = "hybrid",
) -> NDArray[np.uint8]:
    """Perform morphological porosimetry (erosion/dilation) on a porous medium.

    Args:
        subdomain: Subdomain object.
        porous_media: Porous media object with .img and .distance attributes.
        radius (float or list): Erosion/dilation radius or [erosion, dilation].
        inlet (bool, optional): If True, require connectivity to inlet.
        multiphase (optional): Optional multiphase constraint.
        mode (str, optional): "hybrid", "distance", or "morph".

    Returns:
        np.ndarray: Resulting binary image after porosimetry.

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
        edt: NDArray[np.float32] = porous_media.distance
        img_results = (edt >= erosion_radius).astype(np.uint8)

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
            img_results = (edt_inverse < dilation_radius).astype(np.uint8)

        elif mode == "hybrid":
            img_results = morphological_operators.addition(
                subdomain=subdomain, img=img_results, radius=dilation_radius, fft=False
            )

    return img_results.astype(np.uint8)


def pore_size_distribution(
    subdomain: PaddedSubdomain | VerletSubdomain,
    porous_media: PorousMedia,
    radii: None | list[float] | NDArray[np.floating[Any]] = None,
    inlet: bool = False,
    mode: Literal["hybrid", "distance", "morph"] = "hybrid",
) -> NDArray[np.double]:
    """Generate image where values are the radius of the largest sphere centered there.

    Args:
        subdomain: Subdomain object.
        porous_media: Porous media object with .img and .distance attributes.
        radii (list or np.ndarray, optional): List of radii to use.
                                              If None, computed from the distance.
        inlet (bool, optional): If True, require connectivity to inlet.
        mode (str, optional): "hybrid", "distance", or "morph".

    Returns:
        np.ndarray: Image with pore size values.

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
    pore_size_counts: dict[float, float],
    plot_type: Literal["cdf", "pdf"] = "pdf",
) -> None:
    """Plot and save the pore size distribution as a PNG.

    Args:
        file_name (str): Output file base name.
        pore_size_counts (dict): Mapping from pore size radius to count.
        plot_type (str, optional): cdf for cumulative or pdf for probability density.

    Returns:
        None

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
