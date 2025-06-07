"""minkowski.py

Minkowski functionals analysis for PMMoTo.

This module provides routines to compute the four Minkowski functionals
(volume, surface area, mean curvature, Euler characteristic) for a given
binary image and a PMMoTo subdomain.
"""

from __future__ import annotations
from typing import Tuple, TYPE_CHECKING
import numpy as np
from numpy.typing import NDArray
from pmmoto.analysis import _minkowski
from pmmoto.core import utils

if TYPE_CHECKING:
    from pmmoto.core.subdomain import Subdomain

__all__ = ["functionals"]


def functionals(
    subdomain: Subdomain, img: NDArray[np.bool_]
) -> Tuple[float, float, float, float]:
    r"""Calculate the Minkowski functionals for a subdomain.

    The algorithm skips the last index for boundary conditions.

    Args:
        subdomain: Subdomain object.
        img (np.ndarray): Input binary image.

    Returns:
        tuple: (volume, surface_area, mean_curvature, euler_characteristic)
            - volume (float): Total volume of the pore space.
            - surface_area (float): Total surface area of the pore space.
            - mean_curvature (float): Integrated mean curvature of the pore space.
            - euler_characteristic (float): Euler characteristic of the pore space.

    Notes:
        The Minkowski functionals are defined as:

        .. math::

            M_{0} (X) = \int_{X} d v, \\
            M_{1} (X) = \frac{1}{8} \int_{\delta X} d s, \\
            M_{2} (X) = \frac{1}{2 \pi^{2}} \int_{\delta X}  \frac{1}{2} 
                \left[\frac{1}{R_{1}} + \frac{1}{R_{2}}\right] d s, \\
            M_{3} (X) = \frac{3}{(4 \pi)^{2}} \int_{\delta X} \left[\frac{1}{R_{1} 
                R_{2}}\right] d s,

        The returned values are computed for the "own" region of the subdomain,
        excluding padding and ghost regions, and are normalized by the voxel size.

    """
    if subdomain.domain.num_subdomains == 1:
        own_img = utils.own_img(subdomain, img)
        _functionals = _minkowski.functionals(
            own_img.astype(bool),
            subdomain.domain.voxels,
            subdomain.domain.resolution,
        )
    else:
        own_voxels = subdomain.get_own_voxels()
        for feature_id, feature in subdomain.features.faces.items():
            if feature.boundary_type == "internal" and np.sum(feature_id) > 0:
                own_voxels[feature.map_to_index()] += 1

        own_img = utils.own_img(subdomain, img, own_voxels)
        _functionals = _minkowski.functionals(
            own_img.astype(bool),
            subdomain.domain.voxels,
            subdomain.domain.resolution,
            parallel=True,
        )

    return _functionals
