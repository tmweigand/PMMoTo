"""Statistical analysis utilities for PMMoTo.

Provides functions to compute global minimum and maximum values across subdomains.
"""

from __future__ import annotations
import numpy as np
from numpy.typing import NDArray
from typing import Any, TYPE_CHECKING
from pmmoto.core import utils
from pmmoto.core import communication

if TYPE_CHECKING:
    from pmmoto.core.subdomain import Subdomain


__all__ = ["get_minimum", "get_maximum"]


def get_minimum(
    subdomain: Subdomain,
    img: NDArray[np.floating[Any] | np.integer[Any]],
    own: bool = True,
) -> Any:
    """Compute the global minimum value from a distributed image array.

    Depending on the `own` flag, the function either restricts the computation
    to the local portion of the image owned by the subdomain or uses the full image.
    If the domain is distributed across multiple subdomains, a parallel reduction is
    performed to obtain the global minimum.

    Args:
        subdomain (Subdomain): The subdomain defining ownership within the domain.
        img (NDArray): The full image array containing data across all subdomains.
        own (bool, optional): If True, compute the minimum over the owned portion
            of the image only; otherwise, use the full image. Defaults to True.

    Returns:
        Any: The global minimum value of the image, reduced across all processes
        if the domain is distributed.

    """
    if own:
        own_img = utils.own_img(subdomain, img)
        _min = np.min(own_img)
    else:
        _min = np.min(img)

    if subdomain.domain.num_subdomains > 1:
        _min = communication.all_reduce(_min, op="min")

    return _min


def get_maximum(
    subdomain: Subdomain,
    img: NDArray[np.floating[Any] | np.integer[Any]],
    own: bool = True,
) -> Any:
    """Compute the global maximum value from a distributed image array.

    Depending on the `own` flag, the function either restricts the computation
    to the local portion of the image owned by the subdomain or uses the full image.
    If the domain is distributed across multiple subdomains, a parallel reduction is
    performed to obtain the global maximum.

    Args:
        subdomain (Subdomain): The subdomain defining ownership within the domain.
        img (NDArray): The full image array containing data across all subdomains.
        own (bool, optional): If True, compute the maximum over the owned portion
            of the image only; otherwise, use the full image. Defaults to True.

    Returns:
        Any: The global maximum value of the image, reduced across all processes
        if the domain is distributed.

    """
    if own:
        own_img = utils.own_img(subdomain, img)
        _max = np.max(own_img)
    else:
        _max = np.max(img)

    if subdomain.domain.num_subdomains > 1:
        _max = communication.all_reduce(_max, op="max")

    return _max
