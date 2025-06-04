"""average.py

Functions for calculating averages of 3D images.
"""

from __future__ import annotations
from typing import Literal, Tuple, TYPE_CHECKING, TypeVar, Union
import numpy as np
from numpy.typing import NDArray
from ..core import utils
from ..core import communication

if TYPE_CHECKING:
    from pmmoto.core.subdomain import Subdomain

T = TypeVar("T", bound=np.generic)

__all__ = ["average_image_along_axis"]


def average_image_along_axis(
    subdomain: Subdomain,
    img: NDArray[T],
    dimension: Union[Literal[0, 1, 2], Tuple[Literal[0, 1, 2], Literal[0, 1, 2]]],
) -> NDArray[np.float64]:
    """Calculate the average of a 3D image along one or two dimensions.

    Args:
        subdomain: The subdomain object containing domain information
        img: 3D numpy array representing the image
        dimension: One or two dimensions along which to average (0=x, 1=y, 2=z)

    Returns:
        np.ndarray: Reduced array (2D or 1D) of averaged values.

    """
    if isinstance(dimension, int):
        dims: Tuple[int, ...] = (dimension,)
    else:
        dims = dimension
        if len(dims) != 2 or dims[0] == dims[1]:
            raise ValueError("Must provide two *distinct* dimensions for 2D averaging")
        if not all(0 <= d <= 2 for d in dims):
            raise ValueError("Dimension indices must be in (0, 1, 2)")

    own_img = utils.own_img(subdomain, img)
    _sum = np.asarray(own_img.sum(axis=dims), dtype=np.float64)

    # Calculate number of voxels over which average is taken
    if len(dims) == 1:
        voxels = int(subdomain.domain.voxels[dims[0]])
    else:
        voxels = int(
            subdomain.domain.voxels[dims[0]] * subdomain.domain.voxels[dims[1]]
        )

    if subdomain.domain.num_subdomains == 1:
        return _sum / voxels

    # Parallel case
    other_dims = [d for d in range(3) if d not in dims]
    global_shape = (
        subdomain.domain.voxels[other_dims[0]]
        if len(other_dims) == 1
        else (
            subdomain.domain.voxels[other_dims[0]],
            subdomain.domain.voxels[other_dims[1]],
        )
    )
    global_sum = np.zeros(global_shape)
    all_sums = communication.all_gather(_sum)

    for proc, data in enumerate(all_sums):
        proc_index = subdomain.get_index(proc, subdomain.domain.subdomains)
        start = subdomain.get_start(
            proc_index, subdomain.domain.voxels, subdomain.domain.subdomains
        )
        proc_voxels = subdomain.get_voxels(
            index=proc_index,
            domain_voxels=subdomain.domain.voxels,
            subdomains=subdomain.domain.subdomains,
        )

        if len(other_dims) == 1:
            dim = other_dims[0]
            start_dim = start[dim] * proc_index[dim]
            end_dim = start_dim + proc_voxels[dim]
            global_sum[start_dim:end_dim] += data
        else:
            slices = []
            for dim in other_dims:
                start_dim = start[dim] * proc_index[dim]
                end_dim = start_dim + proc_voxels[dim]
                slices.append(slice(start_dim, end_dim))
            global_sum[slices[0], slices[1]] += data

    return global_sum / voxels
