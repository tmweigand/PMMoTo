"""Core utility functions for PMMoTo.

This module provides utility functions for array manipulation, validation,
MPI-aware operations, and subdomain/grid management.
"""

from __future__ import annotations
from typing import Any, TYPE_CHECKING, TypeVar, Callable, cast
import sys
import numpy as np
from numpy.typing import NDArray
from mpi4py import MPI
from .logging import get_logger, USE_LOGGING
from . import communication

if TYPE_CHECKING:
    from .subdomain import Subdomain

T = TypeVar("T", bound=np.generic)

comm = MPI.COMM_WORLD
if USE_LOGGING:
    logger = get_logger()


__all__ = [
    "phase_exists",
    "constant_pad_img",
    "unpad",
    "determine_maximum",
    "bin_image",
    "own_img",
]


def raise_error() -> None:
    """Exit gracefully by finalizing MPI and exiting the program."""
    MPI.Finalize()
    sys.exit()


def check_img_for_solid(subdomain: Subdomain, img: NDArray[T]) -> None:
    """Warn if a subdomain contains only pore voxels (no solid).

    Args:
        subdomain: Subdomain object with rank and voxels attributes.
        img (np.ndarray): Image array.

    """
    if np.sum(img) == np.prod(subdomain.voxels):
        logger.warning(
            "Many functions in pmmoto require one solid voxel in each subdomain. "
            "Process with rank: %i is all pores.",
            subdomain.rank,
        )


def unpad(img: NDArray[T], pad: tuple[tuple[int, int], ...]) -> NDArray[T]:
    """Remove padding from a NumPy array.

    Args:
        img (np.ndarray): The padded array.
        pad (list or tuple): Padding

    Returns:
        np.ndarray: The unpadded array.

    """
    slices = tuple(slice(p[0], img.shape[i] - p[1]) for i, p in enumerate(pad))
    return np.ascontiguousarray(img[slices])


def constant_pad_img(
    img: NDArray[T], pad: tuple[tuple[int, int], ...], pad_value: int | float
) -> NDArray[T]:
    """Pad a grid with a constant value.

    Args:
        img (np.ndarray): Input array.
        pad (list or tuple): Padding amounts for each dimension.
        pad_value (scalar): Value to use for padding.

    Returns:
        np.ndarray: The padded array.

    """
    img = np.pad(
        img,
        ((pad[0][0], pad[0][1]), (pad[1][0], pad[1][1]), (pad[2][0], pad[2][1])),
        "constant",
        constant_values=pad_value,
    )
    return img


def own_img(
    subdomain: Subdomain,
    img: NDArray[T],
    own_voxels: None | NDArray[np.integer[Any]] = None,
) -> NDArray[T]:
    """Return array with only nodes owned by the current process.

    Args:
        subdomain: Subdomain object with get_own_voxels method.
        img (np.ndarray): Input image.
        own_voxels: NDArray of size 2*dims with bounds of image to extract

    Returns:
        np.ndarray: Array of owned voxels.

    """
    if own_voxels is not None:
        own = own_voxels
    else:
        own = subdomain.get_own_voxels()

    img_out = img[own[0] : own[1], own[2] : own[3], own[4] : own[5]]

    return np.ascontiguousarray(img_out)


def phase_exists(img: NDArray[T], phase: int | float) -> bool:
    """Determine if a phase exists in the grid (globally).

    Args:
        img (np.ndarray): Input array.
        phase (int): Phase value to check.

    Returns:
        bool: True if phase exists, False otherwise.

    """
    phase_exists = False
    local_count = np.count_nonzero(img == phase)
    global_count = communication.all_reduce(local_count, op="sum")

    if global_count > 0:
        phase_exists = True

    return phase_exists


def determine_maximum(img: NDArray[T]) -> int | float:
    """Determine the global maximum of an input image.

    Args:
        img (np.ndarray): Input array.

    Returns:
        scalar: Global maximum value.

    """
    local_max = np.amax(img)

    proc_local_max = communication.all_gather(local_max)

    return cast(int | float, np.amax(np.asarray(proc_local_max)).item())


def bin_image(subdomain: Subdomain, img: NDArray[T], own: bool = True) -> dict[T, int]:
    """Count the number of times each unique element occurs in the input array.

    Args:
        subdomain: Subdomain object.
        img (np.ndarray): Input array.
        own (bool): If True, use only owned voxels.

    Returns:
        dict: Mapping from element value to count.

    """
    if own:
        _img = own_img(subdomain, img)
    else:
        _img = img

    local_counts = np.unique(_img, return_counts=True)

    global_counts = communication.all_gather(local_counts)
    image_counts: dict[T, int] = {}
    for proc_data in global_counts:
        for element, count in zip(proc_data[0], proc_data[1]):
            if element not in image_counts:
                image_counts[element] = 0
            image_counts[element] += count

    return image_counts


def decompose_img(
    img: NDArray[T],
    start: tuple[int, ...],
    shape: tuple[int, ...],
) -> NDArray[T]:
    """Decompose an image into a wrapped slice for a subdomain.

    Args:
        img (np.ndarray): Input array.
        start (tuple): Starting index for the slice.
        shape (tuple): Shape of the slice.
        padded_img (bool): Unused.

    Returns:
        np.ndarray: The resulting wrapped slice.

    """
    # Create indices with wrapping
    index: list[NDArray[np.intp]] = []
    for _start, _shape, _img_shape in zip(start, shape, img.shape):
        index.append(np.arange(_start, _start + _shape) % _img_shape)

    # Use advanced indexing to extract the slice
    return cast(NDArray[T], img[np.ix_(index[0], index[1], index[2])])


def check_subdomain_condition(
    subdomain: Subdomain,
    condition_fn: Callable[[Any, Any], np.bool_],
    args: Any,
    error_message: str,
    error_args: Any,
) -> bool:
    """Check a generic condition on the subdomain using provided arguments.

    If an error is detected on any rank, all ranks are terminated.

    Args:
        subdomain: Object with attributes `rank` and `own_voxels`.
        condition_fn: Callable(subdomain, *args) -> bool.
        args: Tuple of arguments to pass to condition_fn.
        error_message: str, a format string for the error message.
        error_args: Tuple of arguments to format into error_message.

    """
    local_error = condition_fn(subdomain, *args)
    if local_error:
        logger.error(error_message, *error_args if error_args else ())

    global_error = comm.allreduce(local_error, op=MPI.LOR)

    if global_error:
        comm.Barrier()
        if subdomain.rank == 0:
            logger.error("Terminating all processes due to distributed error condition")
        raise_error()

    return True
