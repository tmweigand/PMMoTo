"""morphological_operators.py"""

import math
import numpy as np
from scipy.signal import fftconvolve
from ..core import communication
from ..core import utils
from . import distance


__all__ = [
    "gen_struct_ratio",
    "gen_struct_element",
    "addition",
    "dilate",
    "subtraction",
    "erode",
    "opening",
    "closing",
]


def gen_struct_ratio(resolution, radius):
    """Generate the structuring element dimensions for halo communication
    https://www.iwaenc.org/proceedings/1997/nsip97/pdf/scan/ns970226.pdf
    """
    if len(resolution) not in {2, 3}:
        raise ValueError("Resolution must be a list of length 2 or 3")

    if radius < np.min(resolution):
        raise ValueError(
            f"The chosen radius {radius} is too small! The resolutions is {resolution}"
        )

    struct_ratio = np.array(
        [math.ceil(radius / res) for res in resolution],
        dtype=np.int64,
    )
    return struct_ratio


def gen_struct_element(resolution, radius):
    """Generate the structuring element for FFT morphology approach.

    Parameters
    ----------
        resolution (list or tuple): Resolution in each dimension (2D: [x, y], 3D: [x, y, z]).
        radius (float): Radius of the structuring element.

    Returns
    -------
        tuple: (struct_ratio, struct_element), where:
            - struct_ratio: Array of structuring ratios.
            - struct_element: 2D or 3D array representing the structuring element.

    """
    if len(resolution) not in {2, 3}:
        raise ValueError("Resolution must be a list or tuple of length 2 or 3.")

    if resolution[0] != resolution[1] != resolution[2]:
        raise ValueError("Resolution must be isotropic for morphological methods.")

    struct_ratio = gen_struct_ratio(resolution, radius)

    grids = [np.linspace(-r, r, r * 2 + 1) for r, res in zip(struct_ratio, resolution)]
    _xg, _yg, _zg = np.meshgrid(*grids, indexing="ij")

    # Compute structuring element
    s = _xg**2 + _yg**2 + _zg**2
    _radius = radius / resolution[0]
    struct_element = np.array(s <= _radius * _radius, dtype=np.uint8)

    return struct_ratio, struct_element


def addition(subdomain, img, radius, fft=False):
    """Perform a morphological dilation on a binary domain
    """
    struct_ratio, struct_element = gen_struct_element(
        subdomain.domain.resolution, radius
    )

    halo_img, halo = communication.update_buffer(
        subdomain=subdomain,
        img=img,
        buffer=struct_ratio,
    )

    if fft:
        _grid = fftconvolve(halo_img, struct_element, mode="same") > 0.1
        _grid_out = _grid.astype(np.uint8)
    else:
        # Calls "local" distance transform since halo_img is padded with the maximum distance allowed
        _grid_distance = distance.edt3d(
            np.logical_not(halo_img),
            resolution=subdomain.domain.resolution,
            squared=True,
        )

        _grid_out = np.where((_grid_distance <= radius * radius), 1, 0).astype(np.uint8)

    grid_out = utils.unpad(_grid_out, halo)

    grid_out = subdomain.set_wall_bcs(grid_out)

    return grid_out


def dilate(subdomain, img, radius, fft=False):
    """Wrapper to morph_add
    """
    img_out = addition(subdomain, img, radius, fft)

    return img_out


def subtraction(subdomain, img, radius, fft=False):
    """Perform a morpological subtraction
    """
    struct_ratio, struct_element = gen_struct_element(
        subdomain.domain.resolution, radius
    )

    halo_img, halo = communication.update_buffer(
        subdomain=subdomain,
        img=img,
        buffer=struct_ratio,
    )

    if fft:
        ### Boundary condition fix for subtraction
        _pad = 1
        if "end" in subdomain.boundary_types.values():
            _pad = np.max(struct_ratio)
        _grid = np.pad(
            array=halo_img, pad_width=_pad, mode="constant", constant_values=1
        )
        _grid = fftconvolve(_grid, struct_element, mode="same") > (
            struct_element.sum() - 0.1
        )
        _grid_out = utils.unpad(_grid, _pad * np.ones_like(halo)).astype(np.uint8)
    else:
        _grid_distance = distance.edt3d(
            halo_img, resolution=subdomain.domain.resolution, squared=True
        )
        _grid_out = np.where((_grid_distance <= radius * radius), 0, 1).astype(np.uint8)

    grid_out = utils.unpad(_grid_out, halo)
    grid_out = subdomain.set_wall_bcs(grid_out)

    return grid_out


def erode(subdomain, grid, radius, fft=False):
    """Wrapper to morph_subtract
    """
    grid_out = subtraction(subdomain, grid, radius, fft)

    return grid_out


def opening(subdomain, grid, radius, fft=False):
    """Morphological opening
    """
    _erode = subtraction(subdomain, grid, radius, fft)
    open_map = addition(subdomain, _erode, radius, fft)
    return open_map


def closing(subdomain, grid, radius, fft=False):
    """Morphological opening
    """
    _dilate = addition(subdomain, grid, radius, fft)
    closing_map = subtraction(subdomain, _dilate, radius, fft)
    return closing_map


def check_radii(subdomain, radii):
    """Validates that each radius in the list does not exceed the subdomain size.

    Args:
        subdomain: Object with `own_voxels` and `rank` attributes.
        radii: Iterable of buffer radii to check against the subdomain.

    """
    error_message = (
        "The specified radius (%.2f) exceeds at least one dimension of the subdomain (%s).\n"
        "To resolve this, use a different subdomain topology—for example, change the configuration of subdomains from (%s).\n"
        "Simulation stopping.\n"
    )

    for radius in radii:
        struct_ratio, _ = gen_struct_element(subdomain.domain.resolution, radius)
        utils.check_subdomain_condition(
            subdomain=subdomain,
            condition_fn=lambda s, r: np.any(s.own_voxels < r),
            args=(struct_ratio,),
            error_message=error_message,
            error_args=(radius, subdomain.get_length(), subdomain.domain.subdomains),
        )
