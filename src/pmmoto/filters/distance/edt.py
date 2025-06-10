"""edt.py

Exact Euclidean Distance Transform (EDT) functions for 2D and 3D images,
with support for periodic boundaries and distributed subdomains.
"""

from __future__ import annotations
from typing import TypeVar, TYPE_CHECKING, Literal
import numpy as np
from numpy.typing import NDArray
from pmmoto.core import _voxels
from pmmoto.core import communication
from . import _distance

if TYPE_CHECKING:
    from pmmoto.core.subdomain_padded import PaddedSubdomain
    from pmmoto.core.subdomain_verlet import VerletSubdomain
    from ._distance import Hull

T = TypeVar("T", bound=np.generic)

__all__ = ["edt", "edt2d", "edt3d"]


def edt(
    img: NDArray[np.uint8],
    subdomain: None | PaddedSubdomain | VerletSubdomain = None,
) -> NDArray[np.float32]:
    """Calculate the exact Euclidean transform of an image.

    Args:
        img (np.ndarray): Input binary image.
        subdomain (optional): Subdomain object for distributed/periodic support.

    Returns:
        np.ndarray: Euclidean distance transform of the image.

    """
    if subdomain is not None:
        if subdomain.domain.periodic or subdomain.domain.num_subdomains > 1:
            img_out = corrected_edt(img, subdomain)
        else:
            img_out = edt3d(img, resolution=subdomain.domain.resolution)

    else:  # Simply perform the edt with no corrections
        if len(img.shape) == 3:
            img_out = edt3d(img)
        elif len(img.shape) == 2:
            img_out = edt2d(img)
        else:
            raise ValueError(f"Wrong img dimension {img.shape}")

    return img_out


def corrected_edt(
    img: NDArray[np.uint8], subdomain: PaddedSubdomain | VerletSubdomain
) -> NDArray[np.float32]:
    """Perform an EDT with correctors for periodic boundaries and distributed memory.

    Args:
        img (np.ndarray): Input binary image.
        subdomain: Subdomain object.

    Returns:
        np.ndarray: Corrected Euclidean distance transform.

    """
    img_out = np.copy(img).astype(np.float32)

    dimension = 0
    lower_correctors, upper_correctors = get_initial_correctors(
        subdomain=subdomain, img=img, dimension=dimension
    )

    img_out = _distance.get_initial_envelope(
        img,
        img_out,
        dimension=dimension,
        resolution=subdomain.domain.resolution[dimension],
        lower_boundary=lower_correctors,
        upper_boundary=upper_correctors,
    )

    for dimension in [1, 2]:
        lower_hull, upper_hull = get_boundary_hull(
            subdomain=subdomain,
            img=img_out,
            og_img=img,
            dimension=dimension,
        )

        _distance.get_parabolic_envelope(
            img_out,
            dimension=dimension,
            resolution=subdomain.domain.resolution[dimension],
            lower_hull=lower_hull,
            upper_hull=upper_hull,
        )

    return np.asarray(np.sqrt(img_out))


def edt2d(
    img: NDArray[np.uint8],
    periodic: tuple[bool, bool] = (False, False),
    resolution: tuple[float, float] = (1.0, 1.0),
) -> NDArray[np.float32]:
    """Perform an exact Euclidean transform on a 2D image.

    Args:
        img (np.ndarray): Input binary image.
        periodic (list[bool], optional): Periodicity for each dimension.
        resolution (tuple[float], optional): Voxel spacing for each dimension.

    Returns:
        np.ndarray: Euclidean distance transform of the image.

    """
    img_out = np.copy(img).astype(np.float32)

    dimension = 1
    _lower_correctors = None
    _upper_correctors = None

    if periodic[dimension]:
        lower_correctors, upper_correctors = (
            _distance.get_initial_envelope_correctors_2d(img, dimension=dimension)
        )

        # correct indexes and swap
        _upper_correctors = np.where(
            lower_correctors != -1, lower_correctors + 1, np.inf
        )

        _lower_correctors = np.where(
            upper_correctors != -1,
            img.shape[dimension] - upper_correctors,
            np.inf,
        )

    img_out = _distance.get_initial_envelope_2d(
        img,
        img_out,
        dimension=dimension,
        resolution=resolution[dimension],
        lower_boundary=_lower_correctors,
        upper_boundary=_upper_correctors,
    )

    dimension = 0
    lower_hull = None
    upper_hull = None
    if periodic[dimension]:

        lower_vertex = _voxels.get_nearest_boundary_index_face_2d(
            img=img,
            dimension=dimension,
            forward=True,
            label=0,
            lower_skip=0,
            upper_skip=0,
        )

        upper_vertex = _voxels.get_nearest_boundary_index_face_2d(
            img=img,
            dimension=dimension,
            forward=False,
            label=0,
            lower_skip=0,
            upper_skip=0,
        )

        num_hull = 4
        _lower = _distance.get_boundary_hull_2d(
            img=img_out,
            bound=lower_vertex,
            dimension=dimension,
            resolution=resolution[dimension],
            num_hull=num_hull,
            forward=True,
        )

        _upper = _distance.get_boundary_hull_2d(
            img=img_out,
            bound=upper_vertex,
            dimension=dimension,
            resolution=resolution[dimension],
            num_hull=num_hull,
            forward=False,
        )

        # swap hulls
        lower_hull = _upper
        upper_hull = _lower

    _distance.get_parabolic_envelope_2d(
        img_out,
        dimension=dimension,
        lower_hull=lower_hull,
        upper_hull=upper_hull,
        resolution=resolution[dimension],
    )

    return np.asarray(np.sqrt(img_out))


def edt3d(
    img: NDArray[np.uint8],
    periodic: tuple[bool, ...] = (False, False, False),
    resolution: tuple[float, ...] = (1.0, 1.0, 1.0),
    squared: bool = False,
) -> NDArray[np.float32]:
    """Perform an exact Euclidean transform on a 3D image.

    Args:
        img (np.ndarray): Input binary image.
        periodic (list[bool], optional): Periodicity for each dimension.
        resolution (tuple[float], optional): Voxel spacing for each dimension.
        squared (bool, optional): If True, return squared distances.

    Returns:
        np.ndarray: Euclidean distance transform of the image.

    """
    img_out = np.copy(img).astype(np.float32)

    dimension = 0
    lower_correctors = None
    upper_correctors = None

    if periodic[dimension]:
        lower_correctors, upper_correctors = _distance.get_initial_envelope_correctors(
            img=img, dimension=dimension
        )

    img_out = _distance.get_initial_envelope(
        img,
        img_out,
        dimension=dimension,
        resolution=resolution[dimension],
        lower_boundary=lower_correctors,
        upper_boundary=upper_correctors,
    )

    for dimension in [1, 2]:
        lower_hull = None
        upper_hull = None

        if periodic[dimension]:

            lower_vertex = _voxels.get_nearest_boundary_index_face(
                img=img,
                dimension=dimension,
                label=0,
                forward=True,
                lower_skip=0,
                upper_skip=0,
            ).astype(np.int64)

            upper_vertex = _voxels.get_nearest_boundary_index_face(
                img=img,
                dimension=dimension,
                label=0,
                forward=False,
                lower_skip=0,
                upper_skip=0,
            ).astype(np.int64)

            _lower = _distance.get_boundary_hull(
                img=img_out,
                bound=lower_vertex,
                dimension=dimension,
                resolution=resolution[dimension],
                num_hull=4,
                forward=True,
            )

            _upper = _distance.get_boundary_hull(
                img=img_out,
                bound=upper_vertex,
                dimension=dimension,
                resolution=resolution[dimension],
                num_hull=4,
                forward=False,
            )

            # swap hulls
            lower_hull = _upper
            upper_hull = _lower

        _distance.get_parabolic_envelope(
            img=img_out,
            dimension=dimension,
            resolution=resolution[dimension],
            lower_hull=lower_hull,
            upper_hull=upper_hull,
        )

    if squared:
        return np.asarray(img_out)

    return np.asarray(np.sqrt(img_out))


def get_nearest_boundary_distance(
    subdomain: PaddedSubdomain | VerletSubdomain,
    img: NDArray[np.uint8],
    label: int,
    dimension: int,
    which_voxels: Literal["own", "pad", "all"] = "all",
    distance_to: Literal["own", "pad", "neighbor", "all"] = "all",
) -> dict[tuple[int, ...], NDArray[np.float32]]:
    """Determine the distance to the nearest subdomain boundary face for label.

    Args:
        subdomain: Subdomain object.
        img (np.ndarray): Input binary image.
        label: Label to search for.
        dimension (int): Dimension to search along.
        which_voxels (str, optional): Voxels to consider ("all", "own", "pad").
        distance_to (str, optional): Distance to compute:
                                    "all", "own", "pad", "neighbor".

    Returns:
        dict: Dictionary of distances for each face.

    """
    lower_skip = 0
    upper_skip = 0

    lower_distance = 0
    upper_distance = 0

    boundary_distance: dict[tuple[int, ...], NDArray[np.float32]] = {}

    for feature_id, feature in subdomain.features.faces.items():
        if feature_id[dimension] != 0:

            if feature.forward:
                if which_voxels == "own":
                    lower_skip = subdomain.pad[feature.info.arg_order[0]][0]
                elif which_voxels == "pad":
                    lower_skip = 2 * subdomain.pad[feature.info.arg_order[0]][0]

                if distance_to == "own":
                    lower_distance = subdomain.pad[feature.info.arg_order[0]][0]
                elif distance_to == "pad":
                    lower_distance = 2 * subdomain.pad[feature.info.arg_order[0]][0]
                elif distance_to == "neighbor":
                    lower_distance = -1

            elif not feature.forward:
                if which_voxels == "own":
                    upper_skip = subdomain.pad[feature.info.arg_order[0]][1]
                elif which_voxels == "pad":
                    upper_skip = 2 * subdomain.pad[feature.info.arg_order[0]][1]

                if distance_to == "own":
                    upper_distance = subdomain.pad[feature.info.arg_order[0]][1]
                elif distance_to == "pad":
                    upper_distance = 2 * subdomain.pad[feature.info.arg_order[0]][1]
                elif distance_to == "neighbor":
                    upper_distance = -1

            boundary_distance[feature_id] = _voxels.get_nearest_boundary_index_face(
                img=img,
                dimension=feature.info.arg_order[0],
                label=label,
                forward=feature.forward,
                lower_skip=lower_skip,
                upper_skip=upper_skip,
            ).astype(np.float32)

            if feature.forward:
                boundary_distance[feature_id] = np.where(
                    boundary_distance[feature_id] != -1,
                    boundary_distance[feature_id] - lower_distance,
                    np.inf,
                )

            else:
                boundary_distance[feature_id] = np.where(
                    boundary_distance[feature_id] != -1,
                    img.shape[feature.info.arg_order[0]]
                    - boundary_distance[feature_id]
                    - upper_distance
                    - 1,
                    np.inf,
                )

    return boundary_distance


def get_initial_correctors(
    subdomain: PaddedSubdomain | VerletSubdomain, img: NDArray[np.uint8], dimension: int
) -> tuple[
    None | NDArray[np.float32],
    None | NDArray[np.float32],
]:
    """Get the initial correctors for a subdomain and image.

    The correctors are defined as the absolute distance to the nearest solid
    (or phase change for multiphase).

    Args:
        subdomain: Subdomain object.
        img (np.ndarray): Input binary image.
        dimension (int, optional): Dimension to compute correctors for.

    Returns:
        tuple: (lower_correctors, upper_correctors)

    """
    boundary_distances = get_nearest_boundary_distance(
        subdomain=subdomain,
        img=img,
        label=0,
        dimension=dimension,
        which_voxels="pad",
        distance_to="own",
    )

    recv_data = communication.communicate_features(
        subdomain=subdomain, send_data=boundary_distances
    )

    lower_dim_key = None
    upper_dim_key = None
    for key in recv_data.keys():
        if key[dimension] < 0:
            lower_dim_key = key
        if key[dimension] > 0:
            upper_dim_key = key

    # Collect correctors - communication already swapped
    if lower_dim_key is not None:
        lower_correctors = recv_data[lower_dim_key]
    else:
        lower_correctors = None

    if upper_dim_key is not None:
        upper_correctors = recv_data[upper_dim_key]
    else:
        upper_correctors = None

    return lower_correctors, upper_correctors


def get_boundary_hull(
    subdomain: PaddedSubdomain | VerletSubdomain,
    img: NDArray[np.float32],
    og_img: NDArray[np.uint8],
    dimension: int,
    num_hull: int = 4,
) -> tuple[
    None | list[list[Hull]],
    None | list[list[Hull]],
]:
    """Get the boundary hull for a subdomain and image.

    Always pad the domain by 1 to allow for exact update of img and account for
    subdomain padding.

    Args:
        subdomain: Subdomain object.
        img (np.ndarray): Image to update.
        og_img (np.ndarray): Original image.
        dimension (int): Dimension to compute hull for.
        num_hull (int, optional): Number of hull points.

    Returns:
        tuple: (lower_hull, upper_hull)

    """
    if dimension not in {0, 1, 2}:
        raise ValueError("`dimension` must be an integer (0, 1, or 2) or None.")

    boundary_hull = {}

    lower_dim_key = tuple(-1 if i == dimension else 0 for i in range(3))
    upper_dim_key = tuple(1 if i == dimension else 0 for i in range(3))

    for feature_id, feature in subdomain.features.faces.items():
        if feature_id[dimension] != 0:

            if feature.forward:
                lower_skip = 2 * subdomain.pad[feature.info.arg_order[0]][0]
                upper_skip = 0
            else:
                lower_skip = 0
                upper_skip = 2 * subdomain.pad[feature.info.arg_order[0]][1]

            nearest_zero = _voxels.get_nearest_boundary_index_face(
                img=og_img,
                dimension=dimension,
                label=0,
                forward=feature.forward,
                lower_skip=lower_skip,
                upper_skip=upper_skip,
            ).astype(np.int64)

            boundary_hull[feature_id] = _distance.get_boundary_hull(
                img=img,
                bound=nearest_zero,
                dimension=feature.info.arg_order[0],
                resolution=subdomain.domain.resolution[dimension],
                num_hull=num_hull,
                forward=feature.forward,
                lower_skip=lower_skip,
                upper_skip=upper_skip,
            )

    recv_data = communication.communicate_features(
        subdomain=subdomain, send_data=boundary_hull
    )

    if lower_dim_key in recv_data.keys():
        lower_hull = recv_data[lower_dim_key]
    else:
        lower_hull = None

    if upper_dim_key in recv_data.keys():
        upper_hull = recv_data[upper_dim_key]
    else:
        upper_hull = None

    return lower_hull, upper_hull
