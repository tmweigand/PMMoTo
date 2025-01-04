import numpy as np
from pmmoto.core import voxels
from pmmoto.core import _voxels
from pmmoto.core import communication
from . import _distance

__all__ = ["edt", "edt2d", "edt3d"]


def edt(img, subdomain=None):
    """
    Calculate the exact Euclidean transform of an image

    Args:
        subdomain (_type_): _description_
        img (numpy array): _description_
    """
    img_out = np.copy(img).astype(np.float32)
    if subdomain is not None:
        if subdomain.domain.periodic or subdomain.num_subdomains > 1:
            dimension = 0

            lower_correctors, upper_correctors = get_initial_correctors(
                subdomain=subdomain, img=img, dimension=dimension
            )

            img_out = _distance.get_initial_envelope(
                img,
                img_out,
                dimension=0,
                lower_boundary=lower_correctors,
                upper_boundary=upper_correctors,
            )

            lower_hull, upper_hull = get_boundary_hull(
                subdomain=subdomain, img=img_out, dimension=1
            )

            _distance.get_parabolic_envelope(
                img_out,
                dimension=1,
                lower_hull=lower_hull,
                upper_hull=upper_hull,
            )

            lower_hull, upper_hull = get_boundary_hull(
                subdomain=subdomain, img=img_out, dimension=2
            )

            _distance.get_parabolic_envelope(
                img_out,
                dimension=2,
                lower_hull=lower_hull,
                upper_hull=upper_hull,
            )

            img_out = np.asarray(np.sqrt(img_out))

    else:  # Simply perform the edt with no corrections
        if len(img.shape) == 3:
            img_out = edt3d(img)
        if len(img.shape) == 2:
            img_out = edt2d(img)
    return img_out


def edt2d(img, periodic=[False, False]):
    """
    Perform an exact Euclidean transform on a image
    For the first pass, collect all the solids (or transitions on all faces)
    The direction needs to be completed first. Then the off-direction need to be correct.
    C
    """
    img_out = np.copy(img).astype(np.float32)

    dimension = 1
    lower_correctors = None
    upper_correctors = None
    if periodic[dimension]:
        lower_correctors, upper_correctors = (
            _distance.get_initial_envelope_correctors_2d(img, dimension=dimension)
        )

    img_out = _distance.get_initial_envelope_2d(
        img,
        img_out,
        dimension=dimension,
        lower_boundary=lower_correctors,
        upper_boundary=upper_correctors,
    )

    dimension = 0
    l_hull = None
    r_hull = None
    if periodic[dimension]:
        num_hull = 3
        l_hull, r_hull = _distance.get_boundary_hull_2d(
            img_out, dimension=dimension, num_hull=num_hull
        )

    _distance.get_parabolic_envelope_2d(
        img_out, dimension=dimension, lower_hull=r_hull, upper_hull=l_hull
    )
    return np.asarray(np.sqrt(img_out))


def edt3d(img, periodic=[False, False, False]):
    """
    Perform an exact Euclidean transform on a image
    For the first pass, collect all the solids (or transitions on all faces)
    The direction needs to be completed first. Then the off-direction need to be correct.
    C
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
        lower_boundary=lower_correctors,
        upper_boundary=upper_correctors,
    )

    for dimension in [1, 2]:
        num_hull = 2
        lower_hull = None
        upper_hull = None

        if periodic[dimension]:
            # Swap sides here on the fly
            upper_hull = _distance.get_boundary_hull(
                img_out, dimension=dimension, num_hull=num_hull, left=True
            )

            lower_hull = _distance.get_boundary_hull(
                img_out, dimension=dimension, num_hull=num_hull, left=False
            )

        _distance.get_parabolic_envelope(
            img_out, dimension=dimension, lower_hull=lower_hull, upper_hull=upper_hull
        )

    return np.asarray(np.sqrt(img_out))


def get_initial_correctors(subdomain, img, dimension=None):
    """
    Get the initial correctors for a subdomain and image
    """
    if dimension is not None and dimension not in {0, 1, 2}:
        raise ValueError("`dimension` must be an integer (0, 1, or 2) or None.")

    dims = {0, 1, 2}
    dims.remove(dimension)
    dim1, dim2 = dims

    boundary_solids = voxels.get_nearest_boundary_index(
        subdomain=subdomain, img=img, label=0, dimension=dimension
    )

    recv_data = communication.communicate(
        subdomain=subdomain, send_data=boundary_solids, feature_types=["faces"]
    )

    lower_dim_key = next(key for key in recv_data.keys() if key[dimension] < 0)
    upper_dim_key = next(key for key in recv_data.keys() if key[dimension] > 0)

    # initialize correctors
    lower_correctors = np.zeros([img.shape[dim1], img.shape[dim2]])
    upper_correctors = np.zeros([img.shape[dim1], img.shape[dim2]])
    # correct indexes
    lower_correctors = np.where(
        recv_data[upper_dim_key] != -1,
        img.shape[dimension] - recv_data[upper_dim_key],
        np.inf,
    )
    upper_correctors = np.where(
        recv_data[lower_dim_key] != -1, recv_data[lower_dim_key] + 1, np.inf
    )

    return lower_correctors, upper_correctors


def get_boundary_hull(subdomain, img, dimension=None, num_hull=2):
    """
    Get the boundary hull for a subdomain and image
    """
    if dimension is not None and dimension not in {0, 1, 2}:
        raise ValueError("`dimension` must be an integer (0, 1, or 2) or None.")

    boundary_hull = {}
    feature_types = ["faces"]

    for feature_type in feature_types:
        for feature_id, feature in subdomain.features[feature_type].items():
            if dimension is None or feature_id[dimension] != 0:
                boundary_hull[feature_id] = _distance.get_boundary_hull(
                    img=img,
                    dimension=feature.info["argOrder"][0],
                    num_hull=num_hull,
                    left=feature.forward,
                )

    recv_data = communication.communicate(
        subdomain=subdomain, send_data=boundary_hull, feature_types=["faces"]
    )

    lower_dim_key = next(key for key in recv_data.keys() if key[dimension] < 0)
    upper_dim_key = next(key for key in recv_data.keys() if key[dimension] > 0)

    # Swap sides
    lower_hull = boundary_hull[upper_dim_key]
    upper_hull = boundary_hull[lower_dim_key]

    return lower_hull, upper_hull
