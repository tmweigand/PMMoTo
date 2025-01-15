import numpy as np
from pmmoto.core import voxels
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
        if subdomain.domain.periodic or subdomain.domain.num_subdomains > 1:

            dimension = 0
            lower_correctors, upper_correctors = get_initial_correctors(
                subdomain=subdomain, img=img, dimension=dimension, own=True
            )

            img_out = _distance.get_initial_envelope(
                img,
                img_out,
                dimension=dimension,
                lower_boundary=lower_correctors,
                upper_boundary=upper_correctors,
            )

            for dimension in [1, 2]:

                lower_hull, upper_hull = get_boundary_hull(
                    subdomain=subdomain,
                    img=img_out,
                    dimension=dimension,
                    own=True,
                )

                _distance.get_parabolic_envelope(
                    img_out,
                    dimension=dimension,
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


def adjust_vertex(vertex, size, pad=(0, 0), is_lower=True, distance=True):
    """
    Adjusts a vertex based on its type (lower or upper), size, padding, and distance flag.

    Parameters:
        vertex (int): The vertex location to adjust.
        size (int): The size used for adjustment.
        pad (tuple): A tuple containing padding values (pad_start, pad_end).
        is_lower (bool): If True, adjust as a lower vertex; otherwise, as an upper vertex.
        distance (bool): If True, adjust for distance; otherwise, adjust for size and padding.

    Returns:
        int: The adjusted vertex.
    """
    if is_lower:
        if distance:
            return vertex + 1
        else:
            return vertex + size
    else:
        if distance:
            return size - vertex
        else:
            return vertex - size + pad[0] + pad[1]


def adjust_hull(hull, size, pad=(0, 0), is_lower=True, distance=False):
    """
    Adjusts a hull (lower or upper) based on the size, padding, and distance flag.

    Parameters:
        hull (list): The hull to adjust.
        size (int): The size used for adjustments.
        pad (tuple): A tuple containing padding values (pad_start, pad_end).
        is_lower (bool): If True, adjusts as a lower hull; otherwise, as an upper hull.
        distance (bool): Distance flag for the adjustment.

    Returns:
        list: The adjusted hull.
    """
    adjusted_hull = []
    for sub_hull in hull:
        adjusted_sub_hull = []
        for h in sub_hull:
            adjusted_h = {
                "vertex": adjust_vertex(h["vertex"], size, pad, is_lower, distance),
                "range": adjust_vertex(h["range"], size, pad, is_lower, distance),
                "height": h["height"],
            }
            adjusted_sub_hull.append(adjusted_h)
        adjusted_hull.append(adjusted_sub_hull)
    return adjusted_hull


def adjust_hulls(
    lower_hull, upper_hull, size, lower_distance=False, upper_distance=False, pad=(0, 0)
):
    """
    Adjusts both lower and upper hulls based on the size, distance flags, and padding.

    Parameters:
        lower_hull (list): The lower hull to adjust.
        upper_hull (list): The upper hull to adjust.
        size (int): The size used for adjustments.
        lower_distance (bool): Distance flag for the lower hull adjustment.
        upper_distance (bool): Distance flag for the upper hull adjustment.
        pad (tuple): A tuple containing padding values (pad_start, pad_end).

    Returns:
        tuple: A tuple containing the adjusted lower and upper hulls.
    """
    adjusted_lower_hull = adjust_hull(
        lower_hull, size, pad, is_lower=True, distance=lower_distance
    )
    adjusted_upper_hull = adjust_hull(
        upper_hull, size, pad, is_lower=False, distance=upper_distance
    )
    return adjusted_lower_hull, adjusted_upper_hull


# def adjust_lower_vertex(lower, size, distance=True):
#     """ """
#     if distance:
#         lower = lower + 1
#     else:
#         lower = lower + size

#     return lower


# def adjust_upper_vertex(upper, size, pad, distance=True):
#     """ """
#     if distance:
#         upper = size - upper
#     else:
#         upper = upper - size + pad[0] + pad[1]

#     return upper


# def adjust_hulls(
#     lower_hull, upper_hull, size, lower_distance=False, upper_distance=False, pad=(0, 0)
# ):

#     import copy

#     _lower_hull = copy.deepcopy(lower_hull)
#     _upper_hull = copy.deepcopy(upper_hull)

#     for hull in _lower_hull:
#         for h in hull:
#             h["vertex"] = adjust_lower_vertex(
#                 h["vertex"], size, distance=lower_distance
#             )
#             h["range"] = adjust_lower_vertex(h["range"], size, distance=lower_distance)

#     for hull in _upper_hull:
#         for h in hull:
#             h["vertex"] = adjust_upper_vertex(
#                 h["vertex"], size, pad=pad, distance=upper_distance
#             )
#             h["range"] = adjust_upper_vertex(
#                 h["range"], size, pad=pad, distance=upper_distance
#             )

#     return _lower_hull, _upper_hull


# def adjust_lower_hull(lower_hull, size, distance=False, pad=(0, 0)):

#     import copy

#     _lower_hull = copy.deepcopy(lower_hull)

#     for hull in _lower_hull:
#         for h in hull:
#             h["vertex"] = adjust_lower_vertex(h["vertex"], size, distance=distance)
#             h["range"] = adjust_lower_vertex(h["range"], size, distance=distance)

#     return _lower_hull


# def adjust_upper_hull(upper_hull, size, distance=False, pad=(0, 0)):

#     import copy

#     _upper_hull = copy.deepcopy(upper_hull)

#     for hull in _upper_hull:
#         for h in hull:
#             h["vertex"] = adjust_upper_vertex(
#                 h["vertex"], size, pad=pad, distance=distance
#             )
#             h["range"] = adjust_upper_vertex(
#                 h["range"], size, pad=pad, distance=distance
#             )

#     return _upper_hull


def edt2d(img, periodic=[False, False]):
    """
    Perform an exact Euclidean transform on a image
    For the first pass, collect all the solids (or transitions on all faces)
    The direction needs to be completed first. Then the off-direction need to be correct.
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
        lower_boundary=_lower_correctors,
        upper_boundary=_upper_correctors,
    )

    dimension = 0
    lower_hull = None
    upper_hull = None
    if periodic[dimension]:
        num_hull = 3
        _lower, _upper = _distance.get_boundary_hull_2d(
            img_out, dimension=dimension, num_hull=num_hull
        )

        _lower, _upper = adjust_hulls(_lower, _upper, img.shape[dimension])

        # swap hulls
        lower_hull = _upper
        upper_hull = _lower

    _distance.get_parabolic_envelope_2d(
        img_out, dimension=dimension, lower_hull=lower_hull, upper_hull=upper_hull
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
        lower_hull = None
        upper_hull = None

        if periodic[dimension]:

            _lower = _distance.get_boundary_hull(
                img_out, dimension=dimension, num_hull=4, left=True
            )

            _upper = _distance.get_boundary_hull(
                img_out, dimension=dimension, num_hull=4, left=False
            )

            _lower, _upper = adjust_hulls(_lower, _upper, img.shape[dimension])

            # swap hulls
            lower_hull = _upper
            upper_hull = _lower

        _distance.get_parabolic_envelope(
            img_out, dimension=dimension, lower_hull=lower_hull, upper_hull=upper_hull
        )

    return np.asarray(np.sqrt((img_out)))


def get_initial_correctors(subdomain, img, dimension=None, own=False):
    """
    Get the initial correctors for a subdomain and image.
    The correctors is defined as the absolute distance to the nearest solid
        (or phase change for multiphase).

    The correctors are adjusted as:
        lower_corrector = neighbor_data +
        upper_corrector = neighbor_data +


    """
    if dimension is not None and dimension not in {0, 1, 2}:
        raise ValueError("`dimension` must be an integer (0, 1, or 2) or None.")

    dims = {0, 1, 2}
    dims.remove(dimension)
    dim1, dim2 = dims

    boundary_solids = voxels.get_nearest_boundary_index(
        subdomain=subdomain,
        img=img,
        label=0,
        dimension=dimension,
        own=own,
        distance=True,
    )

    recv_data = communication.communicate_features(
        subdomain=subdomain, send_data=boundary_solids, feature_types=["faces"]
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
        # initialize correctors
        lower_correctors = np.zeros([img.shape[dim1], img.shape[dim2]])
        # need to adjust flag
        lower_correctors = np.where(
            recv_data[lower_dim_key] != -1,
            recv_data[lower_dim_key],
            np.inf,
        )
    else:
        lower_correctors = None

    # Collect correctors - communication already swapped
    if upper_dim_key is not None:
        # initialize correctors
        upper_correctors = np.zeros([img.shape[dim1], img.shape[dim2]])
        # need to adjust flag
        upper_correctors = np.where(
            recv_data[upper_dim_key] != -1, recv_data[upper_dim_key], np.inf
        )
    else:
        upper_correctors = None

    return lower_correctors, upper_correctors


def get_boundary_hull(subdomain, img, dimension, num_hull=4, own=False):
    """
    Get the boundary hull for a subdomain and image.
    Always pad the domain by 1 to allow for exact update of img.
    """
    if dimension not in {0, 1, 2}:
        raise ValueError("`dimension` must be an integer (0, 1, or 2) or None.")

    boundary_hull = {}
    feature_types = ["faces"]

    dim_key = [0, 0, 0]
    lower_dim_key = dim_key
    lower_dim_key[dimension] = -1
    lower_dim_key = tuple(lower_dim_key)

    upper_dim_key = dim_key
    upper_dim_key[dimension] = 1
    upper_dim_key = tuple(upper_dim_key)

    for feature_type in feature_types:
        for feature_id, feature in subdomain.features[feature_type].items():

            pad = [0, 0]
            # times two to skip nodes on neighbor processes
            if own and feature.forward:
                pad[0] = 2 * subdomain.pad[feature.info["argOrder"][0]][0]
            if own and not feature.forward:
                pad[1] = 2 * subdomain.pad[feature.info["argOrder"][0]][1]

            if feature_id[dimension] != 0:
                boundary_hull[feature_id] = _distance.get_boundary_hull(
                    img=img,
                    dimension=feature.info["argOrder"][0],
                    num_hull=num_hull,
                    left=feature.forward,
                    lower_pad=pad[0],
                    upper_pad=pad[1],
                )

            if feature_id == lower_dim_key:
                boundary_hull[feature_id] = adjust_hull(
                    boundary_hull[feature_id],
                    img.shape[dimension],
                    pad=pad,
                    is_lower=True,
                    distance=True,
                )

            if feature_id == upper_dim_key:
                boundary_hull[feature_id] = adjust_hull(
                    boundary_hull[feature_id],
                    img.shape[dimension],
                    pad=pad,
                    is_lower=False,
                    distance=False,
                )

    recv_data = communication.communicate_features(
        subdomain=subdomain, send_data=boundary_hull, feature_types=feature_types
    )

    if lower_dim_key in recv_data.keys():
        lower_hull = recv_data[lower_dim_key]
    else:
        lower_hull = None

    if upper_dim_key in recv_data.keys():
        upper_hull = recv_data[upper_dim_key]
        upper_hull = adjust_hull(upper_hull, img.shape[dimension] - 1, is_lower=True)
    else:
        upper_hull = None

    return lower_hull, upper_hull
