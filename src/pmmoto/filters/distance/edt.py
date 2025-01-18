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
        if subdomain.domain.periodic or subdomain.domain.num_subdomains > 1:

            dimension = 0
            lower_correctors, upper_correctors = get_initial_correctors(
                subdomain=subdomain, img=img, dimension=dimension
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
                    og_img=img,
                    dimension=dimension,
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
            num_hull=num_hull,
            forward=True,
        )

        _upper = _distance.get_boundary_hull_2d(
            img=img_out,
            bound=upper_vertex,
            dimension=dimension,
            num_hull=num_hull,
            forward=False,
        )

        print(upper_vertex)
        for l in _upper:
            print(l)

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

            lower_vertex = _voxels.get_nearest_boundary_index_face(
                img=img,
                dimension=dimension,
                label=0,
                forward=True,
                lower_skip=0,
                upper_skip=0,
            )

            upper_vertex = _voxels.get_nearest_boundary_index_face(
                img=img,
                dimension=dimension,
                label=0,
                forward=False,
                lower_skip=0,
                upper_skip=0,
            )

            _lower = _distance.get_boundary_hull(
                img=img_out,
                bound=lower_vertex,
                dimension=dimension,
                num_hull=4,
                forward=True,
            )

            _upper = _distance.get_boundary_hull(
                img=img_out,
                bound=upper_vertex,
                dimension=dimension,
                num_hull=4,
                forward=False,
            )

            # swap hulls
            lower_hull = _upper
            upper_hull = _lower

        _distance.get_parabolic_envelope(
            img_out,
            dimension=dimension,
            lower_hull=lower_hull,
            upper_hull=upper_hull,
        )

    return np.asarray(np.sqrt(img_out))


def get_nearest_boundary_distance(
    subdomain,
    img,
    label,
    dimension,
    which_voxels="all",
    distance_to="all",
):
    """
    Determines the distance of the index nearest each subdomain boundary face for a specified
    label in img. The start and end locations can be controlled but

        which_voxels = "all" start = 0, end = 0
        which_voxels = "own" start = pad[0], end = pad[1]
        which_voxels = "pad" start = 2*pad[0], end = 2*pad[1]

    Args:
        subdomain (_type_): _description_
        img (_type_): _description_
        label (_type_): _description_
        dimension (_type_): _description_
        which_voxels (str, optional): _description_. Defaults to "all".
        distance_to (str, optional): _description_. Defaults to "all".
    """
    if dimension is not None and dimension not in {0, 1, 2}:
        raise ValueError("`dimension` must be an integer (0, 1, or 2) or None.")

    lower_skip = 0
    upper_skip = 0

    lower_distance = 0
    upper_distance = 0

    boundary_distance = {}

    for feature_id, feature in subdomain.features["faces"].items():
        if dimension is None or feature_id[dimension] != 0:

            if feature.forward:
                if which_voxels == "own":
                    lower_skip = subdomain.pad[feature.info["argOrder"][0]][0]
                elif which_voxels == "pad":
                    lower_skip = 2 * subdomain.pad[feature.info["argOrder"][0]][0]

                if distance_to == "own":
                    lower_distance = subdomain.pad[feature.info["argOrder"][0]][0]
                elif distance_to == "pad":
                    lower_distance = 2 * subdomain.pad[feature.info["argOrder"][0]][0]
                elif distance_to == "neighbor":
                    lower_distance = -1

            elif not feature.forward:
                if which_voxels == "own":
                    upper_skip = subdomain.pad[feature.info["argOrder"][0]][1]
                elif which_voxels == "pad":
                    upper_skip = 2 * subdomain.pad[feature.info["argOrder"][0]][1]

                if distance_to == "own":
                    upper_distance = subdomain.pad[feature.info["argOrder"][0]][1]
                elif distance_to == "pad":
                    upper_distance = 2 * subdomain.pad[feature.info["argOrder"][0]][1]
                elif distance_to == "neighbor":
                    upper_distance = -1

            boundary_distance[feature_id] = _voxels.get_nearest_boundary_index_face(
                img=img,
                dimension=feature.info["argOrder"][0],
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
                    img.shape[feature.info["argOrder"][0]]
                    - boundary_distance[feature_id]
                    - upper_distance
                    - 1,
                    np.inf,
                )

    return boundary_distance


def get_initial_correctors(subdomain, img, dimension=None):
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

    boundary_distances = get_nearest_boundary_distance(
        subdomain=subdomain,
        img=img,
        label=0,
        dimension=dimension,
        which_voxels="pad",
        distance_to="own",
    )

    recv_data = communication.communicate_features(
        subdomain=subdomain, send_data=boundary_distances, feature_types=["faces"]
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


def get_boundary_hull(subdomain, img, og_img, dimension, num_hull=4):
    """
    Get the boundary hull for a subdomain and image.
    Always pad the domain by 1 to allow for exact update of img.
    """
    if dimension not in {0, 1, 2}:
        raise ValueError("`dimension` must be an integer (0, 1, or 2) or None.")

    boundary_hull = {}

    dim_key = [0, 0, 0]
    lower_dim_key = dim_key
    lower_dim_key[dimension] = -1
    lower_dim_key = tuple(lower_dim_key)

    upper_dim_key = dim_key
    upper_dim_key[dimension] = 1
    upper_dim_key = tuple(upper_dim_key)

    for feature_id, feature in subdomain.features["faces"].items():
        if feature_id[dimension] != 0:
            lower_skip = 2 * subdomain.pad[feature.info["argOrder"][0]][0]
            upper_skip = 2 * subdomain.pad[feature.info["argOrder"][0]][1]

            nearest_zero = _voxels.get_nearest_boundary_index_face(
                img=og_img,
                dimension=dimension,
                label=0,
                forward=feature.forward,
                lower_skip=lower_skip,
                upper_skip=upper_skip,
            )

            boundary_hull[feature_id] = _distance.get_boundary_hull(
                img=img,
                bound=nearest_zero,
                dimension=feature.info["argOrder"][0],
                num_hull=num_hull,
                forward=feature.forward,
                lower_skip=lower_skip,
                upper_skip=upper_skip,
            )

    recv_data = communication.communicate_features(
        subdomain=subdomain, send_data=boundary_hull, feature_types=["faces"]
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
