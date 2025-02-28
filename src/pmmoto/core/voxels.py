"""voxel.py"""

import numpy as np
from . import _voxels
from . import communication


__all__ = [
    "renumber_image",
    "get_nearest_boundary_index",
    "get_id",
    "get_boundary_voxels",
    "gen_grid_to_label_map",
    "gen_inlet_label_map",
    "gen_outlet_label_map",
    "count_label_voxels",
    "match_neighbor_boundary_voxels",
    "match_global_boundary_voxels",
]


def renumber_image(img, conversion_map: dict):
    """
    Renumbers an image using a provided mapping.

    Args:
        img (Any): The image to be renumbered. Typically, this will be a 3D or 2D array-like structure.
        conversion_map (dict): A dictionary mapping current image IDs to new image IDs.
            Example: {1: 101, 2: 102, ...}
            Note: All IDs in `img` must be defined in `conversion_map`.

    Returns:
        Any: The renumbered image, with IDs replaced based on the mapping.

    Note:
        This function assumes all required IDs are present in the `conversion_map`.
        No error handling is performed for missing or invalid keys.

    Example:
        img = [[1, 2], [2, 1]]
        conversion_map = {1: 101, 2: 102}
        renumber_image(img, conversion_map)
        # Output: [[101, 102], [102, 101]]
    """

    img = _voxels.renumber_grid(img, conversion_map)

    return np.ascontiguousarray(img)


def get_nearest_boundary_index(
    subdomain, img, label, dimension=None, which_voxels="all"
):
    """
    Determines the index nearest each subdomain boundary face for a specified
    label in img. The start and end locations can be controlled but

        which_voxels = "all" start = 0, end = 0
        which_voxels = "own" start = pad[0], end = pad[1]
        which_voxels = "pad" start = 2*pad[0], end = 2*pad[1]

    Note: The boundary index is always with respect to img.shape[dimension].
    Start and end are used for searching.

    Args:
        subdomain (_type_): _description_
        img (_type_): _description_
    """

    if dimension is not None and dimension not in {0, 1, 2}:
        raise ValueError("`dimension` must be an integer (0, 1, or 2) or None.")

    lower_skip = 0
    upper_skip = 0

    boundary_index = {}

    for feature_id, feature in subdomain.features["faces"].items():
        if dimension is None or feature_id[dimension] != 0:

            if feature.forward:
                if which_voxels == "own":
                    lower_skip = subdomain.pad[feature.info["argOrder"][0]][0]
                elif which_voxels == "pad":
                    lower_skip = 2 * subdomain.pad[feature.info["argOrder"][0]][0]
            elif not feature.forward:
                if which_voxels == "own":
                    upper_skip = subdomain.pad[feature.info["argOrder"][0]][1]
                elif which_voxels == "pad":
                    upper_skip = 2 * subdomain.pad[feature.info["argOrder"][0]][1]

            boundary_index[feature_id] = _voxels.get_nearest_boundary_index_face(
                img=img,
                dimension=feature.info["argOrder"][0],
                label=label,
                forward=feature.forward,
                lower_skip=lower_skip,
                upper_skip=upper_skip,
            ).astype(np.float32)

    return boundary_index


def get_boundary_voxels(subdomain, img, neighbors_only=False):
    """
    This function returns the values on the boundary features.
    The features are divided into
        own: feature voxels owned by subdomain
        neighbor: feature voxels owned by a neighbor subdomain

    Args:
        subdomain (_type_): _description_
        img (_type_): _description_
    """
    out_voxels = {}

    boundary_types = ["own", "neighbor"]
    feature_types = ["faces", "edges", "corners"]
    for feature_type in feature_types:
        for feature_id, feature in subdomain.features[feature_type].items():
            if neighbors_only and feature.neighbor_rank < -1:
                continue
            out_voxels[feature_id] = {}
            for kind in boundary_types:
                out_voxels[feature_id][kind] = img[
                    feature.loop[kind][0][0] : feature.loop[kind][0][1],
                    feature.loop[kind][1][0] : feature.loop[kind][1][1],
                    feature.loop[kind][2][0] : feature.loop[kind][2][1],
                ].flatten()

    return out_voxels


def get_id(x, total_voxels):
    """
    Wrapper to _get_id

    Determine the ID for a voxel.
    Input:
        - x: 3D index of the voxel (x, y, z)
        - total_voxels: Size of the domain (number of voxels in each dimension)
    Output:
        - Global or local ID of the voxel.
    Period
    Periodic boundary conditions are applied by using modulo arithmetic.
    """
    id = _voxels.get_id(
        np.array(x, dtype=np.int64), np.array(total_voxels, dtype=np.uint64)
    )
    return id


def gen_grid_to_label_map(grid, labels):
    """_summary_

    Args:
        grid (_type_): _description_
        label_grid (_type_): _description_
    """
    assert grid.shape == labels.shape

    return _voxels.gen_grid_to_label_map(
        grid.astype(np.uint8), labels.astype(np.uint64)
    )


def count_label_voxels(grid, map):
    """_summary_

    Args:
        grid (_type_): _description_
        map (_type_): _description_
    """
    _map = _voxels.count_label_voxels(grid, map)


def boundary_voxels_pack(subdomain, boundary_voxels):
    """
    This function packs the data to send based on get_boundary_voxels
    """
    send_data = {}
    periodic_data = {}

    feature_types = ["faces", "edges", "corners"]
    for feature_type in feature_types:
        for feature_id, feature in subdomain.features[feature_type].items():
            if feature.neighbor_rank > -1 and feature.neighbor_rank != subdomain.rank:
                send_data[feature_id] = {
                    "rank": subdomain.rank,
                    "own": boundary_voxels[feature_id]["own"],
                    "neighbor": boundary_voxels[feature_id]["neighbor"],
                }
            if feature.neighbor_rank == subdomain.rank:
                periodic_data[feature_id] = {
                    "rank": subdomain.rank,
                    "own": boundary_voxels[feature_id]["own"],
                    "neighbor": boundary_voxels[feature_id]["neighbor"],
                }

    return send_data, periodic_data


def boundary_voxels_unpack(subdomain, boundary_voxels, recv_data):
    """
    Unpack the neighboring boundary neighbor data. This also handles
    periodic boundary conditions.

    The feature_id for the return value has been accounted for:
        own_data[feature_id] = data_out[feature_id]

    """

    data_out = {}

    feature_types = ["faces", "edges", "corners"]
    for feature_type in feature_types:
        for feature_id, feature in subdomain.features[feature_type].items():
            if feature.n_proc > -1 and feature.n_proc != subdomain.rank:
                data_out[feature_id] = recv_data[feature_id]
            elif feature.n_proc == subdomain.rank:
                data_out[feature_id] = boundary_voxels[feature.opp_info]

    return data_out


def match_neighbor_boundary_voxels(
    subdomain, boundary_voxels, recv_data, skip_zero=False
):
    """
    Matches boundary voxels of the subdomain with neighboring voxels and returns unique matches.

    Args:
        subdomain (object): Subdomain object containing feature information.
        boundary_voxels (dict): Dictionary with 'own' and 'neighbor' boundary voxel data.
        recv_data (dict): Received data containing neighbor voxel information.
        skip_zero (bool): Determines if matches with label of 0 should be added.
                         Useful for connnected componentes.

    Returns:
        dict: Unique matches in the format
        key: (subdomain rank, own voxel)
        neighbor: list[neighbor rank, neighbor voxel)]
    """
    unique_matches = {}

    feature_types = ["faces", "edges", "corners"]
    for feature_type in feature_types:
        for feature_id, feature in subdomain.features[feature_type].items():
            if feature_id in recv_data:
                to_match = np.stack(
                    [
                        np.concatenate(
                            (
                                boundary_voxels[feature_id]["own"],
                                boundary_voxels[feature_id]["neighbor"],
                            )
                        ),
                        np.concatenate(
                            (
                                recv_data[feature_id]["neighbor"],
                                recv_data[feature_id]["own"],
                            )
                        ),
                    ],
                    axis=1,
                )

                matches = _voxels.find_unique_pairs(to_match)

                if skip_zero:
                    matches = matches[~np.any(matches == 0, axis=1)]

                unique_matches = _voxels.process_matches_by_feature(
                    matches, unique_matches, subdomain.rank, feature.neighbor_rank
                )

    return unique_matches


def match_global_boundary_voxels(subdomain, matches, label_count):
    """_summary_

    Args:
        subdomain (_type_): _description_
        matches (_type_): _description_
    """

    ### Send number of labels on rank for re-labeling
    matches["label_count"] = label_count
    all_matches = communication.all_gather(matches)

    ### Generate the local-global label map
    local_global_map, boundary_label_count = _voxels.merge_matched_voxels(all_matches)

    final_map, global_label_count = local_to_global_labeling(
        all_matches, local_global_map, boundary_label_count, own=subdomain.rank
    )

    return final_map, global_label_count


def local_to_global_labeling(all_matches, boundary_map, boundary_label_count, own=None):
    """ """

    local_counts = [matches["label_count"] for matches in all_matches]
    boundary_counts = [len(matches) for matches in all_matches]

    num_ranks = len(local_counts)

    # Determine unique global labels relative so expect negative!
    local_starts = {}
    for rank in range(num_ranks):
        if rank == 0:
            local_starts[rank] = boundary_label_count
        else:
            local_starts[rank] = (
                local_starts[rank - 1]
                + local_counts[rank - 1]
                - boundary_counts[rank - 1]
                + 1
            )

    ### Generate the global id for non-boundary labels as well
    final_map = {}
    label_count = 0
    for rank, match in enumerate(all_matches):
        final_map[rank] = {0: 0}
        count = 0
        for n in range(1, match["label_count"] + 1):
            if (rank, n) in boundary_map:
                final_map[rank][n] = boundary_map[(rank, n)]["global_id"]
            else:
                final_map[rank][n] = local_starts[rank] + count
                count += 1
        label_count = count

    max_label = label_count + local_starts[num_ranks - 1]
    if max_label < 1:
        max_label = boundary_label_count

    if own is not None and own in final_map:
        return final_map[own], max_label
    else:
        return final_map, max_label


def gen_inlet_label_map(subdomain, label_grid):
    """
    Determine which face is on inlet.
    Currently restricted to a single face

    Args:
        subdomain (_type_): _description_
        label_grid (_type_): _description_
    """
    inlet_labels = np.empty(0)
    for feature_id, feature in subdomain.features["faces"].items():
        if feature.inlet:
            inlet_labels = np.unique(
                label_grid[
                    feature.loop["own"][0][0] : feature.loop["own"][0][1],
                    feature.loop["own"][1][0] : feature.loop["own"][1][1],
                    feature.loop["own"][2][0] : feature.loop["own"][2][1],
                ]
            )
    return inlet_labels


def gen_outlet_label_map(subdomain, label_grid):
    """
    Determine which face is on outlet.
    Currently restricted to a single face

    Args:
        subdomain (_type_): _description_
        label_grid (_type_): _description_
    """
    outlet_labels = []
    for feature_id, feature in subdomain.features["faces"].items():
        if feature.outlet:
            outlet_labels = np.unique(
                label_grid[
                    feature.loop["own"][0][0] : feature.loop["own"][0][1],
                    feature.loop["own"][1][0] : feature.loop["own"][1][1],
                    feature.loop["own"][2][0] : feature.loop["own"][2][1],
                ]
            )

    return outlet_labels
