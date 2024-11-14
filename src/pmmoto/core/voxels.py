"""voxel.py"""

import numpy as np
from . import _voxels
from . import communication


__all__ = [
    "renumber_image",
    "get_id",
    "get_boundary_voxels",
    "gen_grid_to_label_map",
    "gen_inlet_label_map",
    "gen_outlet_label_map",
    "count_label_voxels",
    "match_neighbor_boundary_voxels",
    "match_global_boundary_voxels",
]


def renumber_image(img, conversion_map):
    """_summary_

    Args:
        img (_type_): _description_
        conversion_map (_type_): _description_
    """
    return _voxels._renumber_grid(img, conversion_map)


def get_boundary_voxels(subdomain, img):
    """_summary_

    Args:
        subdomain (_type_): _description_
        img (_type_): _description_
    """
    out_voxels = {}

    boundary_types = ["own", "neighbor"]
    feature_types = ["faces", "edges", "corners"]
    for feature_type in feature_types:
        for feature_id, feature in subdomain.features[feature_type].items():
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


def match_neighbor_boundary_voxels(subdomain, boundary_voxels, recv_data):
    """
    Matches boundary voxels of the subdomain with neighboring voxels and returns unique matches.

    Args:
        subdomain (object): Subdomain object containing feature information.
        boundary_voxels (dict): Dictionary with 'own' and 'neighbor' boundary voxel data.
        recv_data (dict): Received data containing neighbor voxel information.

    Returns:
        dict: Unique matches in the format
        key:(subdomain rank, own voxel)
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

                for match in np.unique(to_match, axis=0):
                    match_tuple = (subdomain.rank, match[0])
                    neighbor_tuple = (
                        recv_data[feature_id]["rank"],
                        match[1],
                    )
                    if match_tuple not in unique_matches.keys():
                        unique_matches[match_tuple] = {"neighbor": [neighbor_tuple]}
                    else:
                        if (
                            neighbor_tuple
                            not in unique_matches[match_tuple]["neighbor"]
                        ):
                            unique_matches[match_tuple]["neighbor"].append(
                                neighbor_tuple
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
    local_global_map = _voxels._merge_matched_voxels(all_matches)

    ### Generate the global id for non-boundary labels as well
    final_map = {}
    local_start = local_global_map[subdomain.rank]
    count = 0
    for n in range(label_count):
        if (subdomain.rank, n) in local_global_map:
            final_map[n] = local_global_map[(subdomain.rank, n)]["global_id"]
        else:
            final_map[n] = local_start + count
            count += 1

    return final_map


def gen_inlet_label_map(subdomain, label_grid):
    """
    Determine which face is on inlet.
    Currently restricted to a single face

    Args:
        subdomain (_type_): _description_
        label_grid (_type_): _description_
    """
    inlet_labels = None
    feature_types = ["faces"]  # only faces can be an inlet
    for feature_type in feature_types:
        for feature_id, feature in subdomain.features[feature_type].items():
            if feature.inlet:
                inlet_labels = np.unique(
                    label_grid[
                        feature.loop["own"][0][0] : feature.loop["own"][0][1],
                        feature.loop["own"][1][0] : feature.loop["own"][1][1],
                        feature.loop["own"][2][0] : feature.loop["own"][2][1],
                    ]
                )[0]
    return inlet_labels


def gen_outlet_label_map(subdomain, label_grid):
    """
    Determine which face is on outlet.
    Currently restricted to a single face

    Args:
        subdomain (_type_): _description_
        label_grid (_type_): _description_
    """
    outlet_labels = None
    feature_types = ["faces"]  # only faces can be an inlet
    for feature_type in feature_types:
        for feature_id, feature in subdomain.features[feature_type].items():
            if feature.outlet:
                inlet_labels = np.unique(
                    label_grid[
                        feature.loop["own"][0][0] : feature.loop["own"][0][1],
                        feature.loop["own"][1][0] : feature.loop["own"][1][1],
                        feature.loop["own"][2][0] : feature.loop["own"][2][1],
                    ]
                )[0]
    return outlet_labels
