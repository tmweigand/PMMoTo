"""voxel.py"""

import numpy as np
from . import _voxels
from . import communication


__all__ = [
    "get_id",
    "get_boundary_voxels",
    "get_label_phase_info",
    "count_label_voxels",
    "match_neighbor_boundary_voxels",
    "match_global_boundary_voxels",
]


def get_boundary_voxels(subdomain, img):
    """_summary_

    Args:
        subdomain (_type_): _description_
        img (_type_): _description_
    """
    out_voxels = {"own": {}, "neighbor": {}}

    feature_types = ["faces", "edges", "corners"]
    for feature_type in feature_types:
        for feature_id, feature in subdomain.features[feature_type].items():
            for kind, out in out_voxels.items():
                out[feature_id] = img[
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


def get_label_phase_info(grid, label_grid):
    """_summary_

    Args:
        grid (_type_): _description_
        label_grid (_type_): _description_
    """
    phase_label = _voxels.get_label_phase_info(
        grid.astype(np.uint8), label_grid.astype(np.uint64)
    )

    return phase_label


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
    send_data = {"own": {}, "neighbor": {}}
    periodic_data = {"own": {}, "neighbor": {}}

    feature_types = ["faces", "edges", "corners"]
    for feature_type in feature_types:
        for feature_id, feature in subdomain.features[feature_type].items():
            if feature.n_proc > -1 and feature.n_proc != subdomain.rank:
                send_data["own"][feature_id] = {
                    "rank": subdomain.rank,
                    "data": boundary_voxels["own"][feature_id],
                }
                send_data["neighbor"][feature_id] = {
                    "rank": subdomain.rank,
                    "data": boundary_voxels["neighbor"][feature_id],
                }
            if feature.n_proc == subdomain.rank:
                periodic_data["own"][feature_id] = {
                    "rank": subdomain.rank,
                    "data": boundary_voxels["own"][feature_id],
                }
                periodic_data["neighbor"][feature_id] = {
                    "rank": subdomain.rank,
                    "data": boundary_voxels["neighbor"][feature_id],
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
        list: Unique matches in the format
            (subdomain rank, own voxel, neighbor rank, neighbor voxel).
    """
    unique_matches = []

    feature_types = ["faces", "edges", "corners"]
    for feature_type in feature_types:
        for feature_id, feature in subdomain.features[feature_type].items():
            if feature.opp_info in recv_data["neighbor"]:
                to_match = np.stack(
                    [
                        np.concatenate(
                            (
                                boundary_voxels["own"][feature_id],
                                boundary_voxels["neighbor"][feature_id],
                            )
                        ),
                        np.concatenate(
                            (
                                recv_data["neighbor"][feature.opp_info]["data"],
                                recv_data["own"][feature.opp_info]["data"],
                            )
                        ),
                    ],
                    axis=1,
                )

                for match in np.unique(to_match, axis=0):
                    match_tuple = (
                        subdomain.rank,
                        match[0],
                        recv_data["own"][feature.opp_info]["rank"],
                        match[1],
                    )
                    if match_tuple not in unique_matches:
                        unique_matches.append(match_tuple)

    return unique_matches


def match_global_boundary_voxels(subdomain, matches):
    """_summary_

    Args:
        subdomain (_type_): _description_
        matches (_type_): _description_
    """
    all_matches = communication.gather(matches)
