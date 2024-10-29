"""voxel.py"""

import numpy as np
from . import _voxels


__all__ = [
    "get_id",
    "get_boundary_voxels",
    "get_label_phase_info",
    "count_label_voxels",
    "match_boundary_voxels",
]


def get_boundary_voxels(subdomain, img, n_labels):
    """_summary_

    Args:
        img (_type_): _description_
        label_grid (_type_): _description_
        n_labels (_type_): _description_
        phase_map (_type_): _description_
        inlet (_type_): _description_
        outlet (_type_): _description_
    """
    boundary_data = {}
    feature_types = ["faces", "edges", "corners"]
    for feature_type in feature_types:
        for feature_id, feature in subdomain.features[feature_type].items():
            boundary_data[feature_id] = _voxels.get_boundary_data(
                img,
                n_labels,
                feature.loop,
                subdomain.domain_voxels,
                subdomain.start,
            )

            # Sort boundary_voxels to make matching more efficient for large number of labels
            for voxels in boundary_data[feature_id]["boundary_voxels"].values():
                voxels.sort()

    return boundary_data


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
    send_data = {}

    feature_types = ["faces", "edges", "corners"]
    for feature_type in feature_types:
        for feature_id, feature in subdomain.features[feature_type].items():
            if feature.n_proc > -1 and feature.n_proc != subdomain.rank:
                send_data[feature.n_proc] = boundary_voxels[feature_id]

    return send_data


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


def match_boundary_voxels(subdomain, boundary_voxels, recv_data):
    """
    Match the boundary voxels
    """
    feature_types = ["faces", "edges", "corners"]
    feature_types = ["edges", "corners"]
    for feature_type in feature_types:
        for feature_id, feature in subdomain.features[feature_type].items():
            if feature_id in recv_data:
                print(
                    feature_id,
                    feature.opp_info,
                    boundary_voxels[feature_id],
                    recv_data[feature_id],
                )
                _voxels.match_boundary_voxels(
                    boundary_voxels[feature_id],
                    recv_data[feature_id],
                )
