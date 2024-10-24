"""voxel.py"""

import numpy as np
from . import _voxels


__all__ = [
    "get_id",
    "get_boundary_set_info_NEW",
    "get_boundary_set_info",
    "get_label_phase_info",
    "count_label_voxels",
]


def get_boundary_set_info_NEW(subdomain, img, n_labels):
    """_summary_

    Args:
        img (_type_): _description_
        label_grid (_type_): _description_
        n_labels (_type_): _description_
        phase_map (_type_): _description_
        inlet (_type_): _description_
        outlet (_type_): _description_
    """
    boundary_node_data = {}
    for f in subdomain.features["faces"].values():
        boundary_node_data[f.info["ID"]] = _voxels.get_boundary_set_info_NEW(
            img,
            n_labels,
            f.get_feature_voxels(img),
            subdomain.domain_voxels,
            subdomain.start,
        )

    for f in subdomain.features["edges"].values():
        boundary_node_data[f.info["ID"]] = _voxels.get_boundary_set_info_NEW(
            img,
            n_labels,
            f.get_feature_voxels(img),
            subdomain.domain_voxels,
            subdomain.start,
        )

    for f in subdomain.features["corners"].values():
        boundary_node_data[f.info["ID"]] = _voxels.get_boundary_set_info_NEW(
            img,
            n_labels,
            f.get_feature_voxels(img),
            subdomain.domain_voxels,
            subdomain.start,
        )

    print(boundary_node_data)

    return boundary_node_data


def get_boundary_set_info(img, label_grid, n_labels):
    """_summary_

    Args:
        img (_type_): _description_
        label_grid (_type_): _description_
        n_labels (_type_): _description_
        phase_map (_type_): _description_
        inlet (_type_): _description_
        outlet (_type_): _description_
    """
    boundary_node_data = _voxels.get_boundary_set_info(
        img.subdomain, label_grid, n_labels, img.loop_info
    )

    print(boundary_node_data)

    return boundary_node_data


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
