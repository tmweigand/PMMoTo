"""voxel.py"""

import numpy as np
from . import _voxels


__all__ = ["get_boundary_set_info", "get_label_phase_info", "count_label_voxels"]


def get_boundary_set_info(img, label_grid, n_labels, phase_map, inlet, outlet):
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
        img.subdomain, label_grid, n_labels, phase_map, img.loop_info, inlet, outlet
    )

    return boundary_node_data


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
