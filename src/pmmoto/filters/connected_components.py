import numpy as np
import cc3d
from pmmoto.core import voxels
from pmmoto.core import sets
from pmmoto.io import output

__all__ = [
    "connect_all_phases",
    "connect_single_phase",
    "get_boundary_label_phase_map",
    "get_label_phase_map",
    "phase_count",
]


def _connect_components(grid):
    """
    Perform a connected components analysis of the given grid
    """
    label_grid, label_count = cc3d.connected_components(
        grid, return_N=True, out_dtype=np.uint64
    )
    label_count += 1

    return label_grid, label_count


def connect_all_phases(
    img, return_grid=False, return_set=False, return_voxel_count=False
):
    """
    Create sets for all phases in grid
    """

    label_grid, label_count = _connect_components(img.grid)

    # Collect phase information
    # phase_map = _get_boundary_label_phase_map(label_grid,label_count,grid,boundary_node_data)
    # phase_map = _get_label_phase_map(img, label_grid)
    phase_count = _phase_count(phase_map)

    print(f"Phase Map {phase_map}")
    print(f"Phase Count {phase_count}")

    all_sets, local_global_map = sets.create_sets_and_merge(
        img, label_count, label_count, label_grid, phase_map, img.inlet, img.outlet
    )

    output = {}

    if return_grid:
        _nodes.renumber_grid(label_grid, local_global_map)

        if return_voxel_count:
            voxel_count = {}
            for n in range(label_count):
                voxel_count[n] = 0

            voxel_count = _nodes.count_label_voxels(label_grid, voxel_count)
            output["voxel_count"] = voxel_count
            output["phase_map"] = phase_map

        output["grid"] = label_grid

    if return_set:
        all_sets.update_global_ID(local_global_map)
        output["sets"] = all_sets

    return output


def connect_single_phase(img, inlet, outlet, phase=None):
    """
    Create sets for all phases in grid
    """

    label_grid, label_count = _connect_components(img.grid)

    phase_map = _get_label_phase_map(img, label_grid, phase)
    phase_count = _phase_count(phase_map)

    all_sets, local_global_map = sets.create_sets_and_merge(
        img, phase_count[phase], label_count, label_grid, phase_map, inlet, outlet
    )

    all_sets.update_global_ID(local_global_map)

    output = {}
    output["sets"] = all_sets

    return output


def get_boundary_label_phase_map(label_grid, label_count, grid, node_data):
    """
    Collect the label to phase mapping
    """
    phase_map = {}
    for label in range(0, label_count):
        if node_data["boundary"][label]:
            phase_map[label] = np.unravel_index(
                node_data["phase"][label], label_grid.shape
            )
            index = np.unravel_index(node_data["phase"][label], label_grid.shape)
            phase_map[label] = grid[index[0], index[1], index[2]]

    return phase_map


def get_label_phase_map(grid, label_grid, phase=None):
    """
    Collect the label to phase mapping for all labels
    TODO: SLOW and probably better way for this
    """

    assert grid.shape == label_grid.shape

    phase_map_all = voxels.get_label_phase_info(
        grid,
        label_grid,
    )
    phase_map = {}
    if phase is not None:
        for label, _phase in phase_map_all.items():
            if _phase == phase:
                phase_map[label] = _phase
    else:
        phase_map = phase_map_all

    return phase_map


def phase_count(phase_map):
    """
    Count the number of labels for a given phase
    """
    phase_count = {}
    for label in phase_map:
        phase = phase_map[label]
        if phase in phase_count:
            phase_count[phase] += 1
        else:
            phase_count[phase] = 1

    return phase_count
