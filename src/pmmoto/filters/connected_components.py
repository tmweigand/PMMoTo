import numpy as np
import cc3d
from pmmoto.core import voxels
from pmmoto.core import sets
from pmmoto.core import communication


__all__ = [
    "connect_components",
    "gen_grid_to_label_map",
    "gen_inlet_label_map",
    "gen_outlet_label_map",
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


def connect_components(
    grid,
    subdomain,
):
    """
    Create sets for all phases in grid
    """

    label_grid, label_count = _connect_components(grid)

    if subdomain.domain.num_subdomains > 1:
        connect_subdomain_boundaries(subdomain, label_grid, label_count)

    return label_grid


def connect_subdomain_boundaries(subdomain, label_grid, label_count):
    """_summary_

    Args:
        subdomain (_type_): _description_
        label_grid (_type_): _description_
        label_count (_type_): _description_
    """
    data = voxels.get_boundary_voxels(
        subdomain=subdomain,
        img=label_grid,
    )

    send_data, own_data = voxels.boundary_voxels_pack(subdomain, data)

    if own_data or send_data:
        if send_data:
            recv_data = communication.communicate(subdomain, send_data, unpack=True)
            own_data.update(recv_data)

        matches = voxels.match_neighbor_boundary_voxels(subdomain, data, own_data)

        local_global_map = voxels.match_global_boundary_voxels(
            subdomain, matches, label_count
        )

        voxels.renumber_image(label_grid, local_global_map)


def gen_grid_to_label_map(grid, label_grid):
    """
    Collect the label to phase mapping for all labels
    """
    return voxels.gen_grid_to_label_map(
        grid,
        label_grid,
    )


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


def gen_inlet_label_map(subdomain, label_grid):
    """_summary_

    Args:
        subdomain (_type_): _description_
        lables (_type_): _description_
    """
    return voxels.gen_inlet_label_map(subdomain, label_grid)


def gen_outlet_label_map(subdomain, label_grid):
    """_summary_

    Args:
        subdomain (_type_): _description_
        lables (_type_): _description_
    """
    return voxels.gen_outlet_label_map(subdomain, label_grid)
