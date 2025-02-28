import numpy as np
import cc3d
from collections import defaultdict
from pmmoto.core import voxels
from pmmoto.core import communication

__all__ = [
    "connect_components",
    "gen_img_to_label_map",
    "gen_inlet_label_map",
    "gen_outlet_label_map",
    "phase_count",
    "inlet_connected_img",
]


def connect_components(img, subdomain, return_label_count=True):
    """
    Create sets for all phases in img.

    Note: Zero is background and will not join to any other voxel!
    """

    # max_label = label_count
    label_img, label_count = cc3d.connected_components(
        img, return_N=True, out_dtype=np.uint64
    )

    if subdomain.domain.periodic or subdomain.domain.num_subdomains > 1:
        label_img, label_count = connect_subdomain_boundaries(
            subdomain, label_img, label_count
        )

    if return_label_count:
        return label_img, label_count
    else:
        return label_img


def connect_subdomain_boundaries(subdomain, label_grid, label_count):
    """_summary_

    Args:
        subdomain (_type_): _description_
        label_grid (_type_): _description_
        label_count (_type_): _description_
    """
    boundary_labels = voxels.get_boundary_voxels(
        subdomain=subdomain, img=label_grid, neighbors_only=True
    )

    recv_data = communication.communicate_features(
        subdomain=subdomain, send_data=boundary_labels
    )

    matches = voxels.match_neighbor_boundary_voxels(
        subdomain, boundary_labels, recv_data, skip_zero=True
    )

    local_global_map, global_label_count = voxels.match_global_boundary_voxels(
        subdomain, matches, label_count
    )

    label_grid = voxels.renumber_image(label_grid, local_global_map)

    return label_grid, global_label_count


def gen_img_to_label_map(img, labeled_img):
    """
    Collect the label to phase mapping for all labels
    """
    return voxels.gen_grid_to_label_map(
        img,
        labeled_img,
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


def gen_inlet_label_map(subdomain, labeled_img):
    """
    Determine the labels that are on the inlet
    """
    return voxels.gen_inlet_label_map(subdomain, labeled_img)


def gen_outlet_label_map(subdomain, labeled_img):
    """
    Determine the labels that are on the outlet
    """
    return voxels.gen_outlet_label_map(subdomain, labeled_img)


def inlet_outlet_labels(subdomain, labeled_img):
    """
    Collect the labels that are on the inlet and outlet!
    """
    inlet = gen_inlet_label_map(subdomain, labeled_img)
    outlet = gen_outlet_label_map(subdomain, labeled_img)

    global_data = communication.all_gather({"inlet": inlet, "outlet": outlet})

    connected = defaultdict(lambda: {"inlet": False, "outlet": False})
    for rank_data in global_data:
        for label in rank_data["inlet"]:
            if label > 0:
                connected[label]["inlet"] = True
        for label in rank_data["outlet"]:
            if label > 0:
                connected[label]["outlet"] = True

    connections = []
    for label_id, label in connected.items():
        if label["inlet"] and label["outlet"]:
            connections.append(label_id)

    return connections


def inlet_connected_img(subdomain, img):
    """
    This function return an img where all voxels are connected to the inlet
    """
    labeled_img = connect_components(img, subdomain, return_label_count=False)
    inlet_labels = gen_inlet_label_map(subdomain, labeled_img)

    inlet_img = np.zeros_like(img)
    if inlet_labels.shape[0] == 0:
        return inlet_img
    else:
        inlet_phase_map = gen_img_to_label_map(img, labeled_img)
        for label in inlet_phase_map.keys():
            if label not in inlet_labels:
                inlet_phase_map[label] = 0

        inlet_img = voxels.renumber_image(labeled_img, inlet_phase_map)

    return inlet_img
