"""Connected components labeling and inlet/outlet connectivity utilities for PMMoTo.

Provides functions for labeling connected regions, mapping labels to phases,
and extracting inlet/outlet-connected regions in distributed or periodic domains.
"""

import numpy as np
import cc3d
from collections import defaultdict
from pmmoto.core import voxels
from pmmoto.core import communication

__all__ = [
    "connect_components",
    "gen_img_to_label_map",
    "phase_count",
    "inlet_connected_img",
]


def connect_components(img, subdomain, return_label_count=True):
    """Label connected components (sets) for all phases in img.

    Note:
        Zero is background and will not join to any other voxel.

    Args:
        img (np.ndarray): Input binary image.
        subdomain: Subdomain object.
        return_label_count (bool, optional): If True, return (label_img, label_count).

    Returns:
        tuple or np.ndarray: (label_img, label_count) or label_img.

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
    """Connect labels across subdomain boundaries for distributed/periodic domains.

    Args:
        subdomain: Subdomain object.
        label_grid (np.ndarray): Labeled image.
        label_count (int): Number of labels.

    Returns:
        tuple: (label_grid, global_label_count)

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
    """Generate a mapping from label to phase for all labels.

    Args:
        img (np.ndarray): Input image.
        labeled_img (np.ndarray): Labeled image.

    Returns:
        dict: Mapping from label to phase.

    """
    return voxels.gen_grid_to_label_map(
        img,
        labeled_img,
    )


def phase_count(phase_map):
    """Count the number of labels for each phase.

    Args:
        phase_map (dict): Mapping from label to phase.

    Returns:
        dict: Mapping from phase to count.

    """
    phase_count = {}
    for label in phase_map:
        phase = phase_map[label]
        if phase in phase_count:
            phase_count[phase] += 1
        else:
            phase_count[phase] = 1

    return phase_count


def inlet_outlet_labels(subdomain, labeled_img):
    """Collect the labels that are on the inlet and outlet.

    Args:
        subdomain: Subdomain object.
        labeled_img (np.ndarray): Labeled image.

    Returns:
        dict: Mapping from label to {"inlet": bool, "outlet": bool}.

    """
    sd_inlet = voxels.gen_inlet_label_map(subdomain, labeled_img)
    sd_outlet = voxels.gen_outlet_label_map(subdomain, labeled_img)

    inlet_outlet = communication.all_gather({"inlet": sd_inlet, "outlet": sd_outlet})

    connected = defaultdict(lambda: {"inlet": False, "outlet": False})
    for rank_data in inlet_outlet:
        for label in rank_data["inlet"]:
            if label > 0:
                connected[label]["inlet"] = True
        for label in rank_data["outlet"]:
            if label > 0:
                connected[label]["outlet"] = True

    return connected


def inlet_outlet_connections(subdomain, labeled_img):
    """Determine the labels that are connected to both the inlet and outlet.

    Args:
        subdomain: Subdomain object.
        labeled_img (np.ndarray): Labeled image.

    Returns:
        list: List of label IDs connected to both inlet and outlet.

    """
    connected = inlet_outlet_labels(subdomain, labeled_img)
    connections = []
    for label_id, label in connected.items():
        if label["inlet"] and label["outlet"]:
            connections.append(label_id)

    return connections


def inlet_connected_img(subdomain, img, phase=None):
    """Return an image where all voxels are connected to the inlet.

    If phase is specified, all other phases are set to zero, leaving only connected
    voxels of the specified phase.

    Args:
        subdomain: Subdomain object.
        img (np.ndarray): Input image.
        phase (optional): Phase to filter for.

    Returns:
        np.ndarray: Image with only inlet-connected voxels.

    """
    # Generate labels for each phase
    labeled_img = connect_components(img, subdomain, return_label_count=False)

    # Collect the inlet and outlet labels
    inlet_labels = inlet_outlet_labels(subdomain, labeled_img)

    # Collect the img-phase to label mapping
    inlet_phase_map = gen_img_to_label_map(img, labeled_img)

    # Modify inlet_phase_map where the phase is set to zero
    # if not connected to the outlet
    if phase is None:
        for label in inlet_phase_map.keys():
            if inlet_labels[label]["inlet"] is False:
                inlet_phase_map[label] = 0
    else:
        for label, _phase in inlet_phase_map.items():
            if _phase != phase or inlet_labels[label]["inlet"] is False:
                inlet_phase_map[label] = 0

    # Renumber the label map with the outlet_phase_map
    inlet_img = voxels.renumber_image(labeled_img, inlet_phase_map)

    return inlet_img


def outlet_connected_img(subdomain, img, phase=None):
    """Return an image where all voxels are connected to the outlet.

    If phase is specified, all other phases are set to zero, leaving only connected
    voxels of the specified phase.

    Args:
        subdomain: Subdomain object.
        img (np.ndarray): Input image.
        phase (optional): Phase to filter for.

    Returns:
        np.ndarray: Image with only outlet-connected voxels.

    """
    # Generate labels for each phase
    labeled_img = connect_components(img, subdomain, return_label_count=False)

    # Collect the inlet and outlet labels
    outlet_labels = inlet_outlet_labels(subdomain, labeled_img)

    # Collect the img-phase to label mapping
    outlet_phase_map = gen_img_to_label_map(img, labeled_img)

    # Modify outlet_phase_map where the phase is set to zero
    # if not connected to the outlet
    if phase is None:
        for label in outlet_phase_map.keys():
            if outlet_labels[label]["outlet"] is False:
                outlet_phase_map[label] = 0
    else:
        for label, _phase in outlet_phase_map.items():
            if _phase != phase or outlet_labels[label]["outlet"] is False:
                outlet_phase_map[label] = 0

    # Renumber the label map with the outlet_phase_map
    outlet_img = voxels.renumber_image(labeled_img, outlet_phase_map)

    return outlet_img
