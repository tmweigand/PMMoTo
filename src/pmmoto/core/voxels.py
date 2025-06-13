"""voxel.py

Core voxel operations for PMMoTo.
"""

from __future__ import annotations
from typing import TypeVar, Literal, Any
import numpy as np
from numpy.typing import NDArray
from . import _voxels
from . import communication
from .logging import get_logger, USE_LOGGING
from .subdomain import Subdomain
from .subdomain_padded import PaddedSubdomain
from .subdomain_verlet import VerletSubdomain

if USE_LOGGING:
    logger = get_logger()

T = TypeVar("T", bound=np.generic)
INT = TypeVar("INT", np.integer[Any], np.unsignedinteger[Any])
INT2 = TypeVar("INT2", np.integer[Any], np.unsignedinteger[Any])

__all__ = [
    "renumber_image",
    "get_nearest_boundary_index",
    "get_id",
    "get_boundary_voxels",
    "gen_img_to_label_map",
    "gen_inlet_label_map",
    "gen_outlet_label_map",
    "count_label_voxels",
    "match_neighbor_boundary_voxels",
    "match_global_boundary_voxels",
]


def renumber_image(img: NDArray[INT2], conversion_map: dict[INT2, INT]) -> NDArray[INT]:
    """Renumber an image using a provided mapping.

    Args:
        img (Any): The image to be renumbered.
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
    _img: NDArray[INT] = _voxels.renumber_img(img, conversion_map)

    return np.ascontiguousarray(_img)


def get_nearest_boundary_index(
    subdomain: Subdomain | PaddedSubdomain | VerletSubdomain,
    img: NDArray[T],
    label: int,
    dimension: None | int = None,
    which_voxels: Literal["all", "own", "pad"] = "all",
) -> dict[tuple[int, ...], NDArray[T]]:
    """Determine the nearest boundary index to a given label

    If which_voxels == "all", always use base version.
    If which_voxels is "own"/ "pad", use extended version.
    """
    if dimension is not None and dimension not in {0, 1, 2}:
        raise ValueError("`dimension` must be an integer (0, 1, or 2) or None.")

    if which_voxels == "all":
        return get_nearest_boundary_index_base(
            subdomain=subdomain,
            img=img,
            label=label,
            dimension=dimension,
        )
    else:
        if not isinstance(subdomain, (PaddedSubdomain, VerletSubdomain)):
            raise TypeError(
                f"which_voxels={which_voxels!r} requires a PaddedSubdomain or "
                f"VerletSubdomain, but got {type(subdomain).__name__}."
            )

        return get_nearest_boundary_index_extended(
            subdomain=subdomain,
            img=img,
            label=label,
            dimension=dimension,
            which_voxels=which_voxels,
        )


def get_nearest_boundary_index_base(
    subdomain: Subdomain,
    img: NDArray[T],
    label: int,
    dimension: None | int = None,
) -> dict[tuple[int, ...], NDArray[T]]:
    """Get nearest boundary index without considering padding."""
    boundary_index: dict[tuple[int, ...], NDArray[T]] = {}

    for feature_id, feature in subdomain.features.faces.items():
        if dimension is None or feature_id[dimension] != 0:
            boundary_index[feature_id] = _voxels.get_nearest_boundary_index_face(
                img=img,
                dimension=feature.info.arg_order[0],
                label=label,
                forward=feature.forward,
                lower_skip=0,
                upper_skip=0,
            )

    return boundary_index


def get_nearest_boundary_index_extended(
    subdomain: PaddedSubdomain | VerletSubdomain,
    img: NDArray[T],
    label: int,
    dimension: None | int = None,
    which_voxels: Literal["all", "own", "pad"] = "all",
) -> dict[tuple[int, ...], NDArray[T]]:
    """Get nearest boundary index with padding support."""
    if dimension is not None and dimension not in {0, 1, 2}:
        raise ValueError("`dimension` must be an integer (0, 1, or 2) or None.")

    boundary_index: dict[tuple[int, ...], NDArray[T]] = {}

    for feature_id, feature in subdomain.features.faces.items():
        if dimension is None or feature_id[dimension] != 0:
            lower_skip = 0
            upper_skip = 0

            arg_dim = feature.info.arg_order[0]

            if feature.forward:
                if which_voxels == "own":
                    lower_skip = subdomain.pad[arg_dim][0]
                elif which_voxels == "pad":
                    lower_skip = 2 * subdomain.pad[arg_dim][0]
            else:
                if which_voxels == "own":
                    upper_skip = subdomain.pad[arg_dim][1]
                elif which_voxels == "pad":
                    upper_skip = 2 * subdomain.pad[arg_dim][1]

            boundary_index[feature_id] = _voxels.get_nearest_boundary_index_face(
                img=img,
                dimension=arg_dim,
                label=label,
                forward=feature.forward,
                lower_skip=lower_skip,
                upper_skip=upper_skip,
            )

    return boundary_index


def get_boundary_voxels(
    subdomain: Subdomain | PaddedSubdomain | VerletSubdomain,
    img: NDArray[INT],
    neighbors_only: bool = False,
) -> dict[tuple[int, ...], dict[str, NDArray[INT]]]:
    """Return the values on the boundary features.

    The features are divided into:
    - own: feature voxels owned by subdomain
    - neighbor: feature voxels owned by a neighbor subdomain

    Args:
        subdomain: Subdomain object.
        img: Image array.
        neighbors_only (bool): If True, only include neighbor voxels.

    Returns:
        dict: Dictionary of boundary voxels.

    """
    out_voxels: dict[tuple[int, ...], dict[str, NDArray[INT]]] = {}

    types = ["own", "neighbor"]
    for feature_id, feature in subdomain.features.all_features:
        if neighbors_only and feature.neighbor_rank < -1:
            continue
        out_voxels[feature_id] = {}
        for _type in types:
            loop = getattr(feature, _type)
            out_voxels[feature_id][_type] = img[
                loop[0][0] : loop[0][1],
                loop[1][0] : loop[1][1],
                loop[2][0] : loop[2][1],
            ].flatten()

    return out_voxels


def get_id(index: tuple[int, ...], total_voxels: tuple[int, ...]) -> np.uint64:
    """Get the global or local ID for a voxel.

    Args:
        index: 3D index of the voxel (x, y, z).
        total_voxels: Number of voxels in each dimension.

    Returns:
        int: Global or local ID of the voxel.

    Note:
        Periodic boundary conditions are applied by using modulo arithmetic.

    """
    id = _voxels.get_id(index, total_voxels)
    return id


def gen_img_to_label_map(img: NDArray[INT], labels: NDArray[INT2]) -> dict[INT2, INT]:
    """Generate a mapping from grid indices to labels.

    Args:
        img: Grid array.
        labels: Label array.

    Returns:
        Mapping array.

    """
    assert img.shape == labels.shape

    return _voxels.gen_img_to_label_map(img, labels)


def count_label_voxels(img: NDArray[T], map: dict[int, int]) -> dict[int, int]:
    """Count the number of voxels for each label in the grid.

    Args:
        img: Grid array.
        map: Mapping array.

    Returns:
        None

    """
    _map = _voxels.count_label_voxels(img, map)

    return _map


def match_neighbor_boundary_voxels(
    subdomain: Subdomain | PaddedSubdomain | VerletSubdomain,
    boundary_voxels: dict[tuple[int, ...], dict[str, NDArray[INT]]],
    recv_data: dict[tuple[int, ...], dict[str, NDArray[INT]]],
    skip_zero: bool = False,
) -> dict[tuple[int, INT], dict[str, tuple[int, INT]]]:
    """Match boundary voxels with subdomain neighbor voxels and return unique matches.

    Args:
        subdomain: Subdomain object containing feature information.
        boundary_voxels: Dictionary with 'own' and 'neighbor' boundary voxel data.
        recv_data: Received data containing neighbor voxel information.
        skip_zero (bool): If True, skip matches with label 0.

    Returns:
        dict: Unique matches in the format:
            key: (subdomain rank, own voxel)
            neighbor: list[neighbor rank, neighbor voxel)]

    """
    unique_matches: dict[tuple[int, INT], dict[str, tuple[int, INT]]] = {}

    for feature_id, feature in subdomain.features.all_features:
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

            matches: NDArray[INT] = _voxels.find_unique_pairs(to_match)

            if skip_zero:
                matches = matches[~np.any(matches == 0, axis=1)]

            unique_matches = _voxels.process_matches_by_feature(
                matches, unique_matches, subdomain.rank, feature.neighbor_rank
            )

    return unique_matches


def match_global_boundary_voxels(
    matches: dict[tuple[int, INT], dict[str, tuple[int, INT]]],
    label_count: int,
) -> tuple[dict[int, dict[int, INT]], int]:
    """Generate a global label map for matched boundary voxels.

    Args:
        subdomain: Subdomain object.
        matches: Matches dictionary.
        label_count: Number of labels on this rank.

    Returns:
        tuple: (final_map, global_label_count)

    """
    ### Send number of labels on rank for re-labeling
    all_matches = communication.all_gather(matches)
    all_counts = communication.all_gather(label_count)

    ### Generate the local-global label map
    # boundary_label_count is the global id + 1 from the merged labels.
    # Local labels (i.e. not on subdomain boundaries) to start counting from here.
    local_global_map, boundary_label_count = _voxels.merge_matched_voxels(all_matches)

    final_map, global_label_count = local_to_global_labeling(
        all_matches,
        all_counts,
        local_global_map,
        boundary_label_count,
    )

    return final_map, global_label_count


def local_to_global_labeling(
    all_matches: list[dict[int, dict[tuple[int, INT], dict[str, tuple[int, INT]]]]],
    all_counts: list[int],
    boundary_map: dict[tuple[int, INT], dict[str, INT]],
    boundary_label_count: int,
) -> tuple[dict[int, dict[int, INT]], int]:
    """Generate the local to global label mapping for all ranks.

    Args:
        all_matches: List of matches from all ranks.
        all_counts: List of label counts from all subdomains
        boundary_map: Mapping of boundary labels.
        boundary_label_count: Number of boundary labels.
        own (int, optional): If set, return only for this rank.

    Returns:
        tuple or dict: (final_map, max_label) or (final_map[own], max_label)

    """
    boundary_counts = [len(matches) for matches in all_matches]

    num_ranks = len(all_counts)

    # Determine unique global labels relative so expect negative!
    local_starts = {}
    for rank in range(num_ranks):
        if rank == 0:
            local_starts[rank] = boundary_label_count
        else:
            local_starts[rank] = (
                local_starts[rank - 1]
                + all_counts[rank - 1]
                - boundary_counts[rank - 1]
            )

    ### Generate the global id for non-boundary labels as well
    final_map: dict[int, dict[int, INT]] = {}
    for rank, (match, _label_count) in enumerate(zip(all_matches, all_counts)):
        final_map[rank] = {0: np.uint64(0)}
        count = 0
        for n in range(1, _label_count + 1):
            n64 = np.uint64(n)
            if (rank, n64) in boundary_map:
                final_map[rank][n] = boundary_map[(rank, n64)]["global_id"]
            else:
                final_map[rank][n] = np.uint64(local_starts[rank] + count)
                count += 1
        label_count = count

    max_label = label_count + local_starts[num_ranks - 1]
    if max_label < 1:
        max_label = boundary_label_count

    return final_map, max_label


def gen_inlet_label_map(
    subdomain: Subdomain | PaddedSubdomain | VerletSubdomain, label_img: NDArray[T]
) -> NDArray[T]:
    """Determine which face is on inlet.

    Currently restricted to a single face.

    Args:
        subdomain: Subdomain object.
        label_img: Labeled image.

    Returns:
        np.ndarray: Array of inlet labels.

    """
    inlet_labels = np.empty(0, dtype=int)
    for _, feature in subdomain.features.faces.items():
        if feature.inlet:
            inlet_labels = np.unique(
                label_img[
                    feature.own[0][0] : feature.own[0][1],
                    feature.own[1][0] : feature.own[1][1],
                    feature.own[2][0] : feature.own[2][1],
                ]
            )
    return inlet_labels


def gen_outlet_label_map(
    subdomain: Subdomain | PaddedSubdomain | VerletSubdomain, label_img: NDArray[T]
) -> NDArray[T]:
    """Determine which face is on outlet.

    Currently restricted to a single face.

    Args:
        subdomain: Subdomain object.
        label_img: Labeled image.

    Returns:
        np.ndarray: Array of outlet labels.

    """
    outlet_labels: NDArray[T] = np.empty(0, dtype=int)
    for _, feature in subdomain.features.faces.items():
        if feature.outlet:
            outlet_labels = np.unique(
                label_img[
                    feature.own[0][0] : feature.own[0][1],
                    feature.own[1][0] : feature.own[1][1],
                    feature.own[2][0] : feature.own[2][1],
                ]
            )

    return outlet_labels
