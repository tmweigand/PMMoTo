"""test_subdomain_features.py"""

import numpy as np
import pmmoto


def test_set_opposite_feature():
    """
    Test subdomain features
    """

    subdomains = [1, 1, 1]
    # subdomains = [5, 5, 5]

    voxels = (100, 100, 100)

    box = [[0, 10], [0, 10], [0, 10]]
    # boundary_types = [[0, 0], [0, 0], [0, 0]]
    boundary_types = [[2, 2], [2, 2], [2, 2]]
    inlet = [[0, 0], [0, 0], [0, 0]]
    outlet = [[0, 0], [0, 0], [0, 0]]

    save_data = True

    sd = {}
    domain = {}
    grid = {}

    size = np.prod(subdomains)

    for rank in range(size):
        sd[rank], domain[rank] = pmmoto.initialize(
            box=box,
            subdomains=subdomains,
            voxels=voxels,
            boundary_types=boundary_types,
            inlet=inlet,
            outlet=outlet,
            rank=rank,
            mpi_size=size,
            reservoir_voxels=0,
        )


# def test_feature_loop():
#     """
#     Test subdomain features
#     """

#     subdomains = (1, 1, 1)
#     # subdomains = (2, 2, 2)

#     voxels = (50, 50, 50)

#     box = [[0, 10], [0, 10], [0, 10]]
#     # boundary_types = [[0, 0], [0, 0], [0, 0]]
#     # boundary_types = [[2, 2], [0, 0], [0, 0]]
#     boundary_types = [[2, 2], [2, 2], [2, 2]]
#     inlet = [[0, 0], [0, 0], [0, 0]]
#     outlet = [[0, 0], [0, 0], [0, 0]]

#     save_data = True

#     sd = {}
#     domain = {}
#     grid = {}

#     size = np.prod(subdomains)

#     for rank in range(size):
#         sd[rank], domain[rank] = pmmoto.initialize(
#             box=box,
#             subdomains=subdomains,
#             voxels=voxels,
#             boundary_types=boundary_types,
#             inlet=inlet,
#             outlet=outlet,
#             rank=rank,
#             mpi_size=size,
#             reservoir_voxels=0,
#             pad=(3, 3, 3),
#         )
#         grid[rank] = np.zeros(sd[rank].voxels, dtype=int)
#         feature_types = [
#             "faces",
#         ]
#         for feature_type in feature_types:
#             for feature_id, feature in sd[rank].features[feature_type].items():

#                 _shape = grid[rank].shape
#                 loop_both = pmmoto.core.subdomain_features.get_feature_voxels(
#                     feature_id,
#                     _shape,
#                     sd[rank].pad,
#                 )

#                 grid[rank][
#                     loop_both["own"][0][0] : loop_both["own"][0][1],
#                     loop_both["own"][1][0] : loop_both["own"][1][1],
#                     loop_both["own"][2][0] : loop_both["own"][2][1],
#                 ] = 1

#                 grid[rank][
#                     loop_both["neighbor"][0][0] : loop_both["neighbor"][0][1],
#                     loop_both["neighbor"][1][0] : loop_both["neighbor"][1][1],
#                     loop_both["neighbor"][2][0] : loop_both["neighbor"][2][1],
#                 ] = 2

#     if save_data:
#         pmmoto.io.save_grid_data("data_out/test_output", sd, grid)
