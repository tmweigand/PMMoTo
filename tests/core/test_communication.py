import numpy as np
from mpi4py import MPI
import pmmoto


def test_update_buffer():

    solution = np.array(
        [
            [
                [26, 24, 25, 26, 24],
                [20, 18, 19, 20, 18],
                [23, 21, 22, 23, 21],
                [26, 24, 25, 26, 24],
                [20, 18, 19, 20, 18],
            ],
            [
                [8, 6, 7, 8, 6],
                [2, 0, 1, 2, 0],
                [5, 3, 4, 5, 3],
                [8, 6, 7, 8, 6],
                [2, 0, 1, 2, 0],
            ],
            [
                [17, 15, 16, 17, 15],
                [11, 9, 10, 11, 9],
                [14, 12, 13, 14, 12],
                [17, 15, 16, 17, 15],
                [11, 9, 10, 11, 9],
            ],
            [
                [26, 24, 25, 26, 24],
                [20, 18, 19, 20, 18],
                [23, 21, 22, 23, 21],
                [26, 24, 25, 26, 24],
                [20, 18, 19, 20, 18],
            ],
            [
                [8, 6, 7, 8, 6],
                [2, 0, 1, 2, 0],
                [5, 3, 4, 5, 3],
                [8, 6, 7, 8, 6],
                [2, 0, 1, 2, 0],
            ],
        ],
        dtype=int,
    )

    subdomains = (1, 1, 1)
    voxels = [3, 3, 3]
    box = [[0, 1], [0, 1], [0, 1]]
    boundary_types = [[2, 2], [2, 2], [2, 2]]
    inlet = [[0, 0], [0, 0], [0, 0]]
    outlet = [[0, 0], [0, 0], [0, 0]]

    sd = pmmoto.initialize(
        box=box,
        subdomains=subdomains,
        voxels=voxels,
        boundary_types=boundary_types,
        inlet=inlet,
        outlet=outlet,
        rank=0,
        mpi_size=1,
        reservoir_voxels=0,
    )

    grid = np.zeros(sd.voxels)
    own_nodes = [sd.voxels[0] - 2, sd.voxels[1] - 2, sd.voxels[2] - 2]
    grid[1:-1, 1:-1, 1:-1] = np.arange(
        own_nodes[0] * own_nodes[1] * own_nodes[2]
    ).reshape(own_nodes)

    updated_grid = pmmoto.core.communication.update_buffer(sd, grid)

    print(updated_grid)

    np.testing.assert_array_almost_equal(updated_grid, solution)


def _create_subdomain(rank, periodic=True):
    box = ((0, 1.0), (0, 1.0), (0, 1.0))
    if periodic:
        boundary_types = ((2, 2), (2, 2), (2, 2))
    else:
        boundary_types = ((0, 0), (0, 0), (0, 0))
    inlet = ((1, 0), (0, 0), (0, 0))
    outlet = ((0, 1), (0, 0), (0, 0))
    voxels = (100, 100, 100)
    subdomains = (1, 1, 1)
    pad = (1, 1, 1)
    reservoir_voxels = 0

    pmmoto_domain = pmmoto.core.domain.Domain(
        box=box, boundary_types=boundary_types, inlet=inlet, outlet=outlet
    )

    pmmoto_discretized_domain = (
        pmmoto.core.domain_discretization.DiscretizedDomain.from_domain(
            domain=pmmoto_domain, voxels=voxels
        )
    )

    pmmoto_decomposed_domain = (
        pmmoto.core.domain_decompose.DecomposedDomain.from_discretized_domain(
            discretized_domain=pmmoto_discretized_domain,
            subdomains=subdomains,
        )
    )

    padded_subdomain = pmmoto.core.subdomain_padded.PaddedSubdomain(
        rank=rank,
        decomposed_domain=pmmoto_decomposed_domain,
        pad=pad,
        reservoir_voxels=reservoir_voxels,
    )
    return padded_subdomain


# @pytest.mark.mpi(min_size=8)
# def test_communicate_features():
#     """
#     Ensure that features are being communicated to neighbor processes
#     """

#     comm = MPI.COMM_WORLD
#     rank = comm.Get_rank()
#     periodic = True
#     sd = _create_subdomain(0, periodic=periodic)
#     img = rank * np.ones(sd.voxels)

#     subdomains = (2, 2, 2)
#     sd_local, local_img = pmmoto.core.pmmoto.deconstruct_grid(
#         sd, img, subdomains=subdomains, rank=rank, periodic=periodic
#     )

#     feature_data = {}
#     feature_types = ["faces", "edges", "corners"]
#     for feature_type in feature_types:
#         for feature_id, feature in sd_local.features[feature_type].items():
#             feature_data[feature_id] = rank

#     recv_data = pmmoto.core.communication.communicate_features(
#         subdomain=sd_local,
#         send_data=feature_data,
#         feature_types=feature_types,
#         unpack=True,
#     )

#     for feature_type in feature_types:
#         for feature_id, feature in sd_local.features[feature_type].items():
#             if feature_id in recv_data.keys():
#                 assert recv_data[feature_id] == feature.neighbor_rank
