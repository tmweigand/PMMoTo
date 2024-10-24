"""test_subdomain_features.py"""

import numpy as np
import pmmoto


def compare_dicts(true_dict, sample_dict, rank, feature):
    """Loop through dictionary to compare if equal

    Args:
        true_dict (dict): The true solution
        sample_dict (dict): The test solution
        rank (int): rank to test
        feature (str): "faces" "edges" or "corner"
    """
    for feat in true_dict[feature][rank]:
        for key in true_dict[feature][rank][feat].__dict__:
            true_value = getattr(true_dict[feature][rank][feat], key)
            _type = type(getattr(true_dict[feature][rank][feat], key))
            if _type != dict:
                assert true_value == getattr(sample_dict[feature][feat], key)
            else:
                sample_value = getattr(sample_dict[feature][feat], key)

                for key_2 in true_value.keys():
                    type_2 = type(true_value[key_2])
                    if type_2 != np.ndarray:
                        assert true_value[key_2] == sample_value[key_2]
                    else:
                        assert all(true_value[key_2] == sample_value[key_2])


# def test_subdomain_features(
#     domain,
#     domain_discretization,
#     domain_decomposed,
#     subdomains,
#     subdomain_features_true,
# ):
#     """
#     Test for subdomain features
#     """

#     pmmoto_domain = pmmoto.core.Domain(
#         domain["box"], domain["boundaries"], domain["inlet"], domain["outlet"]
#     )

#     pmmoto_discretized_domain = pmmoto.core.DiscretizedDomain.from_domain(
#         domain=pmmoto_domain,
#         voxels=domain_discretization["voxels"],
#     )

#     pmmoto_decomposed_domain = (
#         pmmoto.core.domain_decompose.DecomposedDomain.from_discretized_domain(
#             discretized_domain=pmmoto_discretized_domain,
#             subdomain_map=domain_decomposed["subdomain_map"],
#         )
#     )

#     # import pickle

#     # data_out = {"faces": {}, "edges": {}, "corners": {}}
#     pmmoto_subdomain = {}
#     padded_subdomain = {}
#     grid = {}
#     grid_padded = {}

#     for rank in range(pmmoto_decomposed_domain.num_subdomains):
#         pmmoto_subdomain[rank] = pmmoto.core.Subdomain(
#             rank=rank,
#             index=subdomains["index"][rank],
#             neighbor_ranks=subdomains["neighbor_ranks"][rank],
#             box=subdomains["box"][rank],
#             boundaries=subdomains["boundaries"][rank],
#             inlet=subdomains["inlet"][rank],
#             outlet=subdomains["outlet"][rank],
#             voxels=subdomains["voxels"][rank],
#             start=subdomains["start"][rank],
#             num_subdomains=pmmoto_decomposed_domain.num_subdomains,
#             domain_voxels=domain_discretization["voxels"],
#         )

#         compare_dicts(
#             subdomain_features_true, pmmoto_subdomain[rank].features, rank, "faces"
#         )
#         compare_dicts(
#             subdomain_features_true, pmmoto_subdomain[rank].features, rank, "edges"
#         )
#         compare_dicts(
#             subdomain_features_true, pmmoto_subdomain[rank].features, rank, "corners"
#         )

#         grid[rank] = np.zeros(pmmoto_subdomain[rank].voxels)
#         for f in pmmoto_subdomain[rank].features["faces"].values():
#             for i in range(f.loop[0][0], f.loop[0][1]):
#                 for j in range(f.loop[1][0], f.loop[1][1]):
#                     for k in range(f.loop[2][0], f.loop[2][1]):
#                         grid[rank][i, j, k] = rank

#     pmmoto.io.save_grid_data("data_out/test_subdomain", pmmoto_subdomain, grid)

# for rank in range(pmmoto_decomposed_domain.num_subdomains):
#     padded_subdomain[rank] = (
#         pmmoto.core.subdomain_padded.PaddedSubdomain.from_subdomain(
#             subdomain=pmmoto_subdomain[rank],
#             pad=(1, 1, 1),
#         )
#     )

#     grid_padded[rank] = np.zeros(padded_subdomain[rank].voxels)
#     for f in padded_subdomain[rank].features["faces"].values():
#         for i in range(f.loop[0][0], f.loop[0][1]):
#             for j in range(f.loop[1][0], f.loop[1][1]):
#                 for k in range(f.loop[2][0], f.loop[2][1]):
#                     grid_padded[rank][i, j, k] = rank

# pmmoto.io.save_grid_data("data_out/test_padded", padded_subdomain, grid_padded)

#     data_out["faces"][rank] = pmmoto_subdomain.features["faces"]
#     data_out["edges"][rank] = pmmoto_subdomain.features["edges"]
#     data_out["corners"][rank] = pmmoto_subdomain.features["corners"]

# with open(
#     "/Users/tim/Desktop/pmmoto/tests/core/test_output/test_subdomain_features.pkl",
#     "wb",
# ) as file:  # open a text file
#     pickle.dump(data_out, file)  # serialize the list


def test():
    """
    Test  subdomain features
    """

    # subdomain_map = [1, 1, 1]
    subdomain_map = [5, 5, 5]

    voxels = [100, 100, 100]

    box = [[0, 10], [0, 10], [0, 10]]
    # boundaries = [[0, 0], [0, 0], [0, 0]]
    boundaries = [[2, 2], [2, 2], [2, 2]]
    inlet = [[0, 0], [0, 0], [0, 0]]
    outlet = [[0, 0], [0, 0], [0, 0]]

    save_data = True

    sd = {}
    domain = {}
    grid = {}

    size = np.prod(subdomain_map)

    for rank in range(size):
        sd[rank], domain[rank] = pmmoto.initialize(
            box=box,
            subdomain_map=subdomain_map,
            voxels=voxels,
            boundaries=boundaries,
            inlet=inlet,
            outlet=outlet,
            rank=rank,
            mpi_size=size,
            reservoir_voxels=0,
        )
        grid[rank] = np.zeros(sd[rank].voxels, dtype=int)
        for f in sd[rank].features["faces"].values():
            for i in range(f.loop[0][0], f.loop[0][1]):
                for j in range(f.loop[1][0], f.loop[1][1]):
                    for k in range(f.loop[2][0], f.loop[2][1]):
                        grid[rank][i, j, k] = 1

        assert np.sum(grid[rank]) == 3888

        grid[rank] = np.zeros(sd[rank].voxels, dtype=int)
        for f in sd[rank].features["edges"].values():
            for i in range(f.loop[0][0], f.loop[0][1]):
                for j in range(f.loop[1][0], f.loop[1][1]):
                    for k in range(f.loop[2][0], f.loop[2][1]):
                        grid[rank][i, j, k] = 2

        assert np.sum(grid[rank]) == 1728

        grid[rank] = np.zeros(sd[rank].voxels, dtype=int)
        for f in sd[rank].features["corners"].values():
            for i in range(f.loop[0][0], f.loop[0][1]):
                for j in range(f.loop[1][0], f.loop[1][1]):
                    for k in range(f.loop[2][0], f.loop[2][1]):
                        grid[rank][i, j, k] = 5

        assert np.sum(grid[rank]) == 320

    if save_data:
        pmmoto.io.save_grid_data("data_out/test_output", sd, grid)
