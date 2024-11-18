"""test_voxels.py"""

import numpy as np
import pmmoto


def test_voxls_get_id():
    """
    Simple test to check voxel id mapping.
    """
    x = [1, 2, 7]
    v = [5, 6, 5]
    id = pmmoto.core.voxels.get_id(x, v)
    assert id == 42

    x = [-1, -1, -1]
    v = [10, 10, 10]
    id = pmmoto.core.voxels.get_id(x, v)
    assert id == 999


# # def test_voxels(domain, domain_decomposed, domain_discretization, subdomains):

# #     pmmoto_domain = pmmoto.core.Domain(
# #         domain["box"], domain["boundaries"], domain["inlet"], domain["outlet"]
# #     )

# #     pmmoto_discretized_domain = pmmoto.core.DiscretizedDomain.from_domain(
# #         domain=pmmoto_domain,
# #         voxels=domain_discretization["voxels"],
# #     )

# #     pmmoto_decomposed_domain = (
# #         pmmoto.core.domain_decompose.DecomposedDomain.from_discretized_domain(
# #             discretized_domain=pmmoto_discretized_domain,
# #             subdomain_map=domain_decomposed["subdomain_map"],
# #         )
# #     )

# #     for rank in range(pmmoto_decomposed_domain.num_subdomains):
# #         pmmoto_subdomain = pmmoto.core.Subdomain(
# #             rank=rank,
# #             index=subdomains["index"][rank],
# #             box=subdomains["box"][rank],
# #             boundaries=subdomains["boundaries"][rank],
# #             inlet=subdomains["inlet"][rank],
# #             outlet=subdomains["outlet"][rank],
# #             voxels=subdomains["voxels"][rank],
# #             start=subdomains["start"][rank],
# #             num_subdomains=pmmoto_decomposed_domain.num_subdomains,
# #             domain_voxels=domain_discretization["voxels"],
# #             neighbor_ranks=subdomains["neighbor_ranks"][rank],
# #         )

# #         grid = np.zeros(pmmoto_subdomain.voxels, dtype=np.uint64)

# #         pmmoto.core.voxels.get_boundary_set_info_NEW(
# #             subdomain=pmmoto_subdomain, img=grid, n_labels=1
# #         )

# # phase_label = pmmoto.core.voxels.get_label_phase_info(grid, label_grid)
# # print(phase_label)

# # pmmoto.core.voxels.count_label_voxels(grid, map)


# def test_boundary_voxel_info():
#     """
#     Test  subdomain features
#     """

#     subdomains = [1, 1, 1]
#     subdomains = [2, 2, 2]

#     voxels = (10, 10, 10)

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
#         )

#         grid = np.arange(np.prod(sd[rank].voxels), dtype=np.uint64).reshape(
#             sd[rank].voxels
#         )

#         grid = pmmoto.core.communication.update_buffer(sd[rank], grid)

#         data = pmmoto.core.voxels.get_boundary_voxels(
#             subdomain=sd[rank],
#             img=grid,
#         )

#         send_data, own_data = pmmoto.core.voxels.boundary_voxels_pack(sd[rank], data)

#         if send_data:
#             recv_data = pmmoto.core.communication.communicate_NEW(sd[rank], send_data)
#             own_data.update(recv_data)

#         matches = pmmoto.core.voxels.match_neighbor_boundary_voxels(
#             sd[rank], data, own_data
#         )

#         pmmoto.core.voxels.match_global_boundary_voxels(sd[rank], matches)


# def test_merge_matched_voxels():
#     pass
#     # all_match_data = [
#     #     {
#     #         (0, 0): {"neighbor": [(1, 0), (2, 0), (3, 0)]},
#     #         (0, 1): {"neighbor": [(1, 1)]},
#     #     },
#     #     {(1, 0): {"neighbor": [(0, 0)]}, (1, 1): {"neighbor": [(0, 1)]}},
#     #     {(2, 0): {"neighbor": [(0, 0)]}},
#     #     {(3, 0): {"neighbor": [(0, 0)]}},
#     # ]

#     # pmmoto.core._voxels._merge_matched_voxels(all_match_data)

#     # print(matches, merged_sets)
