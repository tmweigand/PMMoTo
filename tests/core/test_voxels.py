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


# def test_voxels(domain, domain_decomposed, domain_discretization, subdomains):

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

#     for rank in range(pmmoto_decomposed_domain.num_subdomains):
#         pmmoto_subdomain = pmmoto.core.Subdomain(
#             rank=rank,
#             index=subdomains["index"][rank],
#             box=subdomains["box"][rank],
#             boundaries=subdomains["boundaries"][rank],
#             inlet=subdomains["inlet"][rank],
#             outlet=subdomains["outlet"][rank],
#             voxels=subdomains["voxels"][rank],
#             start=subdomains["start"][rank],
#             num_subdomains=pmmoto_decomposed_domain.num_subdomains,
#             domain_voxels=domain_discretization["voxels"],
#             neighbor_ranks=subdomains["neighbor_ranks"][rank],
#         )

#         grid = np.zeros(pmmoto_subdomain.voxels, dtype=np.uint64)

#         pmmoto.core.voxels.get_boundary_set_info_NEW(
#             subdomain=pmmoto_subdomain, img=grid, n_labels=1
#         )

# phase_label = pmmoto.core.voxels.get_label_phase_info(grid, label_grid)
# print(phase_label)

# pmmoto.core.voxels.count_label_voxels(grid, map)


def test_boundary_set_info():
    """
    Test  subdomain features
    """

    subdomain_map = [1, 1, 1]
    # subdomain_map = [2, 2, 2]

    voxels = (10, 10, 10)

    box = [[0, 10], [0, 10], [0, 10]]
    # boundaries = [[0, 0], [0, 0], [0, 0]]
    boundaries = [[2, 2], [0, 0], [0, 0]]
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

        grid = np.zeros(sd[rank].voxels, dtype=np.uint64)
        # grid[0, :, :] = 1
        data = pmmoto.core.voxels.get_boundary_voxels(
            subdomain=sd[rank], img=grid, n_labels=2
        )

        send_data = pmmoto.core.voxels.boundary_voxels_pack(sd[rank], data)
        # _recv_data = pmmoto.core.communication.communicate_NEW(sd[rank], send_data)
        recv_data = pmmoto.core.voxels.boundary_voxels_unpack(sd[rank], data, data)

        print(data)
