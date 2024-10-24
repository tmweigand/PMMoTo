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


def test_voxels(domain, domain_decomposed, domain_discretization, subdomains):

    pmmoto_domain = pmmoto.core.Domain(
        domain["box"], domain["boundaries"], domain["inlet"], domain["outlet"]
    )

    pmmoto_discretized_domain = pmmoto.core.DiscretizedDomain.from_domain(
        domain=pmmoto_domain,
        voxels=domain_discretization["voxels"],
    )

    pmmoto_decomposed_domain = (
        pmmoto.core.domain_decompose.DecomposedDomain.from_discretized_domain(
            discretized_domain=pmmoto_discretized_domain,
            subdomain_map=domain_decomposed["subdomain_map"],
        )
    )

    for rank in range(pmmoto_decomposed_domain.num_subdomains):
        pmmoto_subdomain = pmmoto.core.Subdomain(
            rank=rank,
            index=subdomains["index"][rank],
            box=subdomains["box"][rank],
            boundaries=subdomains["boundaries"][rank],
            inlet=subdomains["inlet"][rank],
            outlet=subdomains["outlet"][rank],
            voxels=subdomains["voxels"][rank],
            start=subdomains["start"][rank],
            num_subdomains=pmmoto_decomposed_domain.num_subdomains,
            domain_voxels=domain_discretization["voxels"],
            neighbor_ranks=subdomains["neighbor_ranks"][rank],
        )

        grid = np.zeros(pmmoto_subdomain.voxels, dtype=np.uint64)

        pmmoto.core.voxels.get_boundary_set_info_NEW(
            subdomain=pmmoto_subdomain, img=grid, n_labels=1
        )

    # phase_label = pmmoto.core.voxels.get_label_phase_info(grid, label_grid)
    # print(phase_label)

    # pmmoto.core.voxels.count_label_voxels(grid, map)
