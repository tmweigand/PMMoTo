"""test_set.py"""

import numpy as np
import pmmoto


def test_set():
    pass
    # pmmoto_domain = pmmoto.core.Domain(
    #     domain["box"], domain["boundaries"], domain["inlet"], domain["outlet"]
    # )

    # pmmoto_discretized_domain = pmmoto.core.DiscretizedDomain.from_domain(
    #     domain=pmmoto_domain,
    #     voxels=domain_discretization["voxels"],
    # )

    # pmmoto_decomposed_domain = (
    #     pmmoto.core.domain_decompose.DecomposedDomain.from_discretized_domain(
    #         discretized_domain=pmmoto_discretized_domain,
    #         subdomain_map=domain_decomposed["subdomain_map"],
    #     )
    # )

    # for rank in range(pmmoto_decomposed_domain.num_subdomains):
    #     pmmoto_subdomain = pmmoto.core.Subdomain(
    #         rank=rank,
    #         index=subdomains["index"][rank],
    #         box=subdomains["box"][rank],
    #         boundaries=subdomains["boundaries"][rank],
    #         inlet=subdomains["inlet"][rank],
    #         outlet=subdomains["outlet"][rank],
    #         voxels=subdomains["voxels"][rank],
    #         start=subdomains["start"][rank],
    #         num_subdomains=pmmoto_decomposed_domain.num_subdomains,
    #         domain_voxels=domain_discretization["voxels"],
    #         neighbor_ranks=subdomains["neighbor_ranks"][rank],
    #     )

    #     _set = pmmoto.core.set.Set(
    #         subdomain=pmmoto_subdomain,
    #         local_ID=0,
    #         phase=-1,
    #     )
