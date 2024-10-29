import numpy as np
import pmmoto


def test_decomposed_domain(
    domain, domain_discretization, domain_decomposed, domain_decomposed_true
):
    """
    Test decomposition of domain
    """

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

    # import pickle

    # data_out = {
    #     "index": {},
    #     "voxels": {},
    #     "boundaries": {},
    #     "boundary_type": {},
    #     "box": {},
    #     "inlet": {},
    #     "outlet": {},
    #     "map": {},
    #     "boundary_features": {},
    #     "neighbor_ranks": {},
    #     "start": {},
    #     "num_subdomains": {},
    #     "domain_voxels": {},
    # }

    for rank in range(pmmoto_decomposed_domain.num_subdomains):
        pmmoto_index = pmmoto_decomposed_domain.get_subdomain_index(rank)
        assert pmmoto_index == domain_decomposed_true["index"][rank]

        voxels = pmmoto_decomposed_domain.get_subdomain_voxels(pmmoto_index)
        assert voxels == domain_decomposed_true["voxels"][rank]

        boundaries, boundary_type = pmmoto_decomposed_domain.get_subdomain_boundaries(
            pmmoto_index
        )
        np.testing.assert_array_equal(
            boundaries, domain_decomposed_true["boundaries"][rank]
        )
        np.testing.assert_array_equal(
            boundary_type, domain_decomposed_true["boundary_type"][rank]
        )

        box = pmmoto_decomposed_domain.get_subdomain_box(pmmoto_index, voxels)
        np.testing.assert_array_equal(box, domain_decomposed_true["box"][rank])

        inlet = pmmoto_decomposed_domain.get_subdomain_inlet(pmmoto_index)
        np.testing.assert_array_equal(inlet, domain_decomposed_true["inlet"][rank])

        outlet = pmmoto_decomposed_domain.get_subdomain_outlet(pmmoto_index)
        np.testing.assert_array_equal(outlet, domain_decomposed_true["outlet"][rank])

        map, boundary_features = pmmoto_decomposed_domain.gen_maps()
        np.testing.assert_array_equal(map, domain_decomposed_true["map"][rank])
        np.testing.assert_array_equal(
            boundary_features, domain_decomposed_true["boundary_features"][rank]
        )

        neighbor_ranks = pmmoto_decomposed_domain.get_neighbor_ranks(pmmoto_index)
        np.testing.assert_array_equal(
            neighbor_ranks, domain_decomposed_true["neighbor_ranks"][rank]
        )

        start = pmmoto_decomposed_domain.get_subdomain_start(pmmoto_index)
        np.testing.assert_array_equal(start, domain_decomposed_true["start"][rank])

        # data_out["index"][rank] = pmmoto_index
        # data_out["voxels"][rank] = voxels
        # data_out["boundaries"][rank] = boundaries
        # data_out["boundary_type"][rank] = boundary_type
        # data_out["box"][rank] = box
        # data_out["inlet"][rank] = inlet
        # data_out["outlet"][rank] = outlet
        # data_out["map"][rank] = map
        # data_out["boundary_features"][rank] = boundary_features
        # data_out["neighbor_ranks"][rank] = neighbor_ranks
        # data_out["start"][rank] = start
        # data_out["num_subdomains"][rank] = pmmoto_decomposed_domain.num_subdomains
        # data_out["domain_voxels"][rank] = pmmoto_decomposed_domain.voxels

    # with open(
    #     "/Users/tim/Desktop/pmmoto/tests/core/test_output/test_decomposed_domain.pkl",
    #     "wb",
    # ) as file:  # open a text file
    #     pickle.dump(data_out, file)  # serialize the list


def test_maps():
    """
    Test  subdomain features
    """

    subdomain_map = (1, 1, 1)
    voxels = (10, 10, 10)
    box = [[0, 10], [0, 10], [0, 10]]
    boundaries = [[2, 2], [0, 0], [0, 0]]
    inlet = [[0, 0], [0, 0], [0, 0]]
    outlet = [[0, 0], [0, 0], [0, 0]]

    pmmoto_decomposed_domain = pmmoto.core.domain_decompose.DecomposedDomain(
        box=box,
        boundaries=boundaries,
        inlet=inlet,
        outlet=outlet,
        voxels=voxels,
        subdomain_map=subdomain_map,
    )

    pmmoto_decomposed_domain.initialize_subdomain(0)

    # print()
    # print()
    # print(pmmoto_decomposed_domain.boundary_features)

    # print(
    #     pmmoto_decomposed_domain.get_subdomain_boundaries((0, 0, 0)),
    #     pmmoto_decomposed_domain.map,
    #     pmmoto_decomposed_domain.boundary_features,
    # )
