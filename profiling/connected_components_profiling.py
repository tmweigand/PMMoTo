import pmmoto

import profiling_utils


@profiling_utils.profile("profiling/connect_components.prof")
def test_connected_components():
    """
    Profiling for connected components.
    To run:
        python profiling/edt_profiling.py
    Note: Cannot be used on python 12!!!!
    """

    voxels = (600, 600, 600)
    prob_zero = 0.5
    seed = 1
    box = ((0, 1), (0, 1), (0, 1))
    subdomains = (1, 1, 1)
    sd = pmmoto.initialize(box, subdomains, voxels)
    img = pmmoto.domain_generation.gen_random_binary_grid(voxels, prob_zero, seed)

    cc = pmmoto.filters.connected_components.connect_components(img, sd)
    inlet = pmmoto.filters.connected_components.gen_inlet_label_map(sd, cc)
    outlet = pmmoto.filters.connected_components.gen_outlet_label_map(sd, cc)


@profiling_utils.profile("profiling/connect_components_periodic.prof")
def test_connected_components_periodic():
    """
    Profiling for connected components.
    To run:
        python profiling/edt_profiling.py
    Note: Cannot be used on python 12!!!!
    """

    voxels = (600, 600, 600)
    prob_zero = 0.5
    seed = 1
    box = ((0, 1), (0, 1), (0, 1))
    subdomains = (1, 1, 1)
    boundary_types = ((2, 2), (2, 2), (2, 2))
    sd = pmmoto.initialize(box, subdomains, voxels, boundary_types)
    img = pmmoto.domain_generation.gen_random_binary_grid(sd.voxels, prob_zero, seed)

    cc = pmmoto.filters.connected_components.connect_components(img, sd)
    inlet = pmmoto.filters.connected_components.gen_inlet_label_map(sd, cc)
    outlet = pmmoto.filters.connected_components.gen_outlet_label_map(sd, cc)

    # pmmoto.io.output.save_grid_data_parallel(
    #     "data_out/test_connect_components",
    #     subdomain=sd,
    #     img=img,
    #     **{"cc": cc},
    # )


if __name__ == "__main__":
    test_connected_components()
    test_connected_components_periodic()
