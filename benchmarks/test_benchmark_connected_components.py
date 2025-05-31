from pmmoto import initialize
from pmmoto import domain_generation
from pmmoto import filters


def test_pmmoto_connected_components(benchmark):
    """ """
    voxels = (300, 300, 300)
    prob_zero = 0.1
    seed = 1
    sd = initialize(voxels)
    img = domain_generation.gen_random_binary_grid(voxels, prob_zero, seed)
    cc_pmmoto = benchmark(filters.connected_components.connect_components, img, sd)


def test_pmmoto_connected_components_periodic(benchmark):
    """ """
    voxels = (300, 300, 300)
    prob_zero = 0.1
    seed = 1
    boundary_types = ((2, 2), (2, 2), (2, 2))
    sd = initialize(voxels, boundary_types)
    img = domain_generation.gen_random_binary_grid(sd.voxels, prob_zero, seed)
    cc_pmmoto_periodic = benchmark(
        filters.connected_components.connect_components, img, sd
    )
