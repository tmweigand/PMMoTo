"""Benchmark tests for PMMoTo connected components filter."""

from pmmoto import initialize
from pmmoto import domain_generation
from pmmoto import filters


def test_pmmoto_connected_components(benchmark):
    """Benchmark the connected components filter on a random binary grid.

    Args:
        benchmark: pytest-benchmark fixture.

    """
    voxels = (300, 300, 300)
    prob_zero = 0.1
    seed = 1
    sd = initialize(voxels)
    img = domain_generation.gen_img_random_binary(voxels, prob_zero, seed)
    _ = benchmark(filters.connected_components.connect_components, img, sd)


def test_pmmoto_connected_components_periodic(benchmark):
    """Benchmark the connected components filter with periodic boundary conditions.

    Args:
        benchmark: pytest-benchmark fixture.

    """
    voxels = (300, 300, 300)
    prob_zero = 0.1
    seed = 1
    boundary_types = ((2, 2), (2, 2), (2, 2))
    sd = initialize(voxels, boundary_types)
    img = domain_generation.gen_img_random_binary(sd.voxels, prob_zero, seed)
    _ = benchmark(filters.connected_components.connect_components, img, sd)
