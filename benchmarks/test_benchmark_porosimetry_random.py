"""Benchmarks for porosimetry algorithms in PMMoTo.

Includes tests for random and inkbottle domains using morph, distance, and hybrid modes.
"""

from pmmoto import domain_generation
from pmmoto import filters
from pmmoto import initialize


def setup_random():
    """Set up benchmarking data for porosimetry with a random binary grid."""
    voxels = (400, 400, 400)
    prob_zero = 0.3
    seed = 1
    sd = initialize(voxels)
    img = domain_generation.gen_img_random_binary(voxels, prob_zero, seed)
    pm = domain_generation.porousmedia.PorousMedia(subdomain=sd, img=img)
    return sd, pm


def test_random_morph_mode_small_r(benchmark):
    """Benchmark porosimetry (morph mode, small radius) on random grid."""
    radius = 0.004
    sd, pm = setup_random()
    _ = benchmark(
        filters.porosimetry.porosimetry,
        subdomain=sd,
        porous_media=pm,
        radius=radius,
        mode="morph",
    )


def test_random_morph_mode_large_r(benchmark):
    """Benchmark porosimetry (morph mode, large radius) on random grid."""
    radius = 0.1
    sd, pm = setup_random()
    _ = benchmark(
        filters.porosimetry.porosimetry,
        subdomain=sd,
        porous_media=pm,
        radius=radius,
        mode="morph",
    )


def test_random_distance_small_r(benchmark):
    """Benchmark porosimetry (distance mode, small radius) on random grid."""
    radius = 0.004
    sd, pm = setup_random()
    _ = benchmark(
        filters.porosimetry.porosimetry,
        subdomain=sd,
        porous_media=pm,
        radius=radius,
        mode="distance",
    )


def test_random_distance_large_r(benchmark):
    """Benchmark porosimetry (distance mode, large radius) on random grid."""
    radius = 0.1
    sd, pm = setup_random()
    _ = benchmark(
        filters.porosimetry.porosimetry,
        subdomain=sd,
        porous_media=pm,
        radius=radius,
        mode="distance",
    )


def test_random_hybrid_small_r(benchmark):
    """Benchmark porosimetry (hybrid mode, small radius) on random grid."""
    radius = 0.004
    sd, pm = setup_random()
    _ = benchmark(
        filters.porosimetry.porosimetry,
        subdomain=sd,
        porous_media=pm,
        radius=radius,
        mode="hybrid",
    )


def test_random_hybrid_large_r(benchmark):
    """Benchmark porosimetry (hybrid mode, large radius) on random grid."""
    radius = 0.1
    sd, pm = setup_random()
    _ = benchmark(
        filters.porosimetry.porosimetry,
        subdomain=sd,
        porous_media=pm,
        radius=radius,
        mode="hybrid",
    )
