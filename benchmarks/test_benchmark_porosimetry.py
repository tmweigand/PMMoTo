from pmmoto import domain_generation
from pmmoto import filters
from pmmoto import initialize
import pytest


def setup_random():
    """Set up benchmarking data for porosimetry with a random binary grid."""
    voxels = (400, 400, 400)
    prob_zero = 0.3
    seed = 1
    sd = initialize(voxels)
    img = domain_generation.gen_random_binary_grid(voxels, prob_zero, seed)
    return sd, img


def setup_inkbottle():
    """Set up benchmarking data for porosimetry with the inkbottle case."""
    voxels = (560, 120, 120)
    box = ((0.0, 14.0), (-1.5, 1.5), (-1.5, 1.5))
    inlet = ((0, 1), (0, 0), (0, 0))
    sd = initialize(voxels, box, inlet=inlet)
    pm = domain_generation.gen_pm_inkbottle(sd)
    img = pm.img
    return sd, img


@pytest.mark.order(1)
def test_random_morph_mode_small_r(benchmark):
    """Benchmark porosimetry (morph mode, small radius) on random grid."""
    radius = 0.004
    sd, img = setup_random()
    _ = benchmark(
        filters.porosimetry.porosimetry,
        subdomain=sd,
        img=img,
        radius=radius,
        mode="morph",
    )


@pytest.mark.order(2)
def test_random_morph_mode_large_r(benchmark):
    """Benchmark porosimetry (morph mode, large radius) on random grid."""
    radius = 0.1
    sd, img = setup_random()
    _ = benchmark(
        filters.porosimetry.porosimetry,
        subdomain=sd,
        img=img,
        radius=radius,
        mode="morph",
    )


@pytest.mark.order(3)
def test_inkbottle_morph_mode_small_r(benchmark):
    """Benchmark porosimetry (morph mode, small radius) on inkbottle case."""
    radius = 0.26
    sd, img = setup_inkbottle()
    _ = benchmark(
        filters.porosimetry.porosimetry,
        subdomain=sd,
        img=img,
        radius=radius,
        inlet=True,
        mode="morph",
    )


@pytest.mark.order(4)
def test_inkbottle_morph_mode_large_r(benchmark):
    """Benchmark porosimetry (morph mode, large radius) on inkbottle case."""
    radius = 1.2545
    sd, img = setup_inkbottle()
    _ = benchmark(
        filters.porosimetry.porosimetry,
        subdomain=sd,
        img=img,
        radius=radius,
        inlet=True,
        mode="morph",
    )


@pytest.mark.order(5)
def test_random_distance_small_r(benchmark):
    """Benchmark porosimetry (distance mode, small radius) on random grid."""
    radius = 0.004
    sd, img = setup_random()
    _ = benchmark(
        filters.porosimetry.porosimetry,
        subdomain=sd,
        img=img,
        radius=radius,
        mode="distance",
    )


@pytest.mark.order(6)
def test_random_distance_large_r(benchmark):
    """Benchmark porosimetry (distance mode, large radius) on random grid."""
    radius = 0.1
    sd, img = setup_random()
    _ = benchmark(
        filters.porosimetry.porosimetry,
        subdomain=sd,
        img=img,
        radius=radius,
        mode="distance",
    )


@pytest.mark.order(7)
def test_inkbottle_distance_small_r(benchmark):
    """Benchmark porosimetry (distance mode, small radius) on inkbottle case."""
    radius = 0.26
    sd, img = setup_inkbottle()
    _ = benchmark(
        filters.porosimetry.porosimetry,
        subdomain=sd,
        img=img,
        radius=radius,
        inlet=True,
        mode="distance",
    )


@pytest.mark.order(8)
def test_inkbottle_distance_large_r(benchmark):
    """Benchmark porosimetry (distance mode, large radius) on inkbottle case."""
    radius = 1.2545
    sd, img = setup_inkbottle()
    _ = benchmark(
        filters.porosimetry.porosimetry,
        subdomain=sd,
        img=img,
        radius=radius,
        inlet=True,
        mode="distance",
    )


@pytest.mark.order(9)
def test_random_hybrid_small_r(benchmark):
    """Benchmark porosimetry (hybrid mode, small radius) on random grid."""
    radius = 0.004
    sd, img = setup_random()
    _ = benchmark(
        filters.porosimetry.porosimetry,
        subdomain=sd,
        img=img,
        radius=radius,
        mode="hybrid",
    )


@pytest.mark.order(10)
def test_random_hybrid_large_r(benchmark):
    """Benchmark porosimetry (hybrid mode, large radius) on random grid."""
    radius = 0.1
    sd, img = setup_random()
    _ = benchmark(
        filters.porosimetry.porosimetry,
        subdomain=sd,
        img=img,
        radius=radius,
        mode="hybrid",
    )


@pytest.mark.order(11)
def test_inkbottle_hybrid_small_r(benchmark):
    """Benchmark porosimetry (hybrid mode, small radius) on inkbottle case."""
    radius = 0.26
    sd, img = setup_inkbottle()
    _ = benchmark(
        filters.porosimetry.porosimetry,
        subdomain=sd,
        img=img,
        radius=radius,
        inlet=True,
        mode="hybrid",
    )


@pytest.mark.order(12)
def test_inkbottle_hybrid_large_r(benchmark):
    """Benchmark porosimetry (hybrid mode, large radius) on inkbottle case."""
    radius = 1.2545
    sd, img = setup_inkbottle()
    _ = benchmark(
        filters.porosimetry.porosimetry,
        subdomain=sd,
        img=img,
        radius=radius,
        inlet=True,
        mode="hybrid",
    )
