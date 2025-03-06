from pmmoto import domain_generation
from pmmoto import filters
from pmmoto import initialize
import pytest


def setup_random():
    """
    Setup for benchmarking porosimetry with random binary grid.
    """
    voxels = (400, 400, 400)
    prob_zero = 0.3
    seed = 1
    sd = initialize(voxels)
    img = domain_generation.gen_random_binary_grid(voxels, prob_zero, seed)
    return sd, img


def setup_inkbottle():
    """
    Setup for benchmarking porosimetry with inkbottle case.
    """
    voxels = (400, 400, 400)
    sd = initialize(voxels)
    pm = domain_generation.gen_pm_inkbottle(sd)
    img = pm.img
    return sd, img


@pytest.mark.order(1)
def test_random_morph_mode_small_r(benchmark):
    """ """
    radius = 0.004
    sd, img = setup_random()
    morph_mode = benchmark(
        filters.porosimetry.porosimetry,
        subdomain=sd,
        img=img,
        radius=radius,
        mode="morph",
    )


@pytest.mark.order(2)
def test_random_morph_mode_large_r(benchmark):
    """ """
    radius = 0.1
    sd, img = setup_random()
    morph_mode = benchmark(
        filters.porosimetry.porosimetry,
        subdomain=sd,
        img=img,
        radius=radius,
        mode="morph",
    )


@pytest.mark.order(3)
def test_inkbottle_morph_mode_small_r(benchmark):
    """ """
    radius = 0.004
    sd, img = setup_inkbottle()
    morph_mode = benchmark(
        filters.porosimetry.porosimetry,
        subdomain=sd,
        img=img,
        radius=radius,
        mode="morph",
    )


@pytest.mark.order(4)
def test_inkbottle_morph_mode_large_r(benchmark):
    """ """
    radius = 0.1
    sd, img = setup_inkbottle()
    morph_mode = benchmark(
        filters.porosimetry.porosimetry,
        subdomain=sd,
        img=img,
        radius=radius,
        mode="morph",
    )


@pytest.mark.order(5)
def test_random_distance_small_r(benchmark):
    """ """
    radius = 0.004
    sd, img = setup_random()
    distance_mode = benchmark(
        filters.porosimetry.porosimetry,
        subdomain=sd,
        img=img,
        radius=radius,
        mode="distance",
    )


@pytest.mark.order(6)
def test_random_distance_large_r(benchmark):
    """ """
    radius = 0.1
    sd, img = setup_random()
    distance_mode = benchmark(
        filters.porosimetry.porosimetry,
        subdomain=sd,
        img=img,
        radius=radius,
        mode="distance",
    )


@pytest.mark.order(7)
def test_inkbottle_distance_small_r(benchmark):
    """ """
    radius = 0.004
    sd, img = setup_inkbottle()
    distance_mode = benchmark(
        filters.porosimetry.porosimetry,
        subdomain=sd,
        img=img,
        radius=radius,
        mode="distance",
    )


@pytest.mark.order(8)
def test_inkbottle_distance_large_r(benchmark):
    """ """
    radius = 0.1
    sd, img = setup_inkbottle()
    distance_mode = benchmark(
        filters.porosimetry.porosimetry,
        subdomain=sd,
        img=img,
        radius=radius,
        mode="distance",
    )


@pytest.mark.order(9)
def test_random_hybrid_small_r(benchmark):
    """ """
    radius = 0.004
    sd, img = setup_random()
    hybrid = benchmark(
        filters.porosimetry.porosimetry,
        subdomain=sd,
        img=img,
        radius=radius,
        mode="hybrid",
    )


@pytest.mark.order(10)
def test_random_hybrid_large_r(benchmark):
    """ """
    radius = 0.1
    sd, img = setup_random()
    hybrid = benchmark(
        filters.porosimetry.porosimetry,
        subdomain=sd,
        img=img,
        radius=radius,
        mode="hybrid",
    )


@pytest.mark.order(11)
def test_inkbottle_hybrid_small_r(benchmark):
    """ """
    radius = 0.004
    sd, img = setup_inkbottle()
    hybrid = benchmark(
        filters.porosimetry.porosimetry,
        subdomain=sd,
        img=img,
        radius=radius,
        mode="hybrid",
    )


@pytest.mark.order(12)
def test_inkbottle_hybrid_large_r(benchmark):
    """ """
    radius = 0.1
    sd, img = setup_inkbottle()
    hybrid = benchmark(
        filters.porosimetry.porosimetry,
        subdomain=sd,
        img=img,
        radius=radius,
        mode="hybrid",
    )
