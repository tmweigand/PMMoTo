from pmmoto import domain_generation
from pmmoto import filters
from pmmoto import initialize
import pytest


def setup():
    """Setup for for benchmarking morphological operators
    """
    voxels = (600, 600, 600)
    prob_zero = 0.3
    seed = 1
    sd = initialize(voxels)
    img = domain_generation.gen_random_binary_grid(voxels, prob_zero, seed)
    return sd, img


@pytest.mark.benchmark
def test_morp_addition_fft_small_r(benchmark):
    """ """
    radius = 0.004
    fft = True
    sd, img = setup()
    morp_addition_fft = benchmark(
        filters.morphological_operators.addition,
        subdomain=sd,
        img=img,
        radius=radius,
        fft=fft,
    )


@pytest.mark.benchmark
def test_morp_addition_fft_large_r(benchmark):
    """ """
    radius = 0.1
    fft = True
    sd, img = setup()
    morp_addition_fft = benchmark(
        filters.morphological_operators.addition,
        subdomain=sd,
        img=img,
        radius=radius,
        fft=fft,
    )


@pytest.mark.benchmark
def test_morp_addition_edt_small_r(benchmark):
    """ """
    radius = 0.004
    fft = False
    sd, img = setup()
    morp_addition_edt = benchmark(
        filters.morphological_operators.addition,
        subdomain=sd,
        img=img,
        radius=radius,
        fft=fft,
    )


@pytest.mark.benchmark
def test_morp_addition_edt_large_r(benchmark):
    """ """
    radius = 0.1
    fft = False
    sd, img = setup()
    morp_addition_edt = benchmark(
        filters.morphological_operators.addition,
        subdomain=sd,
        img=img,
        radius=radius,
        fft=fft,
    )
