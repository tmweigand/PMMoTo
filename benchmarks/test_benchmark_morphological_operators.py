"""Benchmark tests for PMMoTo morphological operators."""

from pmmoto import domain_generation
from pmmoto import filters
from pmmoto import initialize
import pytest


def setup():
    """Set up benchmarking data for morphological operators.

    Returns:
        tuple: (subdomain, binary image)

    """
    voxels = (600, 600, 600)
    prob_zero = 0.3
    seed = 1
    sd = initialize(voxels)
    img = domain_generation.gen_random_binary_grid(voxels, prob_zero, seed)
    return sd, img


@pytest.mark.benchmark
def test_morp_addition_fft_small_r(benchmark):
    """Benchmark morphological addition (FFT method, small radius).

    Args:
        benchmark: pytest-benchmark fixture.

    """
    radius = 0.004
    fft = True
    sd, img = setup()
    _ = benchmark(
        filters.morphological_operators.addition,
        subdomain=sd,
        img=img,
        radius=radius,
        fft=fft,
    )


@pytest.mark.benchmark
def test_morp_addition_fft_large_r(benchmark):
    """Benchmark morphological addition (FFT method, large radius).

    Args:
        benchmark: pytest-benchmark fixture.

    """
    radius = 0.1
    fft = True
    sd, img = setup()
    _ = benchmark(
        filters.morphological_operators.addition,
        subdomain=sd,
        img=img,
        radius=radius,
        fft=fft,
    )


@pytest.mark.benchmark
def test_morp_addition_edt_small_r(benchmark):
    """Benchmark morphological addition (EDT method, small radius).

    Args:
        benchmark: pytest-benchmark fixture.

    """
    radius = 0.004
    fft = False
    sd, img = setup()
    _ = benchmark(
        filters.morphological_operators.addition,
        subdomain=sd,
        img=img,
        radius=radius,
        fft=fft,
    )


@pytest.mark.benchmark
def test_morp_addition_edt_large_r(benchmark):
    """Benchmark morphological addition (EDT method, large radius).

    Args:
        benchmark: pytest-benchmark fixture.

    """
    radius = 0.1
    fft = False
    sd, img = setup()
    _ = benchmark(
        filters.morphological_operators.addition,
        subdomain=sd,
        img=img,
        radius=radius,
        fft=fft,
    )
