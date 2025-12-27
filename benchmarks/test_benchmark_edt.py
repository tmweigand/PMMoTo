"""Benchmark tests for PMMoTo and edt distance transform implementations."""

import edt
import pytest
from pmmoto import domain_generation
from pmmoto import filters


@pytest.mark.benchmark
def test_pmmoto_edt(benchmark):
    """Benchmark the PMMoTo 3D Euclidean distance transform (non-periodic).

    Args:
        benchmark: pytest-benchmark fixture.

    """
    voxels = (300, 300, 300)
    prob_zero = 0.1
    seed = 1
    img = domain_generation.gen_img_random_binary(voxels, prob_zero, seed)
    _ = benchmark(filters.distance.edt3d, img, periodic=[False, False, False])


def test_pmmoto_periodic_edt(benchmark):
    """Benchmark the PMMoTo 3D Euclidean distance transform with periodic boundaries.

    Args:
        benchmark: pytest-benchmark fixture.

    """
    voxels = (300, 300, 300)
    prob_zero = 0.1
    seed = 1
    img = domain_generation.gen_img_random_binary(voxels, prob_zero, seed)
    _ = benchmark(filters.distance.edt3d, img, periodic=[True, True, True])


def test_edt(benchmark):
    """Benchmark the reference edt.edt distance transform implementation.

    Args:
        benchmark: pytest-benchmark fixture.

    """
    voxels = (300, 300, 300)
    prob_zero = 0.1
    seed = 1
    img = domain_generation.gen_img_random_binary(voxels, prob_zero, seed)
    _ = benchmark(edt.edt, img)
