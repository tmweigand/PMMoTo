"""Benchmarks for porosimetry algorithms in PMMoTo.

Includes tests for random and inkbottle domains using morph, distance, and hybrid modes.
"""

import pytest

from pmmoto import domain_generation
from pmmoto import filters
from pmmoto import initialize


def setup_inkbottle():
    """Set up benchmarking data for porosimetry with the inkbottle case."""
    voxels = (560, 120, 120)
    box = ((0.0, 14.0), (-1.5, 1.5), (-1.5, 1.5))
    inlet = ((0, 1), (0, 0), (0, 0))
    sd = initialize(voxels, box, inlet=inlet)
    pm = domain_generation.gen_pm_inkbottle(sd)
    return sd, pm


@pytest.mark.benchmark(group="porosimetry_inkbottle_small_radius")
def test_inkbottle_morph_mode_small_r(benchmark):
    """Benchmark porosimetry (morph mode, small radius) on inkbottle case."""
    radius = 0.26
    sd, pm = setup_inkbottle()
    _ = benchmark(
        filters.porosimetry.porosimetry,
        subdomain=sd,
        porous_media=pm,
        radius=radius,
        inlet=True,
        mode="morph",
    )


@pytest.mark.benchmark(group="porosimetry_inkbottle_large_radius")
def test_inkbottle_morph_mode_large_r(benchmark):
    """Benchmark porosimetry (morph mode, large radius) on inkbottle case."""
    radius = 1.2545
    sd, pm = setup_inkbottle()
    _ = benchmark(
        filters.porosimetry.porosimetry,
        subdomain=sd,
        porous_media=pm,
        radius=radius,
        inlet=True,
        mode="morph",
    )


@pytest.mark.benchmark(group="porosimetry_inkbottle_small_radius")
def test_inkbottle_distance_small_r(benchmark):
    """Benchmark porosimetry (distance mode, small radius) on inkbottle case."""
    radius = 0.26
    sd, pm = setup_inkbottle()
    _ = benchmark(
        filters.porosimetry.porosimetry,
        subdomain=sd,
        porous_media=pm,
        radius=radius,
        inlet=True,
        mode="distance",
    )


@pytest.mark.benchmark(group="porosimetry_inkbottle_large_radius")
def test_inkbottle_distance_large_r(benchmark):
    """Benchmark porosimetry (distance mode, large radius) on inkbottle case."""
    radius = 1.2545
    sd, pm = setup_inkbottle()
    _ = benchmark(
        filters.porosimetry.porosimetry,
        subdomain=sd,
        porous_media=pm,
        radius=radius,
        inlet=True,
        mode="distance",
    )


@pytest.mark.benchmark(group="porosimetry_inkbottle_small_radius")
def test_inkbottle_hybrid_small_r(benchmark):
    """Benchmark porosimetry (hybrid mode, small radius) on inkbottle case."""
    radius = 0.26
    sd, pm = setup_inkbottle()
    _ = benchmark(
        filters.porosimetry.porosimetry,
        subdomain=sd,
        porous_media=pm,
        radius=radius,
        inlet=True,
        mode="hybrid",
    )


@pytest.mark.benchmark(group="porosimetry_inkbottle_large_radius")
def test_inkbottle_hybrid_large_r(benchmark):
    """Benchmark porosimetry (hybrid mode, large radius) on inkbottle case."""
    radius = 1.2545
    sd, pm = setup_inkbottle()
    _ = benchmark(
        filters.porosimetry.porosimetry,
        subdomain=sd,
        porous_media=pm,
        radius=radius,
        inlet=True,
        mode="hybrid",
    )
