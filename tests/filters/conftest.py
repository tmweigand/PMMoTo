"""Pytest fixtures for PMMoTo filter tests.

Provides subdomain and simple subdomain generators for use in test cases.
"""

import pytest
import pmmoto


@pytest.fixture
def generate_subdomain() -> pmmoto.core.subdomain_padded.PaddedSubdomain:
    """Generate a padded subdomain.

    This allows rank to be passed as an argument
    """

    def _create_subdomain(rank, periodic=True, specified_types=None):
        box = ((0, 1.0), (0, 1.0), (0, 1.0))
        if specified_types is not None:
            boundary_types = specified_types
        elif periodic:
            boundary_types = (
                (pmmoto.BoundaryType.PERIODIC, pmmoto.BoundaryType.PERIODIC),
                (pmmoto.BoundaryType.PERIODIC, pmmoto.BoundaryType.PERIODIC),
                (pmmoto.BoundaryType.PERIODIC, pmmoto.BoundaryType.PERIODIC),
            )
        else:
            boundary_types = (
                (pmmoto.BoundaryType.END, pmmoto.BoundaryType.END),
                (pmmoto.BoundaryType.END, pmmoto.BoundaryType.END),
                (pmmoto.BoundaryType.END, pmmoto.BoundaryType.END),
            )
        inlet = ((True, False), (False, False), (False, False))
        outlet = ((False, True), (False, False), (False, False))
        voxels = (100, 100, 100)
        subdomains = (1, 1, 1)
        pad = (1, 1, 1)
        reservoir_voxels = 0

        pmmoto_domain = pmmoto.core.domain.Domain(
            box=box, boundary_types=boundary_types, inlet=inlet, outlet=outlet
        )

        pmmoto_discretized_domain = (
            pmmoto.core.domain_discretization.DiscretizedDomain.from_domain(
                domain=pmmoto_domain, voxels=voxels
            )
        )

        pmmoto_decomposed_domain = (
            pmmoto.core.domain_decompose.DecomposedDomain.from_discretized_domain(
                discretized_domain=pmmoto_discretized_domain,
                subdomains=subdomains,
            )
        )

        padded_subdomain = pmmoto.core.subdomain_padded.PaddedSubdomain(
            rank=rank,
            decomposed_domain=pmmoto_decomposed_domain,
            pad=pad,
            reservoir_voxels=reservoir_voxels,
        )
        return padded_subdomain

    return _create_subdomain


@pytest.fixture
def generate_simple_subdomain() -> pmmoto.core.subdomain_padded.PaddedSubdomain:
    """Generate a padded subdomain.

    This allows rank to be passed as an argument
    """

    def _create_subdomain(
        rank,
        box=((0, 1.0), (0, 1.0), (0, 1.0)),
        periodic=True,
        specified_types=None,
        voxels_in=(10, 10, 10),
    ):

        if specified_types is not None:
            boundary_types = specified_types
        elif periodic:
            boundary_types = (
                (pmmoto.BoundaryType.PERIODIC, pmmoto.BoundaryType.PERIODIC),
                (pmmoto.BoundaryType.PERIODIC, pmmoto.BoundaryType.PERIODIC),
                (pmmoto.BoundaryType.PERIODIC, pmmoto.BoundaryType.PERIODIC),
            )
        else:
            boundary_types = (
                (pmmoto.BoundaryType.END, pmmoto.BoundaryType.END),
                (pmmoto.BoundaryType.END, pmmoto.BoundaryType.END),
                (pmmoto.BoundaryType.END, pmmoto.BoundaryType.END),
            )
        inlet = ((True, False), (False, False), (False, False))
        outlet = ((False, True), (False, False), (False, False))
        voxels = voxels_in
        subdomains = (1, 1, 1)
        pad = (1, 1, 1)
        reservoir_voxels = 0

        pmmoto_domain = pmmoto.core.domain.Domain(
            box=box, boundary_types=boundary_types, inlet=inlet, outlet=outlet
        )

        pmmoto_discretized_domain = (
            pmmoto.core.domain_discretization.DiscretizedDomain.from_domain(
                domain=pmmoto_domain, voxels=voxels
            )
        )

        pmmoto_decomposed_domain = (
            pmmoto.core.domain_decompose.DecomposedDomain.from_discretized_domain(
                discretized_domain=pmmoto_discretized_domain,
                subdomains=subdomains,
            )
        )

        padded_subdomain = pmmoto.core.subdomain_padded.PaddedSubdomain(
            rank=rank,
            decomposed_domain=pmmoto_decomposed_domain,
            pad=pad,
            reservoir_voxels=reservoir_voxels,
        )
        return padded_subdomain

    return _create_subdomain
