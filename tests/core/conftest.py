import pytest
import pickle
import numpy as np
import pmmoto


@pytest.fixture
def generate_subdomain():
    """
    Generate a un-padded subdomain
    This allows rank to be passed as an argument
    """

    def _create_subdomain(rank):

        box = ((77, 100), (-45, 101.21), (-9.0, -3.14159))
        boundary_types = ((0, 0), (1, 1), (2, 2))
        inlet = ((1, 0), (0, 0), (0, 0))
        outlet = ((0, 1), (0, 0), (0, 0))
        voxels = (100, 100, 100)
        subdomains = (3, 3, 3)

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

        sd = pmmoto.core.subdomain.Subdomain(
            rank=rank,
            decomposed_domain=pmmoto_decomposed_domain,
        )
        return sd

    return _create_subdomain


@pytest.fixture
def generate_single_subdomain():
    """
    Generate a padded subdomain
    THis allows rank to be passed as an argument
    """

    def _create_subdomain(rank, periodic=True):
        box = ((0, 1.0), (0, 1.0), (0, 1.0))
        if periodic:
            boundary_types = ((2, 2), (2, 2), (2, 2))
        else:
            boundary_types = ((0, 0), (0, 0), (0, 0))
        inlet = ((1, 0), (0, 0), (0, 0))
        outlet = ((0, 1), (0, 0), (0, 0))
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
