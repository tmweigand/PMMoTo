"""pmmoto.py"""

import numpy as np

from . import domain_decompose
from . import domain
from . import domain_discretization
from . import subdomain_padded
from . import subdomain_verlet
from . import utils


__all__ = ["initialize", "deconstruct_grid"]


def initialize(
    voxels,
    box=((0, 1.0), (0, 1.0), (0, 1)),
    subdomains=(1, 1, 1),
    boundary_types=((0, 0), (0, 0), (0, 0)),
    inlet=((0, 0), (0, 0), (0, 0)),
    outlet=((0, 0), (0, 0), (0, 0)),
    reservoir_voxels=0,
    rank=0,
    pad=(1, 1, 1),
    verlet_domains=(1, 1, 1),
):
    """Initialize PMMoTo domain and subdomain classes and check for valid inputs."""
    # utils.check_inputs(mpi_size, subdomain_map, voxels, boundaries, inlet, outlet)

    pmmoto_domain = domain.Domain(
        box=box, boundary_types=boundary_types, inlet=inlet, outlet=outlet
    )

    pmmoto_discretized_domain = domain_discretization.DiscretizedDomain.from_domain(
        domain=pmmoto_domain,
        voxels=voxels,
    )

    pmmoto_decomposed_domain = (
        domain_decompose.DecomposedDomain.from_discretized_domain(
            discretized_domain=pmmoto_discretized_domain,
            subdomains=subdomains,
        )
    )

    verlet_subdomain = subdomain_verlet.VerletSubdomain.from_subdomain(
        rank=rank,
        decomposed_domain=pmmoto_decomposed_domain,
        pad=pad,
        reservoir_voxels=reservoir_voxels,
        verlet_domains=verlet_domains,
    )

    return verlet_subdomain


def deconstruct_grid(
    subdomain,
    img,
    subdomains,
    rank=None,
    pad=(1, 1, 1),
    reservoir_voxels=0,
):
    """Deconstruct the grid from a single process to multiple subdomains and images

    The shape of the img must equal subdomain.domain.voxels!
    """
    num_procs = np.prod(subdomains)
    _domain = subdomain.domain

    if img.shape != _domain.voxels:
        raise ValueError(
            f"Error: img dimensions are incorrect. They must be {_domain.voxels}."
        )

    pmmoto_decomposed_domain = (
        domain_decompose.DecomposedDomain.from_discretized_domain(
            discretized_domain=_domain,
            subdomains=subdomains,
        )
    )

    if rank is not None:
        padded_subdomain = subdomain_padded.PaddedSubdomain(
            rank=rank,
            decomposed_domain=pmmoto_decomposed_domain,
            pad=pad,
            reservoir_voxels=reservoir_voxels,
        )
        local_grid = utils.decompose_img(
            img=img,
            start=padded_subdomain.start,
            shape=padded_subdomain.voxels,
        )

        local_grid = padded_subdomain.set_wall_bcs(local_grid)

    else:
        padded_subdomain = {}
        local_grid = {}
        for n in range(0, num_procs):
            padded_subdomain[n] = subdomain_padded.PaddedSubdomain(
                rank=n,
                decomposed_domain=pmmoto_decomposed_domain,
                pad=pad,
                reservoir_voxels=reservoir_voxels,
            )

            local_grid[n] = utils.decompose_img(
                img=img,
                start=padded_subdomain[n].start,
                shape=padded_subdomain[n].voxels,
            )

            local_grid[n] = padded_subdomain[n].set_wall_bcs(local_grid[n])

    return padded_subdomain, local_grid
