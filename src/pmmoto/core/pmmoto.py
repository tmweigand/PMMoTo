"""pmmoto.py"""

from __future__ import annotations
from typing import TypeVar
import numpy as np

from .boundary_types import BoundaryType
from . import domain_decompose
from . import domain
from . import domain_discretization
from .subdomain import Subdomain
from .subdomain_padded import PaddedSubdomain
from .subdomain_verlet import VerletSubdomain

T = TypeVar("T", bound=np.generic)

__all__ = ["initialize"]


def initialize(
    voxels: tuple[int, ...],
    box: tuple[tuple[float, float], ...] = ((0, 1.0), (0, 1.0), (0, 1)),
    subdomains: tuple[int, ...] = (1, 1, 1),
    boundary_types: tuple[tuple[BoundaryType, BoundaryType], ...] = (
        (BoundaryType.END, BoundaryType.END),
        (BoundaryType.END, BoundaryType.END),
        (BoundaryType.END, BoundaryType.END),
    ),
    inlet: tuple[tuple[bool, bool], ...] = (
        (False, False),
        (False, False),
        (False, False),
    ),
    outlet: tuple[tuple[bool, bool], ...] = (
        (False, False),
        (False, False),
        (False, False),
    ),
    reservoir_voxels: int = 0,
    rank: int = 0,
    pad: tuple[int, ...] = (1, 1, 1),
    verlet_domains: tuple[int, ...] = (1, 1, 1),
    return_subdomain: bool = False,
) -> Subdomain | PaddedSubdomain | VerletSubdomain:
    """Initialize PMMoTo domain and subdomain classes and check for valid inputs."""
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

    if return_subdomain:
        _subdomain = Subdomain(rank=rank, decomposed_domain=pmmoto_decomposed_domain)
    elif verlet_domains == (0, 0, 0):
        _subdomain = PaddedSubdomain(
            rank=rank,
            decomposed_domain=pmmoto_decomposed_domain,
            pad=pad,
            reservoir_voxels=reservoir_voxels,
        )
    else:
        _subdomain = VerletSubdomain(
            rank=rank,
            decomposed_domain=pmmoto_decomposed_domain,
            pad=pad,
            reservoir_voxels=reservoir_voxels,
            verlet_domains=verlet_domains,
        )

    return _subdomain
