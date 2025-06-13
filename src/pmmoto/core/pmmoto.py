"""pmmoto.py"""

from __future__ import annotations
from typing import TypeVar
import numpy as np
from numpy.typing import NDArray

from .boundary_types import BoundaryType
from . import domain_decompose
from . import domain
from . import domain_discretization
from .subdomain import Subdomain
from .subdomain_padded import PaddedSubdomain
from .subdomain_verlet import VerletSubdomain
from . import utils

T = TypeVar("T", bound=np.generic)

__all__ = ["initialize", "deconstruct_grid"]


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


def deconstruct_grid(
    subdomain: Subdomain | PaddedSubdomain | VerletSubdomain,
    img: NDArray[T],
    subdomains: tuple[int, ...],
    rank: None | int = None,
    pad: tuple[int, ...] = (1, 1, 1),
    reservoir_voxels: int = 0,
) -> tuple[
    PaddedSubdomain | dict[int, PaddedSubdomain], NDArray[T] | dict[int, NDArray[T]]
]:
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
        subdomain_out_single = PaddedSubdomain(
            rank=rank,
            decomposed_domain=pmmoto_decomposed_domain,
            pad=pad,
            reservoir_voxels=reservoir_voxels,
        )
        local_img_single = utils.decompose_img(
            img=img,
            start=subdomain_out_single.start,
            shape=subdomain_out_single.voxels,
        )

        local_img_single = subdomain_out_single.set_wall_bcs(local_img_single)

        return subdomain_out_single, local_img_single

    else:
        subdomain_out: dict[int, PaddedSubdomain] = {}
        local_img: dict[int, NDArray[T]] = {}
        for n in range(0, num_procs):
            subdomain_out[n] = PaddedSubdomain(
                rank=n,
                decomposed_domain=pmmoto_decomposed_domain,
                pad=pad,
                reservoir_voxels=reservoir_voxels,
            )

            local_img[n] = utils.decompose_img(
                img=img,
                start=subdomain_out[n].start,
                shape=subdomain_out[n].voxels,
            )

            local_img[n] = subdomain_out[n].set_wall_bcs(local_img[n])

        return subdomain_out, local_img
