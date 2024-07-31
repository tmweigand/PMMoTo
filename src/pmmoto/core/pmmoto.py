from . import utils
from . import domain
from . import domain_decompose
from . import domain_discretization

__all__ = [
    "initialize",
]


def initialize(
    box,
    subdomain_map,
    num_voxels,
    boundaries,
    inlet=None,
    outlet=None,
    rank=0,
    mpi_size=1,
):
    """
    Initialize PMMoTo domain and subdomain classes and check for valid inputs.
    """

    # utils.check_inputs(mpi_size, subdomain_map, num_voxels, boundaries, inlet, outlet)

    pmmoto_domain = domain.Domain(
        box=box,
        boundaries=boundaries,
        inlet=inlet,
        outlet=outlet,
    )

    pmmoto_discretized_domain = domain_discretization.DiscretizedDomain(
        box=box,
        boundaries=boundaries,
        inlet=inlet,
        outlet=outlet,
        num_voxels=num_voxels,
    )

    pmmoto_decomposed_domain = domain_decompose.DecomposedDomain(
        box=box,
        boundaries=boundaries,
        inlet=inlet,
        outlet=outlet,
        num_voxels=num_voxels,
        rank=rank,
        subdomain_map=subdomain_map,
    )

    return pmmoto_decomposed_domain
