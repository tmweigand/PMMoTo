from . import utils
from . import domain
from . import domain_decompose
from . import domain_discretization
from . import subdomain
from . import subdomain_padded

__all__ = [
    "initialize",
]


def initialize(
    box,
    subdomains,
    voxels,
    boundary_types=((0, 0), (0, 0), (0, 0)),
    inlet=((0, 0), (0, 0), (0, 0)),
    outlet=((0, 0), (0, 0), (0, 0)),
    reservoir_voxels=0,
    rank=0,
    mpi_size=1,
    pad=(1, 1, 1),
):
    """
    Initialize PMMoTo domain and subdomain classes and check for valid inputs.
    """

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

    padded_subdomain = subdomain_padded.PaddedSubdomain(
        rank=rank,
        decomposed_domain=pmmoto_decomposed_domain,
        pad=pad,
        reservoir_voxels=reservoir_voxels,
    )

    return padded_subdomain, pmmoto_decomposed_domain
