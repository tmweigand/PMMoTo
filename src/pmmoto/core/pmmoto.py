from . import utils
from . import domain
from . import domain_decompose
from . import domain_discretization
from . import subdomain_padded

__all__ = [
    "initialize",
]


def initialize(
    box,
    subdomain_map,
    voxels,
    boundaries=((0, 0), (0, 0), (0, 0)),
    inlet=((0, 0), (0, 0), (0, 0)),
    outlet=((0, 0), (0, 0), (0, 0)),
    reservoir_voxels=0,
    rank=0,
    mpi_size=1,
):
    """
    Initialize PMMoTo domain and subdomain classes and check for valid inputs.
    """

    # utils.check_inputs(mpi_size, subdomain_map, voxels, boundaries, inlet, outlet)

    pmmoto_domain = domain.Domain(
        box=box, boundaries=boundaries, inlet=inlet, outlet=outlet
    )

    pmmoto_discretized_domain = domain_discretization.DiscretizedDomain.from_domain(
        domain=pmmoto_domain,
        voxels=voxels,
    )

    pmmoto_decomposed_domain = (
        domain_decompose.DecomposedDomain.from_discretized_domain(
            discretized_domain=pmmoto_discretized_domain,
            subdomain_map=subdomain_map,
        )
    )

    pmmoto_subdomain = pmmoto_decomposed_domain.initialize_subdomain(rank)

    padded_subdomain = subdomain_padded.PaddedSubdomain.from_subdomain(
        subdomain=pmmoto_subdomain, pad=(1, 1, 1), reservoir_voxels=reservoir_voxels
    )

    return padded_subdomain, pmmoto_decomposed_domain
