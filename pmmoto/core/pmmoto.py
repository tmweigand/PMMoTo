from . import utils
from . import domain
from . import subdomain

__all__ = [
    "initialize",
    ]

def initialize(rank,mpi_size,subdomains,nodes,boundaries,inlet = None,outlet = None):
    """
    Initialize PMMoTo domain and subdomain classes and check for valid inputs. 
    """

    utils.check_inputs(mpi_size,subdomains,nodes,boundaries,inlet,outlet)

    pmmoto_domain = domain.Domain(nodes = nodes,
                           subdomains = subdomains,
                           boundaries = boundaries,
                           inlet = inlet,
                           outlet = outlet)
    
    pmmoto_domain.get_subdomain_nodes()

    pmmoto_subdomain = subdomain.Subdomain(ID = rank, subdomains = subdomains)
    pmmoto_subdomain.get_info()
    pmmoto_subdomain.gather_cube_info()

    return pmmoto_subdomain
