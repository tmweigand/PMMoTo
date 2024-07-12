from . import utils
from . import domain
from . import Subdomain

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

    subdomain = Subdomain.Subdomain(domain = domain, ID = rank, subdomains = subdomains)
    subdomain.get_info()
    subdomain.gather_cube_info()

    return subdomain
