"""decomposed_domain.py"""
import numpy as np
from . import subdomain


class DecomposedDomain(subdomain.Subdomain):
    """
    Collection of subdomains
    """
    def __init__(self,rank,subdomain_map = (1,1,1)):
        self.rank = rank
        self.subdomain_map = subdomain_map
        self.num_subdomains = np.prod(subdomain_map)
        self.map = self.gen_subdomain_map()
        self.subdomains = self.initialize_subdomains()
        

    def gen_subdomain_map(self):
        """
        Generate the mapping of the subdomains in the domain
        """
        return np.arange(self.num_subdomains).reshape(self.subdomain_map)


    def initialize_subdomains(self):
        """
        Innitialize the subdomains
        """
        subdomains = []
        for sd_id in self.map.flatten():
            sd_index = np.unravel_index(sd_id,self.map.shape)
            subdomains.append(
                subdomain.Subdomain(sd_id,sd_index)
                )
        
        return subdomains
