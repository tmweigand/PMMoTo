"""domain_discretization.py"""
import numpy as np
from . import domain as pmmoto_domain

class DiscretizedDomain(pmmoto_domain.Domain):
    """
    Class for discretizing the domain 
    """
    def __init__(self,
                 domain,
                 subdomain_map,
                 nodes = (1,1,1)
                 ):
        self.nodes = nodes
        self.voxel = self.get_voxel_size(domain)
        self.sd_nodes,self.rem_nodes = self.get_subdomain_nodes(domain,subdomain_map)

    def get_voxel_size(self,domain):
        """
        Get domain length and voxel size
        """
        voxel = np.zeros([domain.dims])
        for n in range(0,domain.dims):
            voxel[n] = domain.length_domain[n]/self.nodes[n]

        return voxel

    def get_subdomain_nodes(self,domain,subdomain_map):
        """
        Calculate number of voxels in each subDomain
        """
        sub_nodes = np.zeros([domain.dims],dtype=np.uint64)
        rem_sub_nodes = np.zeros([domain.dims],dtype=np.uint64)
        for n in range(0,domain.dims):
            sub_nodes[n],rem_sub_nodes[n] = divmod(self.nodes[n],subdomain_map[n])

        return sub_nodes,rem_sub_nodes