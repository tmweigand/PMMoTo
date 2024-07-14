"""domain_discretization.py"""
import numpy as np
from . import domain as pmmoto_domain

class DiscretizedDomain(pmmoto_domain.Domain):
    """
    Class for discretizing the domain 
    """
    def __init__(self,
                 nodes = (1,1,1),
                 **kwargs
                 ):
        super().__init__(**kwargs)
        self.nodes = nodes
        self.voxel = self.get_voxel_size()

    def get_voxel_size(self):
        """
        Get domain length and voxel size
        """
        voxel = np.zeros([self.dims])
        for n in range(0,self.dims):
            voxel[n] = self.length_domain[n]/self.nodes[n]

        return voxel

    def get_subdomain_nodes(self,subdomain_map):
        """
        Calculate number of voxels in each subDomain
        """
        sub_nodes = np.zeros([self.dims],dtype=np.uint64)
        rem_sub_nodes = np.zeros([self.dims],dtype=np.uint64)
        for n in range(0,self.dims):
            sub_nodes[n],rem_sub_nodes[n] = divmod(self.nodes[n],subdomain_map[n])

        return sub_nodes,rem_sub_nodes