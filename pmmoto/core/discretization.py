"""discretization.py"""
import numpy as np
import domain 

class Discretized(domain.Domain):
    """
    Class for discretizing the domain 
    """
    def __init__(self,
                 domain,
                 subdomain,
                 nodes = (1,1,1)
                 ):
        self.nodes = nodes
        self.sd_nodes,self.rem_nodes = self.get_subdomain_nodes(domain,subdomain)

    def get_voxel_size(self,domain):
        """
        Get domain length and voxel size
        """
        voxel = np.zeros([domain.dims])
        for n in range(0,domain.dims):
            voxel[n] = domain.length_domain[n]/self.nodes[n]


    def get_subdomain_nodes(self,domain,subdomain):
        """
        Calculate number of voxels in each subDomain
        """
        sub_nodes = np.zeros([domain.dims],dtype=np.uint64)
        rem_sub_nodes = np.zeros([domain.dims],dtype=np.uint64)
        for n in range(0,domain.dims):
            sub_nodes[n],rem_sub_nodes[n] = divmod(self.nodes[n],subdomain.subdomains[n])

        return sub_nodes,rem_sub_nodes
    

    def get_coordinates(self, pad = None, get_coords = True, multiphase = False):
        """
        Determine actual coordinate information (x,y,z)
        If boundaryID and Domain.boundary == 0, buffer is not added
        Everywhere else a buffer is added
        Pad is also Reservoir Size for mulitPhase 
        """

        if pad is None:
            pad = self.buffer

        sd_size = [None,None]
        for n in range(self.domain.dims):
    
            self.nodes[n] += pad[n*2] + pad[n*2+1]
            self.index_start[n] -= pad[n*2]

            self.index_own_nodes[n*2] += pad[n*2]
            self.index_own_nodes[n*2+1] = self.index_own_nodes[n*2] +self.own_nodes[n]

            self.index_global[n*2] = self.index_start[n] + pad[n*2]
            self.index_global[n*2+1] = self.index_start[n] + self.nodes[n] - pad[n*2+1]

            if get_coords:
                vox = self.domain.voxel[n]
                d_size = self.domain.size_domain[n]
                self.coords[n] = np.zeros(self.nodes[n],dtype = np.double)
                sd_size[0] = vox/2 + d_size[0] + vox*(self.index_start[n])
                sd_size[1] = vox/2 + d_size[0] + vox*(self.index_start[n] + self.nodes[n] - 1)
                self.coords[n] = np.linspace(sd_size[0], sd_size[1], self.nodes[n] )
                
                self.size_subdomain[n] = sd_size[1] - sd_size[0]
                self.bounds[n] = [sd_size[0],sd_size[1]]

            ### Not Sure why I have this. Commenting out in case useful
            if multiphase:
                if pad[n*2] > 0:
                    self.domain.nodes[n] += pad[n*2]
                if pad[n*2+1] > 0:
                    self.domain.nodes[n] += pad[n*2+1]