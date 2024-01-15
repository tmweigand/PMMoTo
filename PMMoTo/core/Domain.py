import numpy as np
from . import communication

class Domain(object):
    """
    Determine information for entire Domain
    """
    def __init__(self,
                 nodes = [1,1,1],
                 size_domain = [[0,1],[0,1],[0,1]],
                 subdomains = [1,1,1],
                 boundaries = [[0,0],[0,0],[0,0]],
                 inlet = [[0,0],[0,0],[0,0]],
                 outlet =[[0,0],[0,0],[0,0]]):
        self.nodes        = nodes
        self.size_domain  = size_domain
        self.boundaries   = boundaries
        self.subdomains   = subdomains
        self.inlet = inlet
        self.outlet = outlet
        self.dims = 3
        self.num_subdomains = np.prod(subdomains)
        self.global_map = -np.ones([sD+2 for sD in subdomains],dtype=np.int64)
        self.sub_nodes     = np.zeros([self.dims],dtype=np.uint64)
        self.rem_sub_nodes = np.zeros([self.dims],dtype=np.uint64)
        self.length_domain = np.zeros([self.dims])
        self.voxel = np.zeros([self.dims])
        self.periodic_check()
        self.input_checks()
        self.generate_global_map()

    def get_voxel_size(self):
        """
        Get domain length and voxel size
        """
        for n in range(0,self.dims):
            self.length_domain[n] = (self.size_domain[n,1]-self.size_domain[n,0])
            self.voxel[n] = self.length_domain[n]/self.nodes[n]

    def get_subdomain_nodes(self):
        """
        Calculate number of voxels in each subDomain
        """
        for n in range(0,self.dims):
            self.sub_nodes[n],self.rem_sub_nodes[n] = divmod(self.nodes[n],self.subdomains[n])

    def periodic_check(self):
        """
        Check if any external boundary is periodic
        """
        self.periodic = False
        for d_bound in self.boundaries:
            for n_bound in d_bound:
                if n_bound == 2:
                    self.periodic = True

    def input_checks(self):
        """
        Ensure Input Parameters are Valid
        """
        error = False

        ### Check Nodes are Positive
        for n in self.nodes:
            if n <= 0:
                error = True
                print("Error: Nodes must be positive integer!")
        
        ### Check subDomain Size
        for n in self.subdomains:
            if n <= 0:
                error = True
                print("Error: Number of Subdomains must be positive integer!")

        ### Check Boundaries and Boundary Pairs
        for d in self.boundaries:
            for n in d:
                if n < 0 or n > 2:
                    error = True
                    print("Error: Allowable Boundary IDs are (0) None (1) Walls (2) Periodic")            

        ### Check Inlet Condition
        n_sum = 0
        for d_in,d_bound in zip(self.inlet,self.boundaries):
            for n_in,n_bound in zip(d_in,d_bound):
                if n_in !=0:
                    n_sum = n_sum + 1
                    if n_bound != 0:
                        error = True
                        print("Error: Boundary must be type (0) None at Inlet")
        if n_sum > 1:
            error = True
            print("Error: Only 1 Inlet Allowed")

        ### Check Outlet Condition
        n_sum = 0
        for d_in,d_bound in zip(self.outlet,self.boundaries):
            for n_in,n_bound in zip(d_in,d_bound): 
                if n_in !=0:
                    n_sum = n_sum + 1
                    if n_bound != 0:
                        error = True
                        print("Error: Boundary must be type (0) None at Outlet")
        if n_sum > 1:
            error = True
            print("Error: Only 1 Outlet Allowed")

        if error:
          communication.raiseError()

    def generate_global_map(self):
        """
        Generate Domain lookup map. 
        -2: Wall Boundary Condition
        -1: No Assumption Boundary Condition
        >=0: proc_ID
        """

        self.global_map[1:-1,1:-1,1:-1] = np.arange(self.num_subdomains).reshape(self.subdomains)
        
        ### Set Boundarys of global SubDomain Map
        if self.boundaries[0][0] == 1:
            self.global_map[0,:,:] = -2
        if self.boundaries[0][1] == 1:
            self.global_map[-1,:,:] = -2
        if self.boundaries[1][0] == 1:
            self.global_map[:,0,:] = -2
        if self.boundaries[1][1] == 1:
            self.global_map[:,-1,:] = -2
        if self.boundaries[2][0] == 1:
            self.global_map[:,:,0] = -2
        if self.boundaries[2][1] == 1:
            self.global_map[:,:,-1] = -2

        if self.boundaries[0][0] == 2:
            self.global_map[0,:,:]  = self.global_map[-2,:,:]
            self.global_map[-1,:,:] = self.global_map[1,:,:]

        if self.boundaries[1][0] == 2:
            self.global_map[:,0,:]  = self.global_map[:,-2,:]
            self.global_map[:,-1,:] = self.global_map[:,1,:]

        if self.boundaries[2][0] == 2:
            self.global_map[:,:,0]  = self.global_map[:,:,-2]
            self.global_map[:,:,-1] = self.global_map[:,:,1]
    