"""subdomains.py"""
import numpy as np

class Subdomain:
    """
    Parallelization is via decomposition of domain into subdomains
    """
    def __init__(self,rank,index):
        self.rank = rank
        self.index = index
        self.boundary = False
        self.boundary_type = -np.ones([6],dtype = np.int8)

    def get_boundary_types(self,boundaries,subdomain_map):
        """
        Determine the boundary type. Internal boundaries are -1 
        """
        for n,_ in enumerate(self.index):
            
            if n == 0:
                self.boundary = True
                self.boundary_type = boundaries[n][0]

            if n == subdomain_map[n] - 1:
                self.boundary = True
                self.boundary_type = boundaries[n][1]
