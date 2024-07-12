"""subdomain_discretization.py"""
import numpy as np
import subdomain 

class Discretized(subdomain.Domain):
    """
    Class for discretizing the subdomain 
    """
    def __init__(self,
                 domain,
                 subdomain,
                 nodes = (1,1,1)
                 ):
        self.nodes = nodes
        self.sd_nodes,self.rem_nodes = self.get_subdomain_nodes(domain,subdomain)