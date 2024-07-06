"""rdf.py"""

import numpy as np
from mpi4py import MPI
from pmmoto.domain_generation import _domainGeneration
from pmmoto.core import communication
from pmmoto.core import utils
from pmmoto.core import porousMedia
from pmmoto.core import Orientation

__all__ = [
    "generate_rdf"
]

def generate_rdf(atom_map,rdf_files):
    """
    Generate Radial Distribution Functions from atom and data. 

    """
    rdf = {}
    for atom,file in zip(atom_map,rdf_files):
        _rdf = RDF(atom,atom_map[atom])
        data = np.genfromtxt(file,delimiter=',',skip_header=1)
        _rdf.set_RDF(r = data[:,1],g = data[:,2])
        rdf[atom] = _rdf
    return rdf


class RDF:
    """
    Radial Distribution Function Class
    """
    def __init__(self,name,ID):
        self.name = name
        self.ID = ID
        self.r_data = None
        self.g_data = None
        self.G = None
        self.bounds = None

    def set_RDF(self,r,g):
        """
        Set r and g(r) of radial distributrion function
        """
        self.r_data = r
        self.g_data = g

    def g(self,r):
        """
        Given a r-value return g
        """
        return np.interp(r,self.r_data,self.g_data)


    def get_G(self):
        """
        Galculate G(r)
        """
        kb = 8.31446261815324
        T = 300
        _sum = np.sum(self.g_data)
        self.G = -kb*T*np.log(self.g_data)/_sum
