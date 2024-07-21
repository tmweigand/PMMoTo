"""rdf.py"""

import numpy as np
from mpi4py import MPI
from pmmoto.domain_generation import _domainGeneration
from pmmoto.core import communication
from pmmoto.core import utils
from pmmoto.core import porousMedia
from pmmoto.core import Orientation

__all__ = [
    "generate_rdf",
    "generate_bounded_rdf"
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

def generate_bounded_rdf(rdf):
    """
    Generate Bounded Radial Distribution Functions from RDF
            g(r)> 0 : r : g(r) = 1
    """
    b_rdf = {}
    for atom in rdf:
        b_rdf[atom] = Bounded_RDF(rdf[atom])
    return b_rdf

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


class Bounded_RDF(RDF):
    """
    Bounded Radial Distibution Class
    """
    def __init__(self,rdf):
        self.rdf = rdf
        self.r_data = None
        self.g_data = None
        self.bounds = [None,None]
        self.set_bounds()

    def set_bounds(self):
        """
        Set the bounds of the Bounded RDF
        """
        self.bounds[0] = self.find_min_r()
        self.bounds[1] = self.find_max_r(1.1)
        self.set_RDF()

    def find_min_r(self):
        """
        Find the smallest r values FROM DATA such that all g(r) values are non-zero after r
        """
        r_loc = np.where([self.rdf.g_data == 0])[1][-1]

        return r_loc

    def find_max_r(self,g):
        """
        Find the smallest r values FROM DATA such that all g(r) values are non-zero after r
        """
        find_r = g - self.rdf.g_data
        r_loc  = np.where([find_r < 0])[1][0]

        return r_loc

    def set_RDF(self):
        """
        Set the Bounds of the Radial Distribution Function
        """    
        self.r_data = self.rdf.r_data[self.bounds[0]:self.bounds[1]]
        self.g_data = self.rdf.g_data[self.bounds[0]:self.bounds[1]]

    def r(self,g):
        """
        Given a g-value return r 
        !!! Only Worls for Bounded RDF
        """
        return np.interp(g,self.g_data,self.r_data)