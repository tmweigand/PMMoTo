"""rdf.py"""

import numpy as np
from mpi4py import MPI
from pmmoto.domain_generation import _domain_generation

__all__ = ["generate_rdf", "generate_bounded_rdf"]


def generate_rdf(atom_map, rdf_data):
    """
    Generate Radial Distribution Functions from atom map and rdf data

    """
    rdf = {}
    for atom, data in rdf_data.items():
        _rdf = RDF(atom, atom_map[atom])
        _rdf.set_RDF(r=data[:, 1], g=data[:, 2])
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
    Radial distribution function class
    RDF = g(r) where:
      r is the radial distance and
      g is the free energy.

    An alternative is:
     G = -k_b*T ln g(r) / \sum_{r=0}^{r_f} g(r) where:
      k_b is the Boltzmann constant
      T is the temperature
      r_f is pairwise distance cutoff

    This is a common output for MD simulations.
    See: https://docs.lammps.org/compute_rdf.html

    This class reads in LAMMPS generated output and generates
    a new interpolated function.
    """

    def __init__(self, name, atom_id=None):
        self.name = name
        self.atom_id = atom_id
        self.r_data = None
        self.g_data = None

        self.bounds = None

    def set_RDF(self, r, g):
        """
        Set r and g(r) of radial distribution function
        """
        self.r_data = r
        self.g_data = g

    def g(self, r):
        """
        Given a r-value return g
        """
        return np.interp(r, self.r_data, self.g_data)

    def get_G(self, k_b=8.31446261815324, temp=300):
        """
        Galculate G(r)
        """
        _sum = np.sum(self.g_data)
        return -k_b * temp * np.log(self.g_data) / _sum


class Bounded_RDF(RDF):
    """
    Bounded radial distribution function class
    where the intepolated function is restricted to:
        g(r)> 0 : r : g(r) = 1

    This class finds those bounds and introduces a function
     r(g)
    """

    def __init__(self, rdf):
        self.rdf = rdf
        self.set_bounds(rdf.g_data)
        self.r_data, self.g_data = self.get_bounded_RDF_data(rdf, bounds)
        self.bounds = [None, None]

    def get_bounds(self, rdf_g_data):
        """
        Get the the r values of the bounded RDF such that:
            g(r) > 0 : r : g(r) = 1
        """
        self.bounds[0] = self.find_min_r(rdf_g_data)
        self.bounds[1] = self.find_max_r(1.1)

    def find_min_r(self, rdf_g_data):
        """
        Find the smallest r values from the RDF data such that:
             min r where g(r) > 0
        """
        r_loc = np.where([rdf_g_data == 0])[1][-1]

        return r_loc

    def find_max_r(self, g):
        """
        Find the smallest r value from the RDF data such that:
          all g(r) values are non-zero after r
        """
        find_r = g - self.rdf.g_data
        r_loc = np.where([find_r < 0])[1][0]

        return r_loc

    def get_bounded_RDF_data(self, rdf, bounds):
        """
        Set the bounds of the radial distribution function
        """
        r_data = rdf.r_data[bounds[0] : bounds[1]]
        g_data = rdf.g_data[bounds[0] : bounds[1]]

        return r_data, g_data

    def r(self, g):
        """
        Given a g-value return r
        Note! This only works for bounded RDFs
        """
        return np.interp(g, self.g_data, self.r_data)
