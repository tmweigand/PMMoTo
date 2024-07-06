import os
import numpy as np
import matplotlib.pyplot as plt
from mpi4py import MPI
import pmmoto

def test_rdf():
    """
    Test the Radial Distributrion Functions
    """
    rdf_folder = './testDomains/lammps/atom_data/'
    atom_map,rdf_files = pmmoto.io.dataRead.read_rdf(rdf_folder)
    rdf = pmmoto.domain_generation.generate_rdf(atom_map,rdf_files)

    n = 1000
    test_r = np.linspace(0,5,n)
    g = np.zeros_like(test_r)

    for n,_r in enumerate(test_r):
        g[n] = rdf['C'].g(_r)

    # plt.plot(test_r,g)
    # plt.plot(rdf['C'].r_data,rdf['C'].g_data)
    # plt.show()

def test_bounded_rdf():
    """
    Test the Radial Distributrion Functions
    """
    rdf_folder = './testDomains/lammps/atom_data/'
    atom_map,rdf_files = pmmoto.io.dataRead.read_rdf(rdf_folder)
    rdf = pmmoto.domain_generation.generate_rdf(atom_map,rdf_files)
    bounded_rdf = pmmoto.domain_generation.generate_bounded_rdf(rdf)
    
    n = 1000
    test_g = np.linspace(0,1,n)
    r = np.zeros_like(test_g)

    for n,_g in enumerate(test_g):
        r[n] = bounded_rdf['C'].r(_g)

    plt.plot(r,test_g)
    plt.plot(rdf['C'].r_data,rdf['C'].g_data,'.')
    plt.show()

if __name__ == "__main__":
    test_rdf()
    test_bounded_rdf()
    MPI.Finalize()