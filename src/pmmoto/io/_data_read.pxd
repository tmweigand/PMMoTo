from libcpp.string cimport string
from libcpp.vector cimport vector
from libcpp.map cimport map
from libcpp.pair cimport pair

cdef extern from "data_read.hpp":
    cdef struct LammpsData:
        vector[vector[double]] atom_positions
        vector[int] atom_types
        vector[vector[double]] domain_data
        double timestep

    cdef cppclass LammpsReader:
        @staticmethod
        LammpsData read_lammps_atoms(const string& filename)
        
        @staticmethod
        LammpsData read_lammps_atoms_with_map(const string& filename,
                                             const map[pair[int, double], int]& type_map)