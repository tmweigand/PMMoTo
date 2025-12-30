from libcpp.string cimport string
from libcpp.vector cimport vector
from libcpp.map cimport map
from libcpp.pair cimport pair
from libc.stdint cimport uint64_t,uint8_t

ctypedef map[pair[int, double], int] AtomIdMap

cdef class AtomIdMapWrapper:
    cdef AtomIdMap* _c_map_ptr

cdef extern from "data_read.hpp":

    cdef struct LammpsData:
        vector[double] atom_positions
        vector[uint64_t] atom_ids
        vector[uint8_t] atom_types
        vector[vector[double]] domain_data
        double timestep

    cdef cppclass LammpsReader:
        @staticmethod
        LammpsData read_lammps_atoms(const string& filename, const AtomIdMap* id_map)
