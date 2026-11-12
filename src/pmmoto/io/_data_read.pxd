from libcpp.string cimport string
from libcpp.vector cimport vector
from libcpp cimport bool
from libc.stdint cimport int64_t

cdef extern from "data_read.hpp":

    cdef struct LammpsColumnSpec:
        string name
        bool is_int

    cdef struct LammpsColumn:
        string name
        bool is_int
        vector[int64_t] int_values
        vector[double] double_values

    cdef struct LammpsData:
        vector[LammpsColumn] data_columns
        vector[vector[double]] domain_data
        double timestep

    cdef cppclass LammpsReader:
        @staticmethod
        LammpsData read_lammps_atoms(const string& filename, const vector[LammpsColumnSpec]& specs) except +
