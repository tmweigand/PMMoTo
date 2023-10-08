from libcpp.vector cimport vector
from numpy cimport npy_intp, npy_int8, npy_uint8, ndarray, npy_float32
from libcpp cimport bool
from libcpp.utility cimport pair

cdef struct boundary_set:
    npy_intp ID
    npy_intp proc_ID
    npy_intp path_ID
    npy_intp num_nodes
    npy_intp num_global_nodes
    bool inlet
    bool outlet
    vector[npy_intp] n_proc_ID
    vector[npy_intp] boundary_nodes
    vector[npy_intp] connected_sets


cdef struct matched_set:
    npy_intp ID
    npy_intp global_ID
    npy_intp proc_ID
    npy_intp path_ID
    bool inlet
    bool outlet
    vector[npy_intp] n_ID
    vector[npy_intp] n_proc_ID
    vector[npy_intp] n_path_ID
    vector[npy_intp] connected_sets
    vector[npy_intp] n_connected_sets

cdef struct vertex:
    npy_intp ID
    bool inlet
    bool outlet
    bool boundary
    bool trim
    vector[npy_intp] proc_ID
    vector[npy_intp] connected_sets
