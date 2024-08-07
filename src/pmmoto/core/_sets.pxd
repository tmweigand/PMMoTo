from libcpp.vector cimport vector
from numpy cimport npy_intp, npy_int8, npy_uint8, ndarray, npy_float32
from libcpp cimport bool
from libcpp.utility cimport pair
from libcpp.algorithm cimport binary_search

cdef struct boundary_set:
    npy_intp ID
    npy_intp proc_ID
    npy_intp num_nodes
    npy_intp num_global_nodes
    bool inlet
    bool outlet
    vector[npy_intp] nProcID
    vector[npy_intp] boundary_nodes

cdef struct matched_set:
    npy_intp ID
    npy_intp global_ID
    npy_intp proc_ID
    bool inlet
    bool outlet
    vector[npy_intp] n_ID
    vector[npy_intp] nProcID

cdef struct vertex:
    npy_intp ID
    bool inlet
    bool outlet
    bool boundary
    bool trim
    vector[npy_intp] proc_ID


cdef inline boundary_set c_convert_boundary_set(set):
    """
    Convert Python Set Object to C++ struct
    """
    cdef boundary_set b_set

    b_set.ID = set.local_ID
    b_set.proc_ID = set.proc_ID
    b_set.nProcID = set.subdomain_data.n_procs
    b_set.num_nodes = set.node_data.num_nodes
    b_set.inlet = set.subdomain_data.inlet
    b_set.outlet = set.subdomain_data.outlet
    b_set.boundary_nodes = set.node_data.boundary_nodes

    return b_set

cdef inline vertex c_convert_vertex(set):
    """
    Convert Python Set Object to C++ struct
    """
    cdef vertex n_set

    n_set.ID = set.localID
    n_set.inlet = set.inlet
    n_set.outlet = set.outlet
    n_set.boundary = set.boundary
    n_set.trim = set.trim
    n_set.proc_ID.push_back(set.proc_ID)

    return n_set
