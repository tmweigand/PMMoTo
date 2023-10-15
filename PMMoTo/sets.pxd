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
    vector[npy_intp] n_proc_ID
    vector[npy_intp] boundary_nodes

cdef struct matched_set:
    npy_intp ID
    npy_intp global_ID
    npy_intp proc_ID
    bool inlet
    bool outlet
    vector[npy_intp] n_ID
    vector[npy_intp] n_proc_ID

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

    b_set.ID = set.localID
    b_set.proc_ID = set.proc_ID
    b_set.n_proc_ID = set.neighborProcID
    b_set.num_nodes = set.numNodes
    b_set.inlet = set.inlet
    b_set.outlet = set.outlet
    b_set.boundary_nodes = set.boundaryNodes

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

cdef inline int count_matched_nodes(vector[npy_intp] list1, vector[npy_intp] list2):
    cdef int count = 0
    for l in list1:
        if (binary_search(list2.begin(), list2.end(), l)):
            count += 1
    return count


cdef inline bool match_boundary_nodes(vector[npy_intp] list1, vector[npy_intp] list2):
    """
    Input: Two Sorted Lists
    Output: Bool of at least one shared element
    """
    cdef bool match =  False
    for l in list1:
        if (binary_search(list2.begin(), list2.end(), l)):
            match = True
            break
    return match