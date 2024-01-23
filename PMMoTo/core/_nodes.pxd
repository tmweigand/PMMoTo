# cython: profile=True
# cython: linetrace=True
# cython: boundscheck=False
# cython: wraparound=False
from libcpp.vector cimport vector
from numpy cimport uint64_t,int64_t

cdef inline uint64_t get_local_ID(uint64_t[3] x, uint64_t[3] subdomain_nodes):
    """
    Determine the the global ID for a node
    """
    return x[0]*subdomain_nodes[1]*subdomain_nodes[2] + x[1]*subdomain_nodes[2] +  x[2]

cdef inline uint64_t get_global_ID(uint64_t[3] x, uint64_t[3] domain_nodes):
    """
    Determine the the global ID for a node
    """
    return x[0]*domain_nodes[1]*domain_nodes[2] +  x[1]*domain_nodes[2] +  x[2]

cdef inline vector[int64_t] get_global_index(uint64_t[3] x, int64_t[3] domain_nodes, int64_t[3] index_start):
    """
    Determine the global index [i,j,k]
    """
    cdef: 
        int n
        vector[int64_t] index

    for n in range(0,3):
        index.push_back(x[n] + index_start[n])
    return index

cdef inline vector[int64_t] get_global_index_periodic(uint64_t[3] x, int64_t[3] domain_nodes, int64_t[3] index_start):
    """
    Determine the global index [i,j,k]. Loop around if periodic so match. 
    """
    cdef: 
        int n
        vector[int64_t] index

    for n in range(0,3):
        index.push_back(x[n] + index_start[n])
        if index[n] >= domain_nodes[n]:
            index[n] = 0
        elif index[n] < 0:
            index[n] = domain_nodes[n] - 1
    return index