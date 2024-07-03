# cython: profile=True
# cython: linetrace=False
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True

# TODO: Clean up type defs
#       Periodic function

import numpy as np
cimport numpy as cnp
from mpi4py import MPI
comm = MPI.COMM_WORLD
from libcpp.vector cimport vector
from libcpp cimport bool
from libcpp.unordered_map cimport unordered_map
from libc.stdio cimport printf

from numpy cimport npy_intp, npy_int8, uint64_t, int64_t, uint8_t

from . import Orientation
from . import _Orientation
from . cimport _Orientation

ctypedef Py_ssize_t(*f_global_ID)(Py_ssize_t[3], int[3]) 

cdef f_global_ID functor_global_ID(periodic) except NULL:
    if periodic:
        return get_global_ID_periodic
    else:
        return get_global_ID

def get_boundary_set_info(subdomain,
                cnp.uint64_t [:,:,:] grid,
                int n_labels,
                cnp.int64_t [:,:,:] loop_info,
                cnp.uint8_t [:] inlet,
                cnp.uint8_t [:] outlet):
    """
    """

    cdef: 
        Py_ssize_t i,j,k
        int n_face,label
        vector[bool] inlets,outlets,boundary,boundary_face
        vector[Py_ssize_t] phase
        unordered_map[int, vector[Py_ssize_t]] b_nodes
        unordered_map[int, vector[Py_ssize_t]] nodes

        Py_ssize_t sx = grid.shape[0]
        Py_ssize_t sy = grid.shape[1]
        Py_ssize_t sz = grid.shape[2]

        int dx = subdomain.domain.nodes[0]
        int dy = subdomain.domain.nodes[1]
        int dz = subdomain.domain.nodes[2]

        int ix = subdomain.index_start[0]
        int iy = subdomain.index_start[1]
        int iz = subdomain.index_start[2]

        int[6] boundary_type = subdomain.boundary_type

        int num_faces = Orientation.num_faces
        uint8_t[:,:] boundary_features = np.zeros([n_labels,Orientation.num_neighbors],dtype=np.uint8)
        Py_ssize_t g_ID,l_ID

        f_global_ID global_ID_func

    for _ in range(0,n_labels):
        boundary.push_back(False)
        inlets.push_back(False)
        outlets.push_back(False)
        phase.push_back(-1)

    # Loop through faces
    for n_face in range(0,num_faces):
        
        for _ in range(0,n_labels):
            boundary_face.push_back(False)
        
        loop = loop_info[n_face]

        if boundary_type[n_face] == 2:
            global_ID_func = functor_global_ID(True)
        else:
            global_ID_func = functor_global_ID(False)
        
        for i in range(loop[0][0],loop[0][1]):
            for j in range(loop[1][0],loop[1][1]):
                for k in range(loop[2][0],loop[2][1]):
                    label = grid[i,j,k]
                    boundary[label] = True
                    boundary_face[label] = True
                    boundary_index = _Orientation.get_boundary_index([i,j,k],[sx,sy,sz])
                    boundary_ID = _Orientation.get_boundary_ID(boundary_index)
                    boundary_features[label][boundary_ID] = True
                    g_ID = global_ID_func([i+ix,j+iy,k+iz],[dx,dy,dz])
                    b_nodes[label].push_back(g_ID)
                    l_ID = get_local_ID([i,j,k],[sx,sy,sz])
                    phase[label] = l_ID
                    nodes[label].push_back(l_ID)
        
        for n in range(0,n_labels):
            if inlet[n_face] and boundary_face[n]:
                inlets[n] = True
            if outlet[n_face] and boundary_face[n]:
                outlets[n] = True


        boundary_face.clear()

    # Modify boundary_feature to add faces for edges, corners
    for n in range(0,n_labels):
        Orientation.add_faces(boundary_features[n])

    output = {
        'phase':phase,
        'boundary_nodes': b_nodes,
        'boundary': boundary,
        'boundary_features':boundary_features,
        'inlets':inlets,
        'outlets':outlets,
        'nodes':nodes
    }

    return output
        
def get_internal_set_info(
                cnp.uint64_t [:,:,:] grid,
                int n_labels,
                cnp.int64_t [:,:,:] loop_info):
    """
    """

    cdef: 
        Py_ssize_t i,j,k
        int label
        vector[Py_ssize_t] phase
        unordered_map[int, vector[Py_ssize_t]] nodes
        Py_ssize_t sx = grid.shape[0]
        Py_ssize_t sy = grid.shape[1]
        Py_ssize_t sz = grid.shape[2]
        int num_faces = Orientation.num_faces
        Py_ssize_t l_ID

    for _ in range(0,n_labels):
        phase.push_back(-1)

    # Loop through faces
    loop = loop_info[num_faces]
    for i in range(loop[0][0],loop[0][1]):
        for j in range(loop[1][0],loop[1][1]):
            for k in range(loop[2][0],loop[2][1]):
                label = grid[i,j,k]
                l_ID = get_local_ID([i,j,k],[sx,sy,sz])
                phase[label] = l_ID
                nodes[label].push_back(l_ID)

    output = {
        'phase':phase,
        'nodes': nodes,
    }

    return output

def renumber_grid(cnp.uint64_t [:,:,:] grid,cnp.int64_t [:] map):
    """
    Renumber a grid in-place based on map.
    """
    cdef: 
        Py_ssize_t i,j,k
        Py_ssize_t sx = grid.shape[0]
        Py_ssize_t sy = grid.shape[1]
        Py_ssize_t sz = grid.shape[2]

    for i in range(0,sx):
        for j in range(0,sy):
            for k in range(0,sz):
                label = grid[i,j,k]
                grid[i,j,k] = map[label]