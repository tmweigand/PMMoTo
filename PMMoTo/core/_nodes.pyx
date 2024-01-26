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

def get_set_info(subdomain,
                cnp.uint64_t [:,:,:] grid,
                int n_labels,
                cnp.int64_t [:,:,:] loop_info,
                cnp.uint8_t [:] inlet,
                cnp.uint8_t [:] outlet):
    """
    Collect information for nodes to assign to sets
    This includes the nodes local_ID, boundary nodes global_ID,
    if it is an subdomain boundary/inlet/outlet set
    and all subdomain boundary IDs whether set is on boundary

    TODO: get boundary_ID in one call
    """

    cdef: 
        Py_ssize_t i,j,k
        int n_face,label
        vector[bool] inlets,outlets,boundary
        unordered_map[int, vector[Py_ssize_t]] nodes
        unordered_map[int, vector[Py_ssize_t]] b_nodes

        Py_ssize_t sx = grid.shape[0]
        Py_ssize_t sy = grid.shape[1]
        Py_ssize_t sz = grid.shape[2]

        int dx = subdomain.domain.nodes[0]
        int dy = subdomain.domain.nodes[1]
        int dz = subdomain.domain.nodes[2]

        int ix = subdomain.index_start[0]
        int iy = subdomain.index_start[1]
        int iz = subdomain.index_start[2]

        int num_faces = Orientation.num_faces
        uint8_t[:,:] boundary_features = np.zeros([n_labels,Orientation.num_neighbors],dtype=np.uint8)
        npy_int8[3] face_index
        vector[int] g_index
        Py_ssize_t g_ID,l_ID

    for n in range(0,n_labels):
        boundary.push_back(False)
        inlets.push_back(False)
        outlets.push_back(False)

    # Loop through faces
    for n_face in range(0,num_faces):
        loop = loop_info[n_face]
        face_index = _Orientation.face_index[n_face]
        if subdomain.boundary_ID[n_face] == 2:
            for i in range(loop[0][0],loop[0][1]):
                for j in range(loop[1][0],loop[1][1]):
                    for k in range(loop[2][0],loop[2][1]):
                        boundary[label] = True
                        label = grid[i,j,k]
                        boundary_index = _Orientation.get_boundary_index([i,j,k],[sx,sy,sz])
                        boundary_ID = _Orientation.get_boundary_ID(boundary_index)
                        boundary_features[label][boundary_ID] = True
                        g_ID = get_global_ID_periodic([i,j,k],[dx,dy,dz])
                        b_nodes[label].push_back(g_ID)
                        l_ID = get_local_ID([i,j,k],[sx,sy,sz])
                        nodes[label].push_back(l_ID)
        else:
            for i in range(loop[0][0],loop[0][1]):
                for j in range(loop[1][0],loop[1][1]):
                    for k in range(loop[2][0],loop[2][1]):
                        boundary[label] = True
                        label = grid[i,j,k]
                        boundary_index = _Orientation.get_boundary_index([i,j,k],[sx,sy,sz])
                        boundary_ID = _Orientation.get_boundary_ID(boundary_index)
                        boundary_features[label][boundary_ID] = True
                        g_ID = get_global_ID([i,j,k],[dx,dy,dz])
                        b_nodes[label].push_back(g_ID)
                        l_ID = get_local_ID([i,j,k],[sx,sy,sz])
                        nodes[label].push_back(l_ID)
        
        if inlet[n_face] and boundary[label]:
            inlets[label] = True
        if outlet[n_face] and boundary[label]:
            outlets[label] = True

    # Loop through interior
    loop = loop_info[num_faces]
    for i in range(loop[0][0],loop[0][1]):
        for j in range(loop[1][0],loop[1][1]):
            for k in range(loop[2][0],loop[2][1]):
                label = grid[i,j,k]
                l_ID = get_local_ID([i,j,k],[sx,sy,sz])
                nodes[label].push_back(l_ID)

    return [nodes,b_nodes,boundary,boundary_features,inlets,outlets]
        