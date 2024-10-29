# cython: profile=True
# cython: linetrace=False
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True

import numpy as np
cimport numpy as cnp
from mpi4py import MPI
comm = MPI.COMM_WORLD
from libcpp.vector cimport vector
from libcpp cimport bool
from libcpp cimport tuple
from libcpp.unordered_map cimport unordered_map
from libc.stdio cimport printf

from numpy cimport npy_intp, npy_int8, uint64_t, int64_t, uint8_t

from . import orientation
# from . import _Orientation
# from . cimport _Orientation

# __all__ = [
#     "_get_id"
# ]


def match_boundary_voxels(own_data,neighbor_data):
    """
    Match the boundary voxels for each feature based on global voxel ID
    """
    for key, voxels in own_data["boundary_voxels"].items():
        for n_key,n_voxels in neighbor_data["boundary_voxels"].items():
            print(_match_boundary_voxels(voxels,n_voxels))
    



cpdef uint64_t get_id(int64_t[:] x, uint64_t[:] voxels):
    """
    Determine the ID for a voxel.
    Input:
        - x: 3D index of the voxel (x, y, z)
        - voxels: Size of the domain (number of voxels in each dimension)
    Output:
        - Global or local ID of the voxel.
    Periodic boundary conditions are applied by using modulo arithmetic.
    """
    cdef Py_ssize_t index_0, index_1, index_2

    # Use modulo to handle periodic boundary conditions
    index_0 = x[0] % voxels[0]
    index_1 = x[1] % voxels[1]
    index_2 = x[2] % voxels[2]

    cdef uint64_t id = index_0 * voxels[1] * voxels[2] + index_1 * voxels[2] + index_2

    return id



# ctypedef Py_ssize_t(*f_global_ID)(Py_ssize_t[3], int[3]) 

def get_boundary_data(
                cnp.uint64_t [:,:,:] grid,
                int n_labels,
                cnp.uint64_t [:,:] loop,
                tuple domain_voxels,
                tuple index
                ):
    """
    This function loops through the features of a subdomain and collects 
    boundary information including whethere the label is on the boundary feature, 
    and all voxels global ID 
    """

    cdef: 
        Py_ssize_t i,j,k
        int label
        vector[bool] boundary
        unordered_map[int, vector[Py_ssize_t]] b_nodes

        int64_t[:] _index = np.zeros(3,dtype=np.int64)
        uint64_t[:] domain_nodes = np.array(domain_voxels,dtype = np.uint64)
    for _ in range(0,n_labels):
        boundary.push_back(False)

    for i in range(loop[0][0],loop[0][1]):
        for j in range(loop[1][0],loop[1][1]):
            for k in range(loop[2][0],loop[2][1]):
                label = grid[i,j,k]
                boundary[label] = True
                _index[0] = i+index[0]
                _index[1] = j+index[1]
                _index[2] = k+index[2]
                b_nodes[label].push_back(
                    get_id(_index,domain_nodes)
                    )

    

    output = {
        'boundary_voxels': b_nodes,
        'boundary': boundary,
    }

    return output



# def get_boundary_set_info(subdomain,
#                 cnp.uint64_t [:,:,:] grid,
#                 int n_labels,
#                 # unordered_map[int, int] phase_map,
#                 cnp.int64_t [:,:,:] loop_info,
#                 # cnp.uint8_t [:,:] inlet,
#                 # cnp.uint8_t [:,:] outlet
#                 ):
#     """
#     """

#     cdef: 
#         Py_ssize_t i,j,k
#         int n_face,label
#         # vector[bool] inlets,outlets,boundary_face
#         vector[bool] boundary
#         # vector[Py_ssize_t] phase
#         unordered_map[int, vector[Py_ssize_t]] b_nodes
#         # unordered_map[int, vector[Py_ssize_t]] nodes

#         Py_ssize_t sx = grid.shape[0]
#         Py_ssize_t sy = grid.shape[1]
#         Py_ssize_t sz = grid.shape[2]

#         uint64_t[:] domain_nodes = np.array(subdomain.domain_voxels,dtype = np.uint64)
#         int64_t[:] index = np.zeros(3,dtype=np.int64)
#         # int dx = subdomain.domain_voxels[0]
#         # int dy = subdomain.domain_voxels[1]
#         # int dz = subdomain.domain_voxels[2]

#         int ix = subdomain.start[0]
#         int iy = subdomain.start[1]
#         int iz = subdomain.start[2]

#         # int[6] boundary_type = subdomain.boundaries

#         int num_faces = orientation.num_faces
#         uint8_t[:,:] boundary_features = np.zeros([n_labels,orientation.num_neighbors],dtype=np.uint8)
#         # Py_ssize_t g_ID,l_ID

#         # f_global_ID global_ID_func

#     index = np.zeros(3,dtype=np.int64)
#     domain_nodes = np.array(subdomain.domain_voxels,dtype = np.uint64)

#     for _ in range(0,n_labels):
#         boundary.push_back(False)
#         # inlets.push_back(False)
#         # outlets.push_back(False)
#         # phase.push_back(-1)

#     # Loop through faces
#     for n_face in range(0,num_faces):
        
#         # for _ in range(0,n_labels):
#         #     boundary_face.push_back(False)
        
#         loop = loop_info[n_face]

#         for i in range(loop[0][0],loop[0][1]):
#             for j in range(loop[1][0],loop[1][1]):
#                 for k in range(loop[2][0],loop[2][1]):
#                     label = grid[i,j,k]
#                     boundary[label] = True
#                     # boundary_face[label] = True
#                     # boundary_index = _Orientation.get_boundary_index([i,j,k],[sx,sy,sz])
#                     boundary_ID = _Orientation.get_boundary_ID(
#                         _Orientation.get_boundary_index([i,j,k],[sx,sy,sz])
#                         )
#                     boundary_features[label][boundary_ID] = True
#                     index[0] = i+ix
#                     index[1] = j+iy
#                     index[2] = k+iz
#                     b_nodes[label].push_back(
#                         get_id(index,domain_nodes)
#                         )
#                     # l_ID = get_id([i,j,k],[sx,sy,sz])
#                     # phase[label] = l_ID
#                     # nodes[label].push_back(l_ID)
        
#         # for n in range(0,n_labels):
#         #     if inlet[phase_map[n]][n_face] and boundary_face[n]:
#         #         inlets[n] = True
#         #     if outlet[phase_map[n]][n_face] and boundary_face[n]:
#         #         outlets[n] = True


#         # boundary_face.clear()

#     # Modify boundary_feature to add faces for edges, corners
#     for n in range(0,n_labels):
#         orientation.add_faces(boundary_features[n])

#     output = {
#         # 'phase':phase,
#         'boundary_nodes': b_nodes,
#         'boundary': boundary,
#         'boundary_features':boundary_features,
#         # 'inlets':inlets,
#         # 'outlets':outlets,
#         # 'nodes':nodes
#     }

#     return output


# def get_boundary_set_info(subdomain,
#                           cnp.uint64_t [:,:,:] grid,
#                           int n_labels,
#                           cnp.int64_t [:,:,:] loop_info):
#     """
#     Optimized version of get_boundary_set_info for improved performance.
#     """

#     cdef: 
#         Py_ssize_t i, j, k, label
#         int n_face, boundary_ID
#         bint boundary_status
#         Py_ssize_t sx = grid.shape[0]
#         Py_ssize_t sy = grid.shape[1]
#         Py_ssize_t sz = grid.shape[2]

#         int dx = subdomain.domain_voxels[0]
#         int dy = subdomain.domain_voxels[1]
#         int dz = subdomain.domain_voxels[2]

#         int ix = subdomain.start[0]
#         int iy = subdomain.start[1]
#         int iz = subdomain.start[2]

#         int num_faces = orientation.num_faces

#         # Preallocate boundary status and boundary features arrays
#         bint[:] boundary = np.zeros(n_labels, dtype=np.bool_)
#         uint8_t[:, :] boundary_features = np.zeros([n_labels, orientation.num_neighbors], dtype=np.uint8)

#         # Using a dictionary to store boundary nodes
#         unordered_map[int, vector[Py_ssize_t]] b_nodes

#     # Loop through faces
#     for n_face in range(num_faces):
#         loop = loop_info[n_face]
        
#         # Process the loop ranges for each dimension
#         for i in range(loop[0, 0], loop[0, 1]):
#             for j in range(loop[1, 0], loop[1, 1]):
#                 for k in range(loop[2, 0], loop[2, 1]):
#                     label = grid[i, j, k]
#                     boundary[label] = True
                    
#                     # Get boundary ID and boundary features
#                     boundary_ID = _Orientation.get_boundary_ID(
#                         _Orientation.get_boundary_index([i, j, k], [sx, sy, sz])
#                     )
#                     boundary_features[label, boundary_ID] = True
                    
#                     # Add boundary node
#                     b_nodes[label].push_back(
#                         get_id([i + ix, j + iy, k + iz], [dx, dy, dz])
#                     )

#     # Modify boundary_features to add faces for edges, corners
#     for n in range(n_labels):
#         orientation.add_faces(boundary_features[n])

#     # Output results in a dictionary
#     output = {
#         'boundary_nodes': b_nodes,
#         'boundary': boundary,
#         'boundary_features': boundary_features
#     }

#     return output



# def get_internal_set_info(
#                 cnp.uint64_t [:,:,:] grid,
#                 int n_labels,
#                 cnp.int64_t [:,:,:] loop_info):
#     """
#     """

#     cdef: 
#         Py_ssize_t i,j,k
#         int label
#         vector[Py_ssize_t] phase
#         unordered_map[int, vector[Py_ssize_t]] nodes
#         Py_ssize_t sx = grid.shape[0]
#         Py_ssize_t sy = grid.shape[1]
#         Py_ssize_t sz = grid.shape[2]
#         int num_faces = orientation.num_faces
#         Py_ssize_t l_ID

#     for _ in range(0,n_labels):
#         phase.push_back(-1)

#     # Loop through faces
#     loop = loop_info[num_faces]
#     for i in range(loop[0][0],loop[0][1]):
#         for j in range(loop[1][0],loop[1][1]):
#             for k in range(loop[2][0],loop[2][1]):
#                 label = grid[i,j,k]
#                 l_ID = get_id([i,j,k],[sx,sy,sz])
#                 phase[label] = l_ID
#                 nodes[label].push_back(l_ID)

#     output = {
#         'phase':phase,
#         'nodes': nodes,
#     }

#     return output

# def map_sets_to_phases(
#                 cnp.uint8_t [:,:,:] phases,
#                 cnp.uint64_t [:,:,:] sets):
# """
# This function provides a mapping between two images. For our purposes, this would be the input (phases) and output (set ids) of a connected componented analysis. 
# """

#     cdef: 
#         Py_ssize_t i,j,k
#         int label
#         unordered_map[int,int] phase_label
#         Py_ssize_t sx = phases.shape[0]
#         Py_ssize_t sy = phases.shape[1]
#         Py_ssize_t sz = phases.shape[2]
    
#     # Loop through faces
#     for i in range(0,sx):
#         for j in range(0,sy):
#             for k in range(0,sz):
#                 label = sets[i,j,k]
#                 phase_label[label] = phases[i,j,k]

    # return phase_label


def renumber_grid(cnp.uint64_t [:,:,:] grid, unordered_map[int, int] map):
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

def count_label_voxels(cnp.uint64_t [:,:,:] grid, unordered_map[int, int] map):
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
                map[label] += 1

    return map