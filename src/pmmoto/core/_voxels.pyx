# cython: profile=True
# cython: linetrace=False
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True

import numpy as np
cimport numpy as np

from libcpp.vector cimport vector
from libcpp cimport bool
from libcpp cimport tuple
from libcpp.unordered_map cimport unordered_map

from numpy cimport npy_intp, npy_int8, uint64_t, int64_t, uint8_t

# from . import orientation

__all__ = [
    "get_nearest_boundary_index",
    "_merge_matched_voxels",
    "_get_id",
    "gen_grid_to_label_map"
]

cdef extern from "_voxel.hpp":
    cdef void loop_through_slice(
        uint8_t *segids,
        uint8_t *out,
        const int n,
        const long int stride,
        bool forward
        ) nogil
        
    cdef long int _get_nearest_boundary_index(
        uint8_t *img, 
        uint8_t label, 
        const int n,
        const long int stride,
        bool forward) nogil

cdef struct match_test:
    npy_intp local_id
    npy_intp rank
    npy_intp neighbor_local_id
    npy_intp neighbor_rank
    npy_intp global_id


def get_start_indices(int dimension, dict location={0: 0, 1: 0, 2: 0}):
    """
    Helper function to compute the starting indices for slicing based on the given dimension
    and location.

    Parameters:
        dimension: The axis (0, 1, or 2) to extract the slice from.
        location: A dictionary specifying the fixed coordinates for the other dimensions.

    Returns:
        A list of start indices for slicing.
    """
    cdef int[3] start = [0, 0, 0]  # Initialize start indices
    for n, d in enumerate([0, 1, 2]):
        if d != dimension:
            start[n] = location.get(d, 0)  # Default to 0 if the key is not in location

    return start


def extract_1d_slice(uint8_t[:, :, :] img,
                     int dimension,
                     dict location={0: 0, 1: 0, 2: 0},
                     bool forward=True):
    """
    Extract a 1D slice from a 3D image array along a specified dimension.

    Parameters:
        img: 3D NumPy array of uint8 type.
        dimension: The axis (0, 1, or 2) to extract the slice from.
        location: A dictionary specifying the fixed coordinates for the other dimensions.
        forward: If True, iterate forward; otherwise, iterate backward.

    Returns:
        A 1D NumPy array containing the extracted slice.
    """

    # Determine the size of the output slice
    cdef size_t size = img.shape[dimension]
    
    # Initialize the output array
    cdef np.ndarray[uint8_t, ndim=1] output = np.zeros(size, dtype=np.uint8)
    
    # Get the start indices for slicing
    cdef int[3] start = get_start_indices(dimension, location)

    # Call the low-level loop function
    loop_through_slice(
        <uint8_t*>&img[start[0], start[1], start[2]],
        <uint8_t*>&output[0],
        size,
        img.strides[dimension],
        forward
    )

    return output


def determine_index_nearest_boundary(uint8_t[:, :, :] img,
                     uint8_t label,
                     int dimension,
                     dict location={0: 0, 1: 0, 2: 0},
                     bool forward=True):
    """
    Determine the index where img[index] = label from a 3D image array along a specified dimension.

    Parameters:
        img: 3D NumPy array of uint8 type.
        label: img[count] = label
        dimension: The axis (0, 1, or 2) to extract the slice from.
        location: A dictionary specifying the fixed coordinates for the other dimensions.
        forward: If True, iterate forward; otherwise, iterate backward.

    Returns:
        The min (or max if forward = False) index where img[count] = label
    """

    # Determine the size of the output slice
    cdef size_t size = img.shape[dimension]
    
    # Get the start indices for slicing
    cdef int[3] start = get_start_indices(dimension, location)

    # Call the low-level function to get the nearest boundary index
    index = _get_nearest_boundary_index(
        <uint8_t*>&img[start[0], start[1], start[2]],
        label,
        size,                                          
        img.strides[dimension],                       
        forward                                        
    )

    return index

def determine_index_nearest_boundary(uint8_t[:] img,
                     uint8_t label,
                     bool forward=True):
    """
    Determine the index where img[index] = label from a 1D image array along a specified dimension.

    Parameters:
        img: 3D NumPy array of uint8 type.
        label: img[count] = label
        dimension: The axis (0, 1, or 2) to extract the slice from.
        location: A dictionary specifying the fixed coordinates for the other dimensions.
        forward: If True, iterate forward; otherwise, iterate backward.

    Returns:
        The min (or max if forward = False) index where img[count] = label
    """

    # Determine the size of the output slice
    cdef size_t size = img.shape[0]
    
    # Call the low-level function to get the nearest boundary index
    index = _get_nearest_boundary_index(
        <uint8_t*>&img[0],
        label,
        size,                                          
        1,                       
        forward                                        
    )

    return index



def get_nearest_boundary_index(
    uint8_t[:,:,:] img,
    int dimension,
    bool direction,
    int label, 
    uint64_t[:,:] index_array
):
    """
    Determine the nearest boundary index of a given label in the specified dimension.

    Parameters:
        img (uint8_t[:,:,:]): The 3D image array.
        dimension (int): The dimension to iterate through (0, 1, or 2).
        direction (bool): The direction of iteration (True for forward, False for backward).
        label (int): The label to locate.
        index_array (uint64_t[:,:]): The output array to store indices for the label.

    Notes:
        - `index_array` should have dimensions matching the two non-iterated dimensions.
        - Iterates through the "face" of the 3D image for the specified dimension.
    """
    
    # Dimensions of the output index array
    cdef size_t sy = index_array.shape[0]  # Rows (corresponds to dim1)
    cdef size_t sx = index_array.shape[1]  # Columns (corresponds to dim2)

    # Identify the two dimensions to iterate over
    dims = {0, 1, 2}
    dims.remove(dimension)
    dim1, dim2 = dims

    location = {}


    # Iterate through the 2D "face" for the given dimension
    for y in range(sy):
        location[dim1] = y
        for x in range(sx):
            location[dim2] = x
            index_array[x,y] = determine_index_nearest_boundary(
                img = img,
                label = label, 
                dimension=dimension, 
                location = location,
                forward=direction)

    print(index_array)




def _merge_matched_voxels(all_match_data):
    """
    Connect all matched voxels from the entire domain.

    Args:
        all_match_data (list): List of dictionaries with matched sets by rank.

    Returns:
        tuple: (List of all matches with updated connections, total merged sets).
    """
    matches = {}
    local_counts = []
    boundary_counts = []
    local_global_map = {}

    # Flatten matched sets by rank and initialize `local_global_map`
    for matches_by_rank in all_match_data:
        local_counts.append(matches_by_rank['label_count'])
        del matches_by_rank['label_count']
        boundary_counts.append(len(matches_by_rank.keys()))

        for key,match in matches_by_rank.items():
            match['visited'] = False
            matches[key] = match
            local_global_map[key] = {}

    # Merge connected sets
    global_id = 0
    for key,match in matches.items():
        if match["visited"]:
            continue

        match["visited"] = True
        queue = [key]
        connections = []

        # Traverse connected matches
        while queue:
            current_id = queue.pop()
            current_match = matches[current_id]
            connections.append(current_id)

            for neighbor_id in current_match["neighbor"]:
                neighbor_match = matches[neighbor_id]
                if not neighbor_match["visited"]:
                    neighbor_match["visited"] = True
                    queue.append(neighbor_id)

        # Update global IDs for all connected matches
        for conn_id in connections:
            # all_matches[conn_id]["neighbor"] = connections
            local_global_map[conn_id]["global_id"] = global_id

        global_id += 1 if connections else 0


    for rank,_ in enumerate(local_counts):
        if rank == 0:
            local_global_map[rank] = global_id 
        else:
            local_labels = local_counts[rank-1] - boundary_counts[rank-1] - 1
            local_global_map[rank] = local_global_map[rank-1] + local_labels

    return local_global_map

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
    cdef uint64_t index_0, index_1, index_2

    # Use modulo to handle periodic boundary conditions
    index_0 = mod(x[0], voxels[0])
    index_1 = mod(x[1], voxels[1])
    index_2 = mod(x[2], voxels[2])

    cdef uint64_t id = index_0 * voxels[1] * voxels[2] + index_1 * voxels[2] + index_2

    return id

def get_boundary_data(
                np.uint64_t [:,:,:] grid,
                int n_labels,
                dict loop_dict,
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

    for loop in loop_dict.values():
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



def gen_grid_to_label_map(
                uint8_t [:,:,:] grid,
                uint64_t [:,:,:] labels):
    """
    This function provides a mapping between two images. For our purposes, 
    this would be the input (phases) and output (set ids) of a connected componented analysis. 
    """

    cdef: 
        Py_ssize_t i,j,k
        unordered_map[int,int] grid_to_label_map
        Py_ssize_t sx = grid.shape[0]
        Py_ssize_t sy = grid.shape[1]
        Py_ssize_t sz = grid.shape[2]
    
    for i in range(0,sx):
        for j in range(0,sy):
            for k in range(0,sz):
                grid_to_label_map[grid[i,j,k]] = labels[i,j,k]

    return grid_to_label_map


def _renumber_grid(uint64_t [:,:,:] grid, unordered_map[int, int] map):
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

def count_label_voxels(uint64_t [:,:,:] grid, unordered_map[int, int] map):
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