# cython: profile=True
# cython: linetrace=True
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True

import numpy as np
cimport numpy as np


from libcpp.vector cimport vector
from libcpp cimport bool
from libcpp cimport tuple
from libcpp.pair cimport pair
from libcpp.unordered_map cimport unordered_map
from numpy cimport int8_t, int16_t, int32_t, int64_t
from numpy cimport uint8_t, uint16_t, uint32_t, uint64_t
from numpy cimport npy_intp


__all__ = [
    "extract_1d_slice",
    "get_nearest_boundary_index_face",
    "get_nearest_boundary_index_face_2d",
    "determine_index_nearest_boundary_1d",
    "merge_matched_voxels",
    "renumber_img",
    "get_id",
    "gen_img_to_label_map"
]

ctypedef fused INT:
    int8_t
    int16_t
    int32_t
    int64_t

ctypedef fused UINT:
    uint8_t
    uint16_t
    uint32_t
    uint64_t
    unsigned long


ctypedef fused INTS:
    UINT
    INT

ctypedef fused INTS2:
    UINT
    INT


cdef struct match_test:
    npy_intp local_id
    npy_intp rank
    npy_intp neighbor_local_id
    npy_intp neighbor_rank
    npy_intp global_id

def normalized_strides(int dtype_size, int stride):
    """
    Compute the strides of a NumPy array normalized by the size of its data type.

    Parameters:
        dtype_size (int): Size of the data type in bytes
        stride (int): The stide for the dimension of interest

    Returns:
        tuple: Normalized strides, where each stride is in units of
            the array's dtype size.
    """
    return stride // dtype_size


def extract_1d_slice(
    uint8_t[:, :, :] img,
    int dimension,
    uint64_t[:] start,
    bool forward=True
):
    """
    Extract a 1D slice from a 3D image array along a specified dimension.

    Parameters:
        img: 3D NumPy array of uint8 type.
        dimension: The axis (0, 1, or 2) to extract the slice from.
        location: A dictionary specifying the fixed coordinates
            for the other dimensions.
        forward: If True, iterate forward; otherwise, iterate backward.

    Returns:
        A 1D NumPy array containing the extracted slice.
    """

    # Determine the size of the output slice
    cdef size_t _size = img.shape[dimension]

    # Initialize the output array
    cdef np.ndarray[uint8_t, ndim=1] output = np.zeros(_size, dtype=np.uint8)
    cdef int stride = normalized_strides(img.itemsize,img.strides[dimension])

    # Call the low-level loop function
    loop_through_slice(
        <uint8_t*>&img[start[0], start[1], start[2]],
        <uint8_t*>&output[0],
        _size,
        stride,
        forward
    )

    return output


def determine_index_nearest_boundary(
    uint8_t[:, :, :] img,
    uint8_t label,
    int dimension,
    uint64_t[:] start,
    uint64_t upper_skip = 0,
    bool forward = True
):
    """
    Determine the index where img[index] = label from a 3D image array
        along a specified dimension.

    The returned index is ALWAYS with respect to:
        img.shape[dimension]

    Parameters:
        img: 3D NumPy array of uint8 type.
        label: img[index] = label
        dimension: The axis (0, 1, or 2) to extract the slice from.
        start: The starting location of img
        upper_skip: The number of voxels to ignore along the given dimension
        forward: If True, iterate forward; otherwise, iterate backward.

    Returns:
        The min (or max if forward = False) index where img[count] = label
    """

    # Determine the size of the output slice
    cdef size_t end = img.shape[dimension] - start[dimension] - upper_skip
    cdef int stride = normalized_strides(img.itemsize,img.strides[dimension])

    # Call the low-level function to get the nearest boundary index
    return  _get_nearest_boundary_index(
        img=<uint8_t*>&img[start[0], start[1], start[2]],
        label=label,
        n=end,
        stride=stride,
        index_corrector=start[dimension],
        forward=forward
    )


def determine_index_nearest_boundary_2d(
    uint8_t[:, :] img,
    uint8_t label,
    int dimension,
    uint64_t[:] start,
    uint64_t upper_skip,
    bool forward=True
):
    """
    Determine the index where img[index] = label from a 2D image array
        along a specified dimension.

    The returned index is ALWAYS with respect to:
        img.shape[dimension]

    Returns:
        The min (or max if forward = False) index where img[count] = label
    """

    # Determine the size of the output slice
    cdef size_t end = img.shape[dimension] - start[dimension] - upper_skip

    # Get the start indices for slicing
    cdef int stride = normalized_strides(img.itemsize,img.strides[dimension])

    # Call the low-level function to get the nearest boundary index
    return _get_nearest_boundary_index(
        img=<uint8_t*>&img[start[0], start[1]],
        label=label,
        n=end,
        stride=stride,
        index_corrector=start[dimension],
        forward=forward
    )


def determine_index_nearest_boundary_1d(
    uint8_t[:] img,
    uint8_t label,
    uint64_t start = 0,
    uint64_t upper_skip = 0,
    bool forward=True
):
    """
    Determine the index where img[index] = label from a 1D image array
        along a specified dimension.

    Parameters:
        img: 3D NumPy array of uint8 type.
        label: img[count] = label
        dimension: The axis (0, 1, or 2) to extract the slice from.
        start: An integer specifying the starting index. 
               Assumed positive for forward=False. So counting from the right. 
        forward: If True, iterate forward; otherwise, iterate backward.

    Returns:
        The min (or max if forward = False) index where img[count] = label
    """

    # Determine the size of the output slice
    cdef size_t end = img.shape[0] - start - upper_skip

    # Call the low-level function to get the nearest boundary index
    return _get_nearest_boundary_index(
        img=<uint8_t*>&img[start],
        label=label,
        n=end,
        stride=1,
        index_corrector=start,
        forward=forward
    )


def get_nearest_boundary_index_face(
    uint8_t[:, :, :] img,
    int dimension,
    bool forward,
    int label,
    uint64_t lower_skip = 0,
    uint64_t upper_skip = 0
):
    """
    Determine the nearest boundary index of a given label in the specified dimension.
    
    The returned index is ALWAYS with respect to:
        img.shape[dimension]

    Parameters:
        img (uint8_t[:,:,:]): The 3D image array.
        dimension (int): The dimension to iterate through (0, 1, or 2).
        forward (bool): The direction of iteration
            (True for forward, False for backward).
        label (int): The label to locate.
        lower_pad: 

    """
    # Identify the two dimensions to iterate over
    dims = {0, 1, 2}
    dims.remove(dimension)
    dim1, dim2 = dims

    # Dimensions of the output index array
    cdef size_t sx = img.shape[dim1]
    cdef size_t sy = img.shape[dim2]
    cdef np.ndarray[int64_t, ndim=2] index_array = np.zeros((sx, sy), dtype=np.int64)

    # Location to start searching
    cdef np.ndarray[uint64_t, ndim=1] start = np.zeros(3, dtype=np.uint64)
    start[dimension] = lower_skip

    # Iterate through the 2D "face" for the given dimension
    
    for x in range(sx):
        start[dim1] = x
        for y in range(sy):
            start[dim2] = y
            index_array[x, y] = determine_index_nearest_boundary(
                img=img,
                label=label,
                dimension=dimension,
                start=start,
                upper_skip=upper_skip,
                forward=forward
            )

    return index_array


def get_nearest_boundary_index_face_2d(
    uint8_t[:, :] img,
    int dimension,
    bool forward,
    int label,
    uint64_t lower_skip = 0,
    uint64_t upper_skip = 0,
):
    """
    Determine the nearest boundary index of a given label in the specified dimension.

    Parameters:
        img (uint8_t[:,:]): The 2D image array.
        dimension (int): The dimension to iterate through (0 or 1).
        direction (bool): The direction of iteration
            (True for forward, False for backward).
        label (int): The label to locate.
        index_array (uint64_t[:]): The output array to store indices for the label.

    Notes:
        - `index_array` should have dimensions matching the two non-iterated dimensions.
        - Iterates through the "face" of the 3D image for the specified dimension.
    """
    # Identify the two dimensions to iterate over
    other_dim = [0, 1]
    other_dim.remove(dimension)
    other_dim = other_dim[0]

    # Dimensions of the output index array
    cdef size_t s = img.shape[other_dim]
    cdef np.ndarray[int64_t, ndim=1] index_array = np.zeros(s, dtype=np.int64)

    
    start = np.array([0, 0], dtype=np.uint64)
    start[dimension] = lower_skip

    for x in range(s):
        start[other_dim] = x
        index_array[x] = determine_index_nearest_boundary_2d(
            img = img,
            label = label,
            dimension = dimension,
            start = start,
            upper_skip=upper_skip,
            forward = forward)

    return index_array


def merge_matched_voxels(all_match_data):
    """
    Connect all matched voxels from the entire domain.

    Args:
        all_match_data (list): List of dictionaries with matched sets by rank.

    Returns:
        tuple: (List of all matches with updated connections, total merged sets).
    """
    matches = {}
    local_global_map = {}

    for matches_by_rank in all_match_data:
        
        for key, match in matches_by_rank.items():
            match['visited'] = False
            matches[key] = match
            local_global_map[key] = {}

    # Merge connected sets
    global_id = 1 # Zero are ignored in connected components
    for key, match in matches.items(): # key is (rank,local label)
        if match["visited"]:
            continue

        # BFS to find all connected components
        queue = [key]
        connections = []

        # Traverse connected matches
        while queue:
            current_id = queue.pop()
            if matches[current_id]['visited']:
                continue
            matches[current_id]['visited'] = True
            connections.append(current_id)

            for neighbor_id in matches[current_id]['neighbor_matches']:
                if not matches[neighbor_id]['visited']:
                    queue.append(neighbor_id)

        # Assign global IDs to connected components
        for conn_id in connections:
            local_global_map[conn_id]["global_id"] = global_id

        global_id += 1 if connections else 0

    return local_global_map, global_id


cpdef uint64_t get_id(tuple[int,...] index, tuple[int,...] voxels):
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
    index_0 = mod(index[0], voxels[0])
    index_1 = mod(index[1], voxels[1])
    index_2 = mod(index[2], voxels[2])

    cdef uint64_t id = index_0 * voxels[1] * voxels[2] + index_1 * voxels[2] + index_2

    return id


def get_boundary_data(
    np.uint64_t [:, :, :] img,
    int n_labels,
    dict loop_dict,
    tuple domain_voxels,
    tuple index
):
    """
    This function loops through the features of a subdomain and collects
    boundary information including where the label is on the boundary feature,
    and all voxels global ID
    """

    cdef:
        Py_ssize_t i, j, k
        int label
        vector[bool] boundary
        unordered_map[int, vector[uint64_t]] b_nodes

        uint64_t[:] _index = np.zeros(3, dtype=np.uint64)
        uint64_t[:] domain_nodes = np.array(domain_voxels, dtype = np.uint64)
    for _ in range(0, n_labels):
        boundary.push_back(False)

    for loop in loop_dict.values():
        for i in range(loop[0][0], loop[0][1]):
            for j in range(loop[1][0], loop[1][1]):
                for k in range(loop[2][0], loop[2][1]):
                    label = img[i, j, k]
                    boundary[label] = True
                    _index[0] = i+index[0]
                    _index[1] = j+index[1]
                    _index[2] = k+index[2]
                    b_nodes[label].push_back(
                        get_id(_index, domain_nodes)
                    )

    output = {
        'boundary_voxels': b_nodes,
        'boundary': boundary,
    }

    return output


def gen_img_to_label_map(
    INTS [:, :, :] img,
    INTS2 [:, :, :] labels
):
    """
    This function provides a mapping between two images. For our purposes,
    this would be the input (phases) and output (set ids) of a connected
    components analysis.
    """

    cdef:
        int i, j, k
        unordered_map[INTS2, INTS] img_to_label_map
        int sx = img.shape[0]
        int sy = img.shape[1]
        int sz = img.shape[2]

    for i in range(0, sx):
        for j in range(0, sy):
            for k in range(0, sz):
                img_to_label_map[labels[i, j, k]] = img[i, j, k]

    return img_to_label_map


def renumber_img(INTS[:, :, :] img, unordered_map[INTS, INTS] map):
    """
    Renumber a img in-place based on map.
    """
    cdef:
        Py_ssize_t i, j, k
        Py_ssize_t sx = img.shape[0]
        Py_ssize_t sy = img.shape[1]
        Py_ssize_t sz = img.shape[2]

    for i in range(0, sx):
        for j in range(0, sy):
            for k in range(0, sz):
                label = img[i, j, k]
                img[i, j, k] = map[label]

    return img


def count_label_voxels(INTS [:, :, :] img, unordered_map[int, int] map):
    """
    Count labels
    """
    cdef:
        Py_ssize_t i, j, k
        Py_ssize_t sx = img.shape[0]
        Py_ssize_t sy = img.shape[1]
        Py_ssize_t sz = img.shape[2]

    for i in range(0, sx):
        for j in range(0, sy):
            for k in range(0, sz):
                label = img[i, j, k]
                map[label] += 1

    return map


def find_unique_pairs(unsigned long[:,:] pairs):
    """
    Compute unique pairs using the C++ function.

    Parameters
    ----------
    pairs : numpy.ndarray
        A N x 2 array of integers.

    Returns
    -------
    numpy.ndarray
        A M x 2 array of unique pairs.
    """
    cdef:
        size_t nrows
        vector[pair[unsigned long,unsigned long]] result_cpp
        Py_ssize_t i, result_size

    nrows = pairs.shape[0]

    # Call the C++ function
    result_cpp = unique_pairs(<unsigned long*> &pairs[0, 0], nrows)
    result_size = result_cpp.size()
    result_np = np.zeros([result_size, 2], dtype=np.uint64)
    for i in range(result_size):
        result_np[i, 0] = result_cpp[i].first
        result_np[i, 1] = result_cpp[i].second

    return result_np


def process_matches_by_feature(
    unsigned long[:,:] matches,  # List of match tuples
    unique_matches: dict,  # Dictionary to store unique matches
    subdomain_rank: int,  # Subdomain rank
    feature_neighbor_rank: int  # Neighbor rank
):
    cdef:
        int i
        size_t n
        tuple match_tuple, neighbor_tuple
        list neighbors
    
    n = matches.shape[0]

    for i in range(n):
        # Access match elements directly as integers
        match_0 = matches[i, 0] #own
        match_1 = matches[i, 1] #neighbor
        
        # Create tuples
        match_tuple = (subdomain_rank, match_0)
        neighbor_tuple = (feature_neighbor_rank, match_1)

        # Check if match_tuple exists
        if match_tuple not in unique_matches:
            unique_matches[match_tuple] = {"neighbor_matches": [neighbor_tuple]}
        else:
            # Retrieve neighbors list and check membership
            if neighbor_tuple not in unique_matches[match_tuple]["neighbor_matches"]:
                unique_matches[match_tuple]["neighbor_matches"].append(neighbor_tuple)

    return unique_matches