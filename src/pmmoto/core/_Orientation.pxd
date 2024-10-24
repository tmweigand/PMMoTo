from libcpp.vector cimport vector
from numpy cimport npy_uint8, npy_int8, uint64_t, int64_t

cdef public npy_int8[6][3] face_index = [[-1, 0, 0],
                                  [ 1, 0, 0],
                                  [ 0,-1, 0],
                                  [ 0, 1, 0],
                                  [ 0, 0,-1],
                                  [ 0, 0, 1]]

cdef inline npy_uint8 get_boundary_ID(vector[npy_int8] boundary_ID):
    """
    Determine boundary ID
    Input: boundary_ID[3] corresponding to [x,y,z] and values of -1,0,1
    Output: boundary_ID
    """
    cdef int[3] b_ID
    cdef int[3][3] params = [[0, 9, 18],[0, 3, 6],[0, 1, 2]]

    # Convert boundary_ID values directly to their corresponding b_ID values
    b_ID[0] = params[0][boundary_ID[0] + 1]
    b_ID[1] = params[1][boundary_ID[1] + 1]
    b_ID[2] = params[2][boundary_ID[2] + 1]

    return b_ID[0] + b_ID[1] + b_ID[2]

cdef inline vector[npy_int8] get_boundary_index(vector[uint64_t] index,
                                         uint64_t[3] shape):
    """
    Determine the boundary index [i,j,k] -> [-1,0,1]
    Needed as only looping through faces but want edges and corners
    """
    cdef: 
        int n
        vector[npy_int8] index_out

    for n in range(0,3):
        index_out.push_back(0)
        if index[n] < 2:
            index_out[n] = -1
        elif index[n] >= shape[n] - 2:
            index_out[n] = 1
    return index_out

