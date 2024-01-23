import numpy as np
cimport numpy as cnp
cimport cython
from libcpp.vector cimport vector
from numpy cimport npy_intp, npy_int8, npy_uint8, uint64_t


cdef inline npy_uint8 get_boundary_ID(vector[npy_int8] boundary_ID):
    """
    Determine boundary ID
    Input: boundary_ID[3] corresponding to [x,y,z] and values of -1,0,1
    Output: boundary_ID
    """
    cdef int n
    cdef int[3] b_ID
    cdef int[3][3] params = [[0, 9, 18],[0, 3, 6],[0, 1, 2]]

    for n in range(0,3):
        if boundary_ID[n] < 0:
            b_ID[n] = params[n][0]
        elif boundary_ID[n] > 0:
            b_ID[n] = params[n][1]
        else:
            b_ID[n] = params[n][2]

    return b_ID[0] + b_ID[1] + b_ID[2]

cdef inline vector[npy_int8] get_boundary_index(vector[uint64_t] index,
                                         uint64_t[3] shape,
                                         npy_int8 b_x, npy_int8 b_y, npy_int8 b_z):
    """
    Determine the boundary index [i,j,k] -> [-1,0,1]
    Needed as only looping through faces but want edges and corners
    """
    cdef int n
    cdef npy_int8[3] boundary_index = [b_x,b_y,b_z]
    cdef vector[npy_int8] index_out
    for n in range(0,3):
        index_out.push_back(boundary_index[n])
        if index[n] < 2:
            index_out[n] = -1
        elif index[n] >= shape[n] - 1:
            index_out[n] = 1
    return index_out



cdef class cOrientation(object):
    cdef public int num_faces,num_edges,num_corners,num_neighbors
    cdef public int[26][5] directions
    cdef public int[6][4] face_info
    cdef public int[6][3] face_index
    def __cinit__(self):
        self.num_faces = 6
        self.num_edges = 12
        self.num_corners = 8
        self.num_neighbors = 26
        self.directions = [[-1,-1,-1,  0, 13],  #0
                           [-1,-1, 1,  1, 12],  #1
                           [-1,-1, 0,  2, 14],  #2
                           [-1, 1,-1,  3, 10],  #3
                           [-1, 1, 1,  4,  9],  #4
                           [-1, 1, 0,  5, 11],  #5
                           [-1, 0,-1,  6, 16],  #6
                           [-1, 0, 1,  7, 15],  #7
                           [-1, 0, 0,  8, 17],  #8
                           [ 1,-1,-1,  9,  4],  #9
                           [ 1,-1, 1, 10,  3],  #10
                           [ 1,-1, 0, 11,  5],  #11
                           [ 1, 1,-1, 12,  1],  #12
                           [ 1, 1, 1, 13,  0],  #13
                           [ 1, 1, 0, 14,  2],  #14
                           [ 1, 0,-1, 15,  7],  #15
                           [ 1, 0, 1, 16,  6],  #16
                           [ 1, 0, 0, 17,  8],  #17
                           [ 0,-1,-1, 18, 22],  #18
                           [ 0,-1, 1, 19, 21],  #19
                           [ 0,-1, 0, 20, 23],  #20
                           [ 0, 1,-1, 21, 19],  #21
                           [ 0, 1, 1, 22, 18],  #22
                           [ 0, 1, 0, 23, 20],  #23
                           [ 0, 0,-1, 24, 25],  #24
                           [ 0, 0, 1, 25, 24]]  #25
        self.face_info = [[0, 1, 2, 1],
                          [0, 1, 2,-1],
                          [1, 0, 2, 1],
                          [1, 0, 2,-1],
                          [2, 0, 1, 1],
                          [2, 0, 1,-1]]
        self.face_index = [[-1, 0, 0],
                        [ 1, 0, 0],
                        [ 0,-1, 0],
                        [ 0, 1, 0],
                        [ 0, 0,-1],
                        [ 0, 0, 1]]

    @cython.boundscheck(False)  # Deactivate bounds checking
    @cython.wraparound(False)   # Deactivate negative indexing.
    cpdef int getBoundaryIDReference(self,cnp.ndarray[cnp.int8_t, ndim=1] boundary_ID):
        """
        Determining boundary ID
        Input: boundary_ID[3] corresponding to [x,y,z] and values range from [-1,0,1]
        Output: boundary_ID
        """
        cdef int cI,cJ,cK
        cdef int i,j,k
        i = boundary_ID[0]
        j = boundary_ID[1]
        k = boundary_ID[2]

        if i < 0:
            cI = 0
        elif i > 0:
            cI = 9
        else:
            cI = 18

        if j < 0:
            cJ = 0
        elif j > 0:
            cJ = 3
        else:
            cJ = 6

        if k < 0:
            cK = 0
        elif k > 0:
            cK = 1
        else:
            cK = 2

        return cI+cJ+cK