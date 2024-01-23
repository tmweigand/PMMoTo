from libcpp.vector cimport vector
from numpy cimport npy_uint8, npy_int8, uint64_t

cdef public npy_int8[6][3] face_index = [[-1, 0, 0],
                                  [ 1, 0, 0],
                                  [ 0,-1, 0],
                                  [ 0, 1, 0],
                                  [ 0, 0,-1],
                                  [ 0, 0, 1]]

cdef inline npy_uint8 get_boundary_ID(vector[npy_int8] boundary_ID)

cdef inline vector[npy_int8] get_boundary_index(vector[uint64_t] index,
                                         uint64_t[3] shape,
                                         npy_int8 b_x, npy_int8 b_y, npy_int8 b_z)
