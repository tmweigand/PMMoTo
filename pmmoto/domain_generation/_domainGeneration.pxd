from libcpp.vector cimport vector
from numpy cimport npy_double

cdef struct verlet_sphere:
  npy_double x 
  npy_double y 
  npy_double z 
  npy_double r 
