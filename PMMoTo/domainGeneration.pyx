#cython: cdivision=True
#cython: boundscheck=False
#cython: nonecheck=False
#cython: wraparound=False

import math
import numpy as np
cimport numpy as cnp
from libc.stdio cimport printf
cnp.import_array()


cdef int inAtom(double cx,double cy,double cz,double x,double y,double z,double r):
    cdef double re
    re = (cx - x)*(cx - x) + (cy - y)*(cy - y) + (cz - z)*(cz - z)
    if (re <= r): # already calculated 0.25*r*r
        return 0
    else:
        return 1

cdef double inAtomCA(double cx,double cy,double cz,double x,double y,double z,double r):
    cdef double re
    re = (cx - x)*(cx - x) + (cy - y)*(cy - y) + (cz - z)*(cz - z)
    if (re <= r): # already calculated 0.25*r*r
        return r


def domainGen( double[:] x, double[:] y, double[:] z, double[:,:] atom):

    cdef int NX = x.shape[0]
    cdef int NY = y.shape[0]
    cdef int NZ = z.shape[0]
    cdef int numObjects = atom.shape[1]

    cdef int i, j, k, c


    _grid = np.ones((NX, NY, NZ), dtype=np.uint8)
    cdef cnp.uint8_t [:,:,:] grid

    grid = _grid

    for i in range(0,NX):
      for j in range(0,NY):
        for k in range(0,NZ):
          c = 0
          while (grid[i,j,k] == 1 and c < numObjects):
              grid[i,j,k] = inAtom(atom[0,c],atom[1,c],atom[2,c],x[i],y[j],z[k],atom[3,c])
              c = c + 1

    return _grid

def domainGenCA( double[:] x, double[:] y, double[:] z, double[:,:] atom):

    cdef int NX = x.shape[0]
    cdef int NY = y.shape[0]
    cdef int NZ = z.shape[0]
    cdef int numObjects = atom.shape[1]

    cdef int i, j, k, c


    _grid = np.zeros((NX, NY, NZ), dtype=np.float64)
    cdef double[:,:,:] grid = _grid

    for i in range(0,NX):
      for j in range(0,NY):
        for k in range(0,NZ):
          c = 0
          #print(grid[i,j,k])
          while (grid[i,j,k] == 0 and c < numObjects):
              #print("A")
              grid[i,j,k] = inAtomCA(atom[0,c],atom[1,c],atom[2,c],x[i],y[j],z[k],atom[3,c])
              c = c + 1

    return _grid


def domainGenINK(double[:] x, double[:] y, double[:] z):

    cdef int NX = x.shape[0]
    cdef int NY = y.shape[0]
    cdef int NZ = z.shape[0]
    cdef int i, j, k
    cdef double r

    _grid = np.zeros((NX, NY, NZ), dtype=np.uint8)
    cdef cnp.uint8_t [:,:,:] grid

    grid = _grid

    for i in range(0,NX):
      for j in range(0,NY):
        for k in range(0,NZ):
          r = (0.01*math.cos(0.01*x[i]) + 0.5*math.sin(x[i]) + 0.75)
          if y[j]*y[j] + z[k]*z[k] <= r*r:
            grid[i,j,k] = 1

    return _grid

def domainGenINKCA(double[:] x, double[:] y, double[:] z):

    print("WARNING: THIS FXN IS HARD CODED FOR 8 SUBDOMAINS")

    cdef int NX = x.shape[0]
    cdef int NY = y.shape[0]
    cdef int NZ = z.shape[0]
    cdef int i, j, k
    cdef double r

    # Small distance for band
    cdef double epsilon = 14.0/((NX-1)*2)  ###change to one voxel length
    #print((NX-1)*2)
    #print("{:.6f}".format(epsilon))

    _grid = np.zeros((NX, NY, NZ), dtype=np.float64)
    cdef double[:,:,:] grid = _grid

    grid = _grid

    for i in range(0,NX):

      # First derivative
      dy_dx = -0.01**2 * np.sin(0.01 * x[i]) + 0.5 * np.cos(x[i])
      # Second derivative
      d2y_dx2 = -(0.000001) * np.cos(0.01 * x[i]) - 0.5 * np.sin(x[i])
      # Radius of curvature
      R = ((1 + dy_dx**2)**1.5) / np.abs(d2y_dx2)

      for j in range(0,NY):
        for k in range(0,NZ):
          r = (0.01*math.cos(0.01*x[i]) + 0.5*math.sin(x[i]) + 0.75)
          distance = y[j] * y[j] + z[k] * z[k]
          # Check if within epsilon distance from the boundary
          if abs(distance - r * r) < epsilon:
              grid[i, j, k] = R

    return _grid
