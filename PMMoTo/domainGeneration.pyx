#cython: cdivision=True
#cython: boundscheck=False
#cython: nonecheck=False
#cython: wraparound=False

import math
import numpy as np
cimport numpy as cnp
from libc.stdio cimport printf
from libc.math cimport sqrt
cnp.import_array()


cdef int inAtom(double cx,double cy,double cz,double x,double y,double z,double r):
    cdef double re
    re = (cx - x)*(cx - x) + (cy - y)*(cy - y) + (cz - z)*(cz - z)
    if (re <= r): # already calculated 0.25*r*r
        return 0
    else:
        return 1

cdef double forceLJ(double r, double sig, double eps):
    cdef double A, B, sigr
    sigr = sig/r
    A = sigr * sigr * sigr * sigr * sigr * sigr
    B = A * A
    return 4 * eps *(-12*B/r + 6*A/r)


cdef double forceC(double r, double q1, double q2):
    cdef double C 
    C = 1 / 4 / 3.141592653589793238462 / 55.26349406 * 10000 * 23.060548
    return -C * q1 * q2 / r / r

cdef double forceEval(double cx,double cy,double cz,double x,double y,double z,double sig,double eps,double q1,double q2):
    cdef double r
    r = sqrt((cx - x)*(cx - x) + (cy - y)*(cy - y) + (cz - z)*(cz - z))
    return forceLJ(r,sig,eps)


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


def domainGenForce(double[:] x, double[:] y, double[:] z, double[:,:] atom, double q2):

    cdef int NX = x.shape[0]
    cdef int NY = y.shape[0]
    cdef int NZ = z.shape[0]
    cdef int numObjects = atom.shape[1]

    cdef int i, j, k, c
    cdef double force


    _grid = np.ones((NX, NY, NZ), dtype=np.uint8)
    cdef cnp.uint8_t [:,:,:] grid

    grid = _grid

    for i in range(0,NX):
      for j in range(0,NY):
        for k in range(0,NZ):
          c = 0
          force = 0
          while c < numObjects:
            force += forceEval(atom[0,c],atom[1,c],atom[2,c],x[i],y[j],z[k],atom[3,c],atom[4,c],atom[5,c],q2)
          if force < 0:
            grid[i,j,k] = 0

    return _grid