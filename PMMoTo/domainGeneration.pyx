#cython: cdivision=True
#cython: boundscheck=False
#cython: nonecheck=False
#cython: wraparound=False

import math
import numpy as np
cimport numpy as cnp
from libc.stdio cimport printf
cnp.import_array()


cdef double [:,:] verletList(double rCut, double x, double y, double z, double[:,:] atom):
  cdef int numObjects = atom.shape[1]
  cdef int c,vListSize,vListC
  cdef double re

  # find out how long verletlist is
  c=0
  vListSize=0
  while c < numObjects:
    re = (atom[0,c] - x)*(atom[0,c] - x) + (atom[1,c] - y)*(atom[1,c] - y) + (atom[2,c] - z)*(atom[2,c] - z)
    if re <= rCut*rCut:
      vListSize += 1 
    c+=1
  
  _verletAtom = np.ones((4, vListSize), dtype=np.double)
  cdef double [:,:] verletAtom

  verletAtom = _verletAtom

  #add atoms to verletlist
  c=0
  vListC=0
  while c < numObjects:
    re = (atom[0,c] - x)*(atom[0,c] - x) + (atom[1,c] - y)*(atom[1,c] - y) + (atom[2,c] - z)*(atom[2,c] - z)
    if re <= rCut*rCut:
      verletAtom[0,vListC] = atom[0,c]
      verletAtom[1,vListC] = atom[1,c]
      verletAtom[2,vListC] = atom[2,c]
      verletAtom[3,vListC] = atom[3,c]
      vListC+=1
    c+=1
  return _verletAtom


cdef int inAtom(double cx,double cy,double cz,double x,double y,double z,double r):
    cdef double re
    re = (cx - x)*(cx - x) + (cy - y)*(cy - y) + (cz - z)*(cz - z)
    if (re <= r): # already calculated 0.25*r*r
        return 0
    else:
        return 1

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

def domainGenVerlet(double rCut, double rMax, double[:] x, double[:] y, double[:] z, double[:,:] atom):
    ## only use with atom array that has format x,y,z,radius
    ## which for an atom, when evaluating maximum accessibility 
    ## (i.e. 0 sigma test particle) radius = 2^(1/6)*sigma/2
    cdef int NX = x.shape[0]
    cdef int NY = y.shape[0]
    cdef int NZ = z.shape[0]

    cdef int i, j, k, c, numObjects, pp,ppx
    
    _grid = np.ones((NX, NY, NZ), dtype=np.uint8)
    cdef cnp.uint8_t [:,:,:] grid

    grid = _grid

    cdef double [:,:] verletAtom
    cdef double xV = x[0]-rCut * 10
    cdef double yV = y[0]-rCut * 10
    cdef double zV = z[0]-rCut * 10
    pp = 0
    for i in range(0,NX):
      for j in range(0,NY):
        for k in range(0,NZ):
          rebuildDist2 = (x[i] - xV)*(x[i] - xV) + (y[j] - yV)*(y[j] - yV)  + (z[k] - zV)*(z[k] - zV)           
          if rebuildDist2 >= (rCut-rMax)*(rCut-rMax):
            verletAtom = verletList(rCut, x[i], y[j], z[k], atom)
            xV = x[i]
            yV = y[j]
            zV = z[k]
            pp=0
          pp+=1
          numObjects = verletAtom.shape[1]
          c = 0
          while (grid[i,j,k] == 1 and c < numObjects):
              grid[i,j,k] = inAtom(verletAtom[0,c],verletAtom[1,c],verletAtom[2,c],x[i],y[j],z[k],verletAtom[3,c])
              c += 1

    
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
