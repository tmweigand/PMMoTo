#cython: cdivision=True
#cython: boundscheck=False
#cython: nonecheck=False
#cython: wraparound=False

import math
import numpy as np
cimport numpy as cnp
cnp.import_array()

__all__ = [
  "domainGen",
  "domainGenVerlet",
  "domainGenINK"
]

cdef double [:,:] gen_verlet_list(double rCut, double x, double y, double z, double[:,:] atom):
  """

  """
  cdef int numObjects = atom.shape[1]
  cdef int c,vListSize,vListC
  cdef double re

  # find out how long verletlist is
  c = 0
  vListSize = 0
  while c < numObjects:
    re = (atom[0,c] - x)*(atom[0,c] - x) + (atom[1,c] - y)*(atom[1,c] - y) + (atom[2,c] - z)*(atom[2,c] - z)
    if re <= rCut*rCut:
      vListSize += 1 
    c += 1
  
  _verletAtom = np.ones((4, vListSize), dtype=np.double)
  cdef double [:,:] verletAtom
  verletAtom = _verletAtom

  #add atoms to verletlist
  c = 0
  vListC = 0
  while c < numObjects:
    re = (atom[0,c] - x)*(atom[0,c] - x) + (atom[1,c] - y)*(atom[1,c] - y) + (atom[2,c] - z)*(atom[2,c] - z)
    if re <= rCut*rCut:
      verletAtom[:,vListC] = atom[:,c]
      # verletAtom[1,vListC] = atom[1,c]
      # verletAtom[2,vListC] = atom[2,c]
      # verletAtom[3,vListC] = atom[3,c]
      vListC += 1
    c += 1
  return _verletAtom

cdef int inAtom(double cx,double cy,double cz,double x,double y,double z,double r):
    """
    """
    cdef double re
    re = (cx - x)*(cx - x) + (cy - y)*(cy - y) + (cz - z)*(cz - z)
    if (re <= r): # already calculated 0.25*r*r
        return 0
    else:
        return 1

def domainGen(double[:] x, double[:] y, double[:] z, double[:,:] atom):
    """
    """
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

def domainGenVerlet(list verletDomains, double[:] x, double[:] y, double[:] z, double[:,:] atom):
    """ 
    Only use with atom array that has format x,y,z,radius
    which for an atom, when evaluating maximum accessibility 
    (i.e. 0 sigma test particle) radius = 2^(1/6)*sigma/2
    """
    cdef int NX = x.shape[0]
    cdef int NY = y.shape[0]
    cdef int NZ = z.shape[0]

    cdef int i, j, k, n, c, numObjects, numDomains
    
    _grid = np.ones((NX, NY, NZ), dtype=np.uint8)
    cdef cnp.uint8_t [:,:,:] grid
    grid = _grid

    ### Generate subsubDomains 
    subNodes     = np.zeros([3],dtype=np.uint64)
    subNodesRem  = np.zeros([3],dtype=np.uint64)
    numDomains = np.prod(verletDomains)
    subNodes[0],subNodesRem[0] = divmod(NX,verletDomains[0])
    subNodes[1],subNodesRem[1] = divmod(NY,verletDomains[1])
    subNodes[2],subNodesRem[2] = divmod(NZ,verletDomains[2])

    ### Get the loop Info for the Verlet Domains
    _loopInfo = np.zeros([numDomains,6],dtype=np.int64)
    cdef cnp.int64_t [:,:] loopInfo
    loopInfo = _loopInfo

    n = 0
    for cI,i in enumerate(range(0,verletDomains[0])):
        for cJ,j in enumerate(range(0,verletDomains[1])):
            for cK,k in enumerate(range(0,verletDomains[2])):

              if i == 0:
                _loopInfo[n,0] = 0
                _loopInfo[n,1] = subNodes[0]
              elif i == verletDomains[0] - 1:
                _loopInfo[n,0] = cI*subNodes[0]
                _loopInfo[n,1] = (cI+1)*subNodes[0] + subNodesRem[0]
              else:
                _loopInfo[n,0] = cI*subNodes[0]
                _loopInfo[n,1] = (cI+1)*subNodes[0]

              if j == 0:
                _loopInfo[n,2] = 0
                _loopInfo[n,3] = subNodes[1]
              elif j == verletDomains[1] - 1:
                _loopInfo[n,2] = cJ*subNodes[1]
                _loopInfo[n,3] = (cJ+1)*subNodes[1] + subNodesRem[1]
              else:
                _loopInfo[n,2] = cJ*subNodes[1]
                _loopInfo[n,3] = (cJ+1)*subNodes[1]

              if k == 0:
                _loopInfo[n,4] = 0
                _loopInfo[n,5] = subNodes[2]
              elif k == verletDomains[2] - 1:
                _loopInfo[n,4] = cK*subNodes[2]
                _loopInfo[n,5] = (cK+1)*subNodes[2] + subNodesRem[2]
              else:
                _loopInfo[n,4] = cK*subNodes[2]
                _loopInfo[n,5] = (cK+1)*subNodes[2]
              
              n += 1

    ### Get Centroid of Verlet Domains
    length = np.zeros([numDomains,3],dtype=np.double)
    centroid = np.zeros([numDomains,3],dtype=np.double)
    maxDiameter = np.zeros([numDomains],dtype=np.double)
    maxAtomRadius = np.sqrt(np.max(atom[3,:]))
    for n in range(numDomains):
      length[n,0] = x[_loopInfo[n,1]-1] - x[_loopInfo[n,0]]
      length[n,1] = y[_loopInfo[n,3]-1] - y[_loopInfo[n,2]]
      length[n,2] = z[_loopInfo[n,5]-1] - z[_loopInfo[n,4]]
      centroid[n,0] = x[_loopInfo[n,0]] + length[n,0]/2. 
      centroid[n,1] = y[_loopInfo[n,2]] + length[n,1]/2.
      centroid[n,2] = z[_loopInfo[n,4]] + length[n,2]/2. 
      maxDiameter[n] = np.max(length[n])+2*maxAtomRadius

    cdef double [:,:] verletAtom
    for n in range(numDomains):
      verletAtom = gen_verlet_list(maxDiameter[n], centroid[n,0], centroid[n,1], centroid[n,2], atom)
      numObjects = verletAtom.shape[1]
      for i in range(loopInfo[n,0],loopInfo[n,1]):
        for j in range(loopInfo[n,2],loopInfo[n,3]):
          for k in range(loopInfo[n,4],loopInfo[n,5]):
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
