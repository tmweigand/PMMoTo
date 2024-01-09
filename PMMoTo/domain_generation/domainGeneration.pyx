#distutils: language = c++
#cython: cdivision=True
#cython: boundscheck=False
#cython: nonecheck=False
#cython: wraparound=False

import numpy as np
cimport numpy as cnp
from libcpp.vector cimport vector
from libc.math cimport sin,cos
cnp.import_array()

__all__ = [
  "gen_domain_sphere_pack",
  "gen_domain_verlet_sphere_pack",
  "domainGenINK"
]

cdef vector[verlet_sphere] gen_verlet_list(double cutoff, double x, double y, double z, double[:,:] spheres):
  """Determine if sphere is a verlet sphere meaning inside subdomain

  """
  cdef int c
  cdef double re
  cdef verlet_sphere c_sphere
  cdef vector[verlet_sphere] verlet_spheres
  cdef int numObjects = spheres.shape[0]

  # find out how long verletlist is
  c = 0
  while c < numObjects:
    re = (spheres[c,0] - x)*(spheres[c,0] - x) + (spheres[c,1] - y)*(spheres[c,1] - y) + (spheres[c,2] - z)*(spheres[c,2] - z)
    if re <= cutoff*cutoff:
      c_sphere.x = spheres[c,0]
      c_sphere.y = spheres[c,1]
      c_sphere.z = spheres[c,2]
      c_sphere.r = spheres[c,3]
      verlet_spheres.push_back(c_sphere)
    c += 1
  
  return verlet_spheres


cdef int in_sphere(double x, double y, double z, double s_x, double s_y, double s_z, double s_r):
    """Check if point in sphere. Assumes radius is squared!
    """

    cdef double re
    re = (s_x - x)*(s_x - x) + (s_y - y)*(s_y - y) + (s_z - z)*(s_z - z)
    if (re <= s_r): # s_r is assumed squared!
        return 0
    else:
        return 1

def gen_domain_sphere_pack(double[:] x, double[:] y, double[:] z, double[:,:] spheres):
    """Determine if voxel centroid is located in a sphere

    """
    cdef int NX = x.shape[0]
    cdef int NY = y.shape[0]
    cdef int NZ = z.shape[0]
    cdef int num_spheres = spheres.shape[0]

    cdef int i, j, k, c

    grid = np.ones((NX, NY, NZ), dtype=np.uint8)
    cdef cnp.uint8_t [:,:,:] _grid
    _grid = grid

    # Square radius to avoid sqrt in in_sphere
    for i in range(0,num_spheres): 
      spheres[i,3] = spheres[i,3]*spheres[i,3]

    # Determine if point in sphere
    for i in range(0,NX):
      for j in range(0,NY):
        for k in range(0,NZ):
          c = 0
          while (_grid[i,j,k] == 1 and c < num_spheres):
            _grid[i,j,k] = in_sphere(x[i],y[j],z[k],spheres[c,0],spheres[c,1],spheres[c,2],spheres[c,3])
            c = c + 1

    return grid

def gen_domain_verlet_sphere_pack(list verlet_domains, double[:] x, double[:] y, double[:] z, double[:,:] spheres):
    """ 
    """
    cdef int NX = x.shape[0]
    cdef int NY = y.shape[0]
    cdef int NZ = z.shape[0]

    cdef int i, j, k, n, c, num_spheres, num_domains
    cdef vector[verlet_sphere] verlet_spheres

    grid = np.ones((NX, NY, NZ), dtype=np.uint8)
    cdef cnp.uint8_t [:,:,:] _grid
    _grid = grid
    
    # Square radius to avoid sqrt in in_sphere
    num_spheres = spheres.shape[0]
    for i in range(0,num_spheres): 
      spheres[i,3] = spheres[i,3]*spheres[i,3]

    # Divide domain into smaler cubes based on verlet_domains
    num_domains = np.prod(verlet_domains)
    nodes  = np.zeros([3],dtype=np.uint64)
    rem_nodes = np.zeros([3],dtype=np.uint64)

    for n,d in enumerate([NX,NY,NZ]):
      nodes[n],rem_nodes[n] = divmod(d,verlet_domains[n])

    # Get the loop Info (nodes) for the verlet domains
    loop_info = np.zeros([num_domains,6],dtype=np.int64)
    cdef cnp.int64_t [:,:] _loop_info
    _loop_info = loop_info

    n = 0
    for i in range(0,verlet_domains[0]):
        for j in range(0,verlet_domains[1]):
            for k in range(0,verlet_domains[2]):
              for nn,d in enumerate([i,j,k]):
                if d == 0:
                  _loop_info[n,nn*2] = 0
                  _loop_info[n,nn*2+1] = nodes[nn]
                elif d == verlet_domains[nn] - 1:
                  _loop_info[n,nn*2] = d*nodes[nn]
                  _loop_info[n,nn*2+1] = (d+1)*nodes[nn] + rem_nodes[nn]
                else:
                  _loop_info[n,nn*2] = d*nodes[nn]
                  _loop_info[n,nn*2+1] = (d+1)*nodes[nn]
              n += 1


    # Get Centroid of Verlet Domains
    max_sphere_radius = np.sqrt(np.max(spheres[:,3]))
    centroid = np.zeros([num_domains,3],dtype=np.double)
    max_diameter = np.zeros([num_domains],dtype=np.double)
    for n in range(num_domains):
      for nn,coord in enumerate([x,y,z]):
        length = coord[_loop_info[n,nn*2+1]-1] - coord[_loop_info[n,nn*2]]
        centroid[n,nn] = coord[_loop_info[n,nn*2]] + length/2
        max_diameter[n] = np.max(length) + 2.*max_sphere_radius

    # Loop through verlet domains, generating domain
    for n in range(num_domains):
      verlet_spheres = gen_verlet_list(max_diameter[n], centroid[n,0], centroid[n,1], centroid[n,2], spheres)
      num_spheres = len(verlet_spheres)
      for i in range(_loop_info[n,0],_loop_info[n,1]):
        for j in range(_loop_info[n,2],_loop_info[n,3]):
          for k in range(_loop_info[n,4],_loop_info[n,5]):
            c = 0
            while (_grid[i,j,k] == 1 and c < num_spheres):
                _grid[i,j,k] = in_sphere(x[i],y[j],z[k],verlet_spheres[c].x,verlet_spheres[c].y,verlet_spheres[c].z,verlet_spheres[c].r)
                c += 1
    
    return grid


def gen_domain_inkbottle(double[:] x, double[:] y, double[:] z):
  """
  """
    cdef int NX = x.shape[0]
    cdef int NY = y.shape[0]
    cdef int NZ = z.shape[0]
    cdef int i, j, k
    cdef double r

    grid = np.zeros((NX, NY, NZ), dtype=np.uint8)
    cdef cnp.uint8_t [:,:,:] _grid
    _grid = grid

    for i in range(0,NX):
      for j in range(0,NY):
        for k in range(0,NZ):
          r = (0.01*cos(0.01*x[i]) + 0.5*sin(x[i]) + 0.75)
          if y[j]*y[j] + z[k]*z[k] <= r*r:
            _grid[i,j,k] = 1

    return grid
