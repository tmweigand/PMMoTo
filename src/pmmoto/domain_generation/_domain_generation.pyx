#distutils: language = c++
#cython: cdivision=True
#cython: boundscheck=True
#cython: nonecheck=False
#cython: wraparound=False

import numpy as np
cimport numpy as cnp
from libcpp.vector cimport vector
from libc.math cimport sin,cos
from libcpp.unordered_map cimport unordered_map
cnp.import_array()

__all__ = [
    "gen_pm_sphere",
    "gen_pm_atom",
    "gen_pm_verlet_sphere",
    "gen_pm_verlet_atom",
    "gen_inkbottle",
    "convert_atoms_to_spheres"
]

cdef vector[verlet_sphere] gen_verlet_list(
                                        double verlet_radius, 
                                        double x, 
                                        double y, 
                                        double z, 
                                        double[:,:] spheres
                                        ):
    """
    Determine if sphere is a verlet sphere meaning inside subdomain
    """

    cdef: 
        int c
        double re
        verlet_sphere c_sphere
        vector[verlet_sphere] verlet_spheres
        int num_objects = spheres.shape[0]

    c = 0
    while c < num_objects:
        re = (  (spheres[c,0] - x)*(spheres[c,0] - x) 
              + (spheres[c,1] - y)*(spheres[c,1] - y) 
              + (spheres[c,2] - z)*(spheres[c,2] - z) )

        r = spheres[c,3] + verlet_radius

        if re <= r*r:
            c_sphere.x = spheres[c,0]
            c_sphere.y = spheres[c,1]
            c_sphere.z = spheres[c,2]
            c_sphere.r = spheres[c,3]
            verlet_spheres.push_back(c_sphere)
        c += 1
    
    return verlet_spheres


cdef cnp.uint8_t in_sphere(
                        double x, 
                        double y,
                        double z, 
                        double s_x, 
                        double s_y, 
                        double s_z, 
                        double s_r
                        ) noexcept:
    """
    Check if point in sphere. Assumes radius is squared!!!
    """

    cdef double re
    re = (s_x - x)*(s_x - x) + (s_y - y)*(s_y - y) + (s_z - z)*(s_z - z)
    if (re <= s_r): # s_r is assumed squared!
        return 0
    else:
        return 1

def gen_pm_sphere(
        double[:] x, 
        double[:] y, 
        double[:] z, 
        double[:,:] spheres
        ):
    """
    Determine if voxel centroid is located in a sphere
    """
    cdef: 
        int i, j, k, c
        int sx = x.shape[0]
        int sy = y.shape[0]
        int sz = z.shape[0]
        int num_spheres = spheres.shape[0]
        cnp.uint8_t [:,:,:] _grid

    grid = np.ones((sx, sy, sz), dtype=np.uint8)
    _grid = grid

    # Determine if voxel in any sphere
    for i in range(0,sx):
        for j in range(0,sy):
            for k in range(0,sz):
                c = 0
                while (_grid[i,j,k] == 1 and c < num_spheres):
                    _grid[i,j,k] = in_sphere(
                                        x[i],
                                        y[j],
                                        z[k],
                                        spheres[c,0],
                                        spheres[c,1],
                                        spheres[c,2],
                                        spheres[c,3]*spheres[c,3]
                                        )
                    c = c + 1

    return grid

def gen_pm_atom(
        double[:] x, 
        double[:] y, 
        double[:] z, 
        double[:,:] atom_locations,
        long[:] atom_types,
        unordered_map[int,double]  atom_cutoff
        ):
    """
    Determine if voxel centroid is located in atom
    """
    spheres = convert_atoms_to_spheres(
        atom_locations,
        atom_types,
        atom_cutoff
        ) 

    grid = gen_pm_sphere(
        x,
        y,
        z,
        spheres
    )
   
    return grid



def get_verlet_loop_info(
        list verlet_domains, 
        nodes,
        rem_nodes
    ):
    """
    Collect the loop information for each verlet domain 
    """
    num_domains = np.prod(verlet_domains)
    loop_info = np.zeros([num_domains,6],dtype=np.int64)
    loop_nodes = np.zeros([num_domains,3])
    n = 0
    for i in range(0,verlet_domains[0]):
        for j in range(0,verlet_domains[1]):
            for k in range(0,verlet_domains[2]):
                for nn,d in enumerate([i,j,k]):
                    if d == 0:
                        loop_info[n,nn*2] = 0
                        loop_info[n,nn*2+1] = nodes[nn]
                        loop_nodes[n,nn] = nodes[nn]
                    elif d == verlet_domains[nn] - 1:
                        loop_info[n,nn*2] = d*nodes[nn]
                        loop_info[n,nn*2+1] = (d+1)*nodes[nn] + rem_nodes[nn]
                        loop_nodes[n,nn] = nodes[nn] + rem_nodes[nn]
                    else:
                        loop_info[n,nn*2] = d*nodes[nn]
                        loop_info[n,nn*2+1] = (d+1)*nodes[nn]
                        loop_nodes[n,nn] = nodes[nn]
                n += 1
    
    return loop_info,loop_nodes


def get_verlet_domain_info(
        list verlet_domains, 
        int sx,
        int sy,
        int sz,
        double[:] x, 
        double[:] y, 
        double[:] z 
    ):
    """
    Divide domain into smaller cubes based on verlet_domains
    """

    num_domains = np.prod(verlet_domains)
    nodes  = np.zeros([3],dtype=np.uint64)
    rem_nodes = np.zeros([3],dtype=np.uint64)

    res = np.zeros([3],dtype=np.double)
    for nn,coord in enumerate([x,y,z]):
        res[nn] = coord[1] - coord[0]

    for n,d in enumerate([sx,sy,sz]):
      nodes[n],rem_nodes[n] = divmod(d,verlet_domains[n])

    loop_info,loop_nodes = get_verlet_loop_info(verlet_domains,nodes,rem_nodes)

    # Get Centroid of Verlet Domains
    centroid = np.zeros([num_domains,3],dtype=np.double)
    max_diameter = np.zeros([num_domains],dtype=np.double)
    length = np.zeros(3,dtype=np.double)
    for n in range(num_domains):
        diam = 0
        for nn,coord in enumerate([x,y,z]):
            length[nn] = res[nn]*loop_nodes[n,nn]
            centroid[n,nn] = coord[loop_info[n,nn*2]] + length[nn]/2.
            diam += length[nn]*length[nn]
        max_diameter[n] = np.sqrt(0.25*diam)

    return centroid,max_diameter,loop_info

def gen_pm_verlet_sphere(
        list verlet_domains, 
        double[:] x, 
        double[:] y, 
        double[:] z, 
        double[:,:] spheres
    ):
    """ 
    """
    cdef:
        int i, j, k, n, c, num_domains
        int num_spheres = spheres.shape[0]
        int sx = x.shape[0]
        int sy = y.shape[0]
        int sz = z.shape[0]
        vector[verlet_sphere] verlet_spheres
        cnp.uint8_t [:,:,:] _grid
        cnp.int64_t [:,:] _loop_info

    grid = np.ones((sx, sy, sz), dtype=np.uint8)
    _grid = grid
    
    centroid,max_diameter,_loop_info = get_verlet_domain_info(
        verlet_domains,
        sx,
        sy,
        sz,
        x,
        y,
        z
    )

    num_domains = np.prod(verlet_domains)
 
    for n in range(num_domains):
        verlet_spheres = gen_verlet_list(
                                    max_diameter[n], 
                                    centroid[n,0], 
                                    centroid[n,1], 
                                    centroid[n,2], 
                                    spheres
                                    )

        num_spheres = len(verlet_spheres)
        for i in range(_loop_info[n,0],_loop_info[n,1]):
            for j in range(_loop_info[n,2],_loop_info[n,3]):
                for k in range(_loop_info[n,4],_loop_info[n,5]):
                    c = 0
                    while (_grid[i,j,k] == 1 and c < num_spheres):
                        _grid[i,j,k] = in_sphere(
                                        x[i],
                                        y[j],
                                        z[k],
                                        verlet_spheres[c].x,
                                        verlet_spheres[c].y,
                                        verlet_spheres[c].z,
                                        verlet_spheres[c].r*verlet_spheres[c].r
                                        )
                        c += 1
    
    return grid

def gen_pm_verlet_atom(
        list verlet_domains, 
        double[:] x, 
        double[:] y, 
        double[:] z, 
        double[:,:] atom_locations,
        long[:] atom_types,
        unordered_map[int,double]  atom_cutoff
    ):
    """ 
    Calculate the radii for atoms and use spheres routines
    """

    spheres = convert_atoms_to_spheres(
        atom_locations,
        atom_types,
        atom_cutoff)

    grid = gen_pm_verlet_sphere(
        verlet_domains, 
        x, 
        y, 
        z, 
        spheres
    )

    return grid


def convert_atoms_to_spheres(
    double[:,:] atom_locations,
    long[:] atom_types,
    unordered_map[int,double]  atom_cutoff 
    ):
    """
    Convert atom locations, index, and cutoff to spheres of given radius
    """
    cdef:
        int n
        int num_atoms = atom_locations.shape[0]
        double [:,:] _spheres

    spheres = np.zeros((num_atoms, 4), dtype=np.double)
    _spheres = spheres
    _spheres[:,0:3] = atom_locations

    for n in range(0,num_atoms):
        _spheres[n,3] = atom_cutoff[atom_types[n]]

    return spheres


def gen_inkbottle(double[:] x, double[:] y, double[:] z):
    """
    Generate pm for inkbottle test case. See Miller_Bruning_etal_2019
    """
    cdef: 
        int i, j, k
        int sx = x.shape[0]
        int sy = y.shape[0]
        int sz = z.shape[0]
        double r
        cnp.uint8_t [:,:,:] _grid

    grid = np.zeros((sx, sy, sz), dtype=np.uint8)
    _grid = grid

    for i in range(0,sx):
        for j in range(0,sy):
            for k in range(0,sz):
                if x[i] < 0: # TMW Hack for reservoirs
                    _grid[i,j,k] = 1
                else:
                    r = (0.01*cos(0.01*x[i]) + 0.5*sin(x[i]) + 0.75)
                    if (y[j]*y[j] + z[k]*z[k]) <= r*r:
                        _grid[i,j,k] = 1

    return grid


def domainGenEllINK(double[:] x, double[:] y, double[:] z):

    cdef int NX = x.shape[0]
    cdef int NY = y.shape[0]
    cdef int NZ = z.shape[0]
    cdef int i, j, k
    cdef double r
    cdef double radiusY = 1.0
    cdef double radiusZ = 2.0

    _grid = np.zeros((NX, NY, NZ), dtype=np.uint8)
    cdef cnp.uint8_t [:,:,:] grid

    grid = _grid

    for i in range(0,NX):
      for j in range(0,NY):
        for k in range(0,NZ):
          r = (0.01*math.cos(0.01*x[i]) + 0.5*math.sin(x[i]) + 0.75)
          rY = r*radiusY
          rz = r*radiusZ
          if y[j]*y[j]/(rY*rY) + z[k]*z[k]/(rz*rz) <= 1:
            grid[i,j,k] = 1

    return _grid