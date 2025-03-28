# Cython optimizations
# cython: profile=True
# cython: linetrace=True
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# cython: nonecheck=False

import numpy as np
cimport numpy as cnp
from numpy cimport uint8_t
from libcpp.unordered_map cimport unordered_map
from libc.math cimport sin, cos
cnp.import_array()

from ..particles._particles cimport PySphereList
from ..particles._particles import create_box
from ._domain_generation cimport Grid, Verlet
from ._domain_generation cimport gen_sphere_img_brute_force
from ._domain_generation cimport gen_sphere_img_kd_method


__all__ = [
    "gen_pm_sphere",
    "gen_pm_atom",
    "gen_inkbottle",
    "gen_elliptical_inkbottle",
]

def gen_pm_sphere(subdomain, spheres, kd: bool = False) -> np.ndarray:
    """
    Determine if voxel centroids are located inside spheres.

    Args:
        subdomain: Subdomain object containing voxel and Verlet information.
        spheres: SphereList or PySphereList object.
        kd (bool): Whether to use KD-tree for sphere lookup.

    Returns:
        np.ndarray: 3D binary image indicating voxel inclusion in spheres.
    """
    cdef:
        cnp.uint8_t[:, :, :] _img
        Grid grid_c
        Verlet verlet_c

    # Initialize binary image
    img = np.ones(subdomain.voxels, dtype=np.uint8)
    _img = img

    # Convert subdomain coordinates to vectors
    grid_c.x, grid_c.y, grid_c.z = subdomain.coords

    # Determine strides for indexing
    for stride in img.strides:
        grid_c.strides.push_back(stride // img.itemsize)

    # Convert Verlet information
    verlet_c.num_verlet = subdomain.num_verlet
    verlet_c.loops = subdomain.verlet_loop
    verlet_c.diameters = subdomain.max_diameters
    verlet_c.centroids = subdomain.centroids

    # Create bounding boxes for Verlet cells
    verlet_c.box = {n: create_box(subdomain.verlet_box[n]) for n in range(subdomain.num_verlet)}

    # Generate sphere image using KD-tree or brute force
    if kd:
        gen_sphere_img_kd_method(
            <uint8_t*>&_img[0, 0, 0],
            grid_c,
            verlet_c,
            (<PySphereList>spheres)._sphere_list,
        )
    else:
        gen_sphere_img_brute_force(
            <uint8_t*>&_img[0, 0, 0],
            grid_c,
            verlet_c,
            (<PySphereList>spheres)._sphere_list
        )

    return np.ascontiguousarray(img)

def gen_pm_atom(subdomain, atoms, kd: bool = False) -> np.ndarray:
    """
    Determine if voxel centroids are located inside atoms.

    Args:
        subdomain: Subdomain object containing voxel and Verlet information.
        atoms: 
    Returns:
        np.ndarray: 3D binary image indicating voxel inclusion in atoms.
    """
    return gen_pm_sphere(subdomain, atoms)


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


def gen_elliptical_inkbottle(double[:] x, double[:] y, double[:] z):
    """
    Generate ellipitical inkbottle test case. See Miller_Bruning_etal_2019
    """
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
          r = (0.01*cos(0.01*x[i]) + 0.5*sin(x[i]) + 0.75)
          rY = r*radiusY
          rz = r*radiusZ
          if y[j]*y[j]/(rY*rY) + z[k]*z[k]/(rz*rz) <= 1:
            _grid[i,j,k] = 1

    return grid