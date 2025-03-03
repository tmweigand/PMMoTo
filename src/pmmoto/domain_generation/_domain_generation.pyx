# cython: profile=True
# cython: linetrace=True
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True
# cython: nonecheck=False

import numpy as np
cimport numpy as cnp
from numpy cimport uint8_t
from libcpp.vector cimport vector
from libcpp.unordered_map cimport unordered_map
from libcpp.memory cimport shared_ptr
from libc.math cimport sin, cos
cnp.import_array()

from .particles.sphere_pack cimport Sphere
from .particles.sphere_pack cimport SphereList
from .particles.sphere_pack cimport initialize_list
from .particles.sphere_pack cimport trim_sphere_list_spheres


__all__ = [
    "gen_pm_sphere",
    "gen_pm_atom",
    "gen_pm_inkbottle",
    "convert_atoms_to_spheres",
    # "trim_list"
]

def gen_pm_sphere(subdomain, spheres, kd = False, trim = False):
    """
    Determine if voxel centroid is located in a sphere
    """
    cdef: 
        cnp.uint8_t [:, :, :] _img
        Grid grid_c
        Verlet verlet_c
        vector[vector[double]] spheres_c
        vector[vector[double]] domain_box,subdomain_box

    # Initialize img
    img = np.ones(subdomain.voxels, dtype=np.uint8)
    _img = img

    # Convert coords to vectors
    grid_c.x = subdomain.coords[0]
    grid_c.y = subdomain.coords[1]
    grid_c.z = subdomain.coords[2]

    # Determine strides for indexing
    for stride in img.strides:
        _stride = stride//img.itemsize
        grid_c.strides.push_back(_stride)

    # Initialize SphereList Class
    spheres_c = spheres
    if trim:
        point = subdomain.get_centroid()
        sd_radius = subdomain.get_radius()
        radius = sd_radius + np.max(spheres[:,3])

    domain_box = subdomain.domain.box
    subdomain_box = subdomain.own_box

    cdef shared_ptr[SphereList] all_spheres = initialize_list[SphereList,Sphere](spheres_c,domain_box,subdomain_box)

    # Convert Verlet info
    verlet_c.num_verlet = subdomain.num_verlet
    verlet_c.centroids = subdomain.centroids
    verlet_c.diameters = subdomain.max_diameters
    verlet_c.loops = subdomain.verlet_loop

    if kd:
        gen_sphere_img_kd_method(
            <uint8_t*>&_img[0,0,0], grid_c, verlet_c, all_spheres
        )
    else:
        gen_sphere_img_brute_force(
            <uint8_t*>&_img[0,0,0], grid_c, verlet_c, all_spheres
        )

    return img

def gen_pm_atom(subdomain,atom_locations,atom_types,atom_cutoff
):
    """
    Determine if voxel centroid is located in atom
    """
    spheres = convert_atoms_to_spheres(
        atom_locations,
        atom_types,
        atom_cutoff
        ) 

    img = gen_pm_sphere(subdomain,spheres)
   
    return img


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


def gen_pm_inkbottle(double[:] x, double[:] y, double[:] z):
    """
    Generate pm for inkbottle test case. See Miller_Bruning_etal_2019
    """
    cdef: 
        int i, j, k
        int sx = x.shape[0]
        int sy = y.shape[0]
        int sz = z.shape[0]
        double r
        cnp.uint8_t [:,:,:] _img

    img = np.zeros((sx, sy, sz), dtype=np.uint8)
    _img = img

    for i in range(0,sx):
        for j in range(0,sy):
            for k in range(0,sz):
                r = (0.01*cos(0.01*x[i]) + 0.5*sin(x[i]) + 0.75)
                if y[j]*y[j] + z[k]*z[k] <= r*r:
                    _img[i,j,k] = 1

    return img


# def trim_list(subdomain, spheres, kd_tree = False):
#     """
#     Trim a list. This is useful in input is not on subdomain basis.
#     """
#     cdef:
#         vector[vector[double]] spheres_in
#         vector[Sphere] spheres_out
#         vector[double] point
#         double radius
#         unique_ptr[SphereList] sphere_list

#     spheres_in = spheres
#     point = subdomain.get_centroid()
#     sd_radius = subdomain.get_radius()

#     # Add maximum sphere radius to account for spheres where centroids are outside
#     radius = sd_radius + np.max(spheres[:,3])

#     spheres_out = trim_sphere_list_spheres(spheres_in,point,radius, kd_tree)

#     cdef cnp.ndarray[cnp.double_t, ndim=2] spheres_result = np.empty((spheres_out.size(), 4), dtype=np.double)
    
#     # Fill the array with data from the spheres_out vector
#     for i in range(spheres_out.size()):
#         # sphere = spheres_out[i]
#         spheres_result[i, 0] = spheres_out[i].coordinates.x
#         spheres_result[i, 1] = spheres_out[i].coordinates.y
#         spheres_result[i, 2] = spheres_out[i].coordinates.z
#         spheres_result[i, 3] = spheres_out[i].radius


#     return spheres_result