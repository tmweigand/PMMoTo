import numpy as np
from mpi4py import MPI
from pmmoto.domain_generation import _domainGeneration
from pmmoto.core import communication
from pmmoto.core import utils
from pmmoto.core import porousMedia
from pmmoto.core import Orientation


__all__ = [
    "gen_pm_spheres_domain",
    "gen_pm_verlet_spheres",
    "gen_pm_inkbottle",
    "gen_mp_constant",
    "gen_mp_from_grid"
]

def gen_pm_spheres_domain(subdomain,spheres,domain_data,res_size = 0,add_periodic = False):
    """
    """
    subdomain.update_domain_size(domain_data)
    if add_periodic:
        periodic_spheres = get_periodic_spheres(subdomain,spheres)
        spheres = np.concatenate( (periodic_spheres,spheres) )
        
    _grid = _domainGeneration.gen_domain_sphere_pack(subdomain.coords[0],
                                                     subdomain.coords[1],
                                                     subdomain.coords[2],
                                                     spheres)
    pm = porousMedia.gen_pm(subdomain,_grid,res_size)
    pm.grid = communication.update_buffer(subdomain,pm.grid)

    utils.check_grid(subdomain,pm.grid)

    return pm

def gen_pm_verlet_spheres(subdomain,spheres,domain_data,verlet=[1,1,1],res_size = 0,add_periodic = False):
    """
    """
    subdomain.update_domain_size(domain_data)
    if add_periodic:
        periodic_spheres = get_periodic_spheres(subdomain,spheres)
        spheres = np.concatenate( (periodic_spheres,spheres) )

    _grid = _domainGeneration.gen_domain_verlet_sphere_pack(verlet,
                                                           subdomain.coords[0],
                                                           subdomain.coords[1],
                                                           subdomain.coords[2],
                                                           spheres)
    pm = porousMedia.gen_pm(subdomain,_grid,res_size)
    pm.grid = communication.update_buffer(subdomain,pm.grid)

    utils.check_grid(subdomain,pm.grid)

    return pm

def gen_pm_inkbottle(subdomain,domain_data,res_size = 0):
    """
    """
    subdomain.update_domain_size(domain_data)
    _grid = _domainGeneration.gen_domain_inkbottle(subdomain.coords[0],
                                                   subdomain.coords[1],
                                                   subdomain.coords[2])
    pm = porousMedia.gen_pm(subdomain,_grid,res_size)
    utils.check_grid(subdomain,pm.grid)
    pm.grid = communication.update_buffer(subdomain,pm.grid)

    return pm


def gen_mp_constant(mp,fluid_phase = 1):
    """
    Set the pore space to be a specific fluid phase
    """
    mp.grid = np.where(mp.porousmedia.grid == 1,fluid_phase,0).astype(np.uint8)
    mp.update_grid()

    return mp


def gen_mp_from_grid(mp,grid):
    """
    Set the multiphase pore space accoring to input grid
    """
    mp.grid = grid
    mp.update_grid()

    return mp


def is_boundary_sphere(sphere_data,voxel,domain):
    """
    Determine if a sphere crosses the domain boundaries
    """
    crosses_boundary = np.zeros(6,dtype=np.uint8)
    for n in range(0,3):
        if sphere_data[n] - sphere_data[3] - voxel[n] <= domain[n,0]:
            crosses_boundary[n*2] = True
        if sphere_data[n] + sphere_data[3] + voxel[n] >= domain[n,1]:
            crosses_boundary[n*2+1] = True
 
    return crosses_boundary

def is_inside_domain(sphere_data,domain):
    """
    Determine if a sphere is within domain boundaries
    """
    count_dim = 0
    for n in range(0,3):
        if domain[n,0] <= sphere_data[n] <=  domain[n,1]:
            count_dim += 1

    return count_dim == 3


def reflect_boundary_sphere(sphere_data,crosses_boundary,domain_length,boundaries):
    """
    Add spheres that cross periodic boundaries
    """
    periodic_spheres = []
    for f_index in Orientation.faces:
        face = Orientation.faces[f_index]
        index = face['argOrder'][0]
        if crosses_boundary[f_index] and boundaries[index] == 2:
            shift_sphere = sphere_data[index] + face['dir']*domain_length[index]
            add_sphere = list(sphere_data)
            add_sphere[index] = shift_sphere
            periodic_spheres.append(add_sphere)

    for e_index in Orientation.edges:
        edge = Orientation.edges[e_index]
        add_sphere = list(sphere_data)
        periodic = [False,False]
        for n_face,f_index in enumerate(edge['faceIndex']):
            face = Orientation.faces[f_index]
            index = face['argOrder'][0]
            if crosses_boundary[f_index] and boundaries[index] == 2:
                shift_sphere = sphere_data[index] + face['dir']*domain_length[index]
                add_sphere[index] = shift_sphere
                periodic[n_face] = True
        if all(periodic):
            periodic_spheres.append(add_sphere)

    for c_index in Orientation.corners:
        corner = Orientation.corners[c_index]
        add_sphere = list(sphere_data)
        periodic = [False,False,False]
        for n_face,f_index in enumerate(corner['faceIndex']):
            face = Orientation.faces[f_index]
            index = face['argOrder'][0]
            if crosses_boundary[f_index] and boundaries[index] == 2:
                shift_sphere = sphere_data[index] + face['dir']*domain_length[index]
                add_sphere[index] = shift_sphere
                periodic[n_face] = True
        if all(periodic):
            periodic_spheres.append(add_sphere)

    return periodic_spheres

def get_periodic_spheres(subdomain,sphere_data):
    """
    Add spheres that extend pass boundary and are periodic
    """
    domain = subdomain.domain.size_domain
    res = subdomain.domain.voxel
    boundaries = np.array(subdomain.domain.boundaries).flatten()
    num_spheres = sphere_data.shape[0]

    domain_length = [domain[0,1]-domain[0,0],
                     domain[1,1]-domain[1,0],
                     domain[2,1]-domain[2,0]]

    all_periodic_spheres = []
    for n_sphere in range(num_spheres):
        inside_domain = is_inside_domain(sphere_data[n_sphere,:],domain)
        if inside_domain:
            crosses_boundary = is_boundary_sphere(sphere_data[n_sphere,:],res,domain)

            # Pass internal spheres
            if np.sum(crosses_boundary) == 0:
                continue
            periodic_spheres = reflect_boundary_sphere(sphere_data[n_sphere,:],
                                                    crosses_boundary,
                                                    domain_length,
                                                    boundaries)
                                                        
            all_periodic_spheres.extend(periodic_spheres)

    return np.array(all_periodic_spheres)
