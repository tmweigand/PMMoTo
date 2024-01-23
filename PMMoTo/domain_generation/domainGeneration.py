import numpy as np
from mpi4py import MPI
from . import _domainGeneration
from pmmoto.core import communication
from pmmoto.core import utils
from pmmoto.core import porousMedia

__all__ = [
    "gen_pm_spheres_domain",
    "gen_pm_verlet_spheres",
    "gen_pm_inkbottle",
    "gen_mp_constant",
    "gen_mp_from_grid"   
]

def gen_pm_spheres_domain(subdomain,sphere_data,domain_data,res_size = 0):
    """
    """
    subdomain.update_domain_size(domain_data)
    _grid = _domainGeneration.gen_domain_sphere_pack(subdomain.coords[0],
                                                     subdomain.coords[1],
                                                     subdomain.coords[2],
                                                     sphere_data)
    pm = porousMedia.gen_pm(subdomain,_grid,res_size)
    pm.grid = communication.update_buffer(subdomain,pm.grid)

    utils.check_grid(subdomain,pm.grid)

    return pm

def gen_pm_verlet_spheres(subdomain,sphere_data,verlet=[2,2,2],res_size = 0):
    """
    """
    _grid = _domainGeneration.gen_domain_verlet_sphere_pack(verlet,
                                                           subdomain.coords[0],
                                                           subdomain.coords[1],
                                                           subdomain.coords[2],
                                                           sphere_data)
    pm = porousMedia.gen_pm(subdomain,_grid,res_size)
    utils.check_grid(subdomain,pm.grid)
    pm.grid = communication.update_buffer(subdomain,pm.grid)

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
