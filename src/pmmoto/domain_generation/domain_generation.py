import numpy as np
from scipy.ndimage import gaussian_filter
from . import _domain_generation
from . import porousmedia
from . import multiphase
from ..io import data_read
from ..particles import particles
from ..core import communication
from ..core import utils


__all__ = [
    "gen_random_binary_grid",
    "gen_smoothed_random_binary_grid",
    "gen_linear_img",
    "gen_pm_spheres_domain",
    "gen_pm_atom_domain",
    "gen_pm_atom_file",
    "gen_pm_inkbottle",
    "gen_mp_constant",
    "gen_mp_from_grid",
]


def gen_random_binary_grid(shape, p_zero=0.5, seed=None):
    """
    Generate a binary grid of the provided shape where
    each voxel is a random selection (0 or 1), with a specified probability.

    Args:
        shape (tuple): Shape of the binary grid (e.g., (depth, rows, columns)).
        p_zero (float): Probability of a 0 occurring. Probability of a 1 is 1 - p_zero.
        seed (int, optional): Seed for the random number generator. If None, the results are not reproducible.

    Returns:
        np.ndarray: A NumPy array with the specified shape, containing random 0s and 1s.
    """
    if not (0 <= p_zero <= 1):
        raise ValueError("Probability p_zero must be between 0 and 1.")

    # Create a random number generator with the given seed
    rng = np.random.default_rng(seed)

    return rng.choice([0, 1], size=shape, p=[p_zero, 1 - p_zero]).astype(np.uint8)


def gen_smoothed_random_binary_grid(shape, p_zero=0.5, smoothness=1.0, seed=None):
    """
    Generate a smoothed binary grid of the provided shape where
    each voxel is a random selection (0 or 1), with a specified probability.

    Args:
        shape (tuple): Shape of the binary grid (e.g., (depth, rows, columns)).
        p_zero (float): Probability of a 0 occurring. Probability of a 1 is 1 - p_zero.
        smoothness (float): Controls the smoothness of the output grid. Higher values
                            result in smoother transitions between 0 and 1.

    Returns:
        np.ndarray: A NumPy array with the specified shape, containing smoothed random 0s and 1s.
    """
    if not (0 <= p_zero <= 1):
        raise ValueError("Probability p_zero must be between 0 and 1.")
    if smoothness < 0:
        raise ValueError("Smoothness must be a non-negative value.")

    # Create a random number generator with the given seed
    rng = np.random.default_rng(seed)

    # Generate a random binary grid
    random_grid = rng.choice([0, 1], size=shape, p=[p_zero, 1 - p_zero]).astype(
        np.float32
    )

    # Apply Gaussian filter to smooth the grid
    smoothed_grid = gaussian_filter(random_grid, sigma=smoothness)

    # Threshold the smoothed grid to return binary values
    binary_grid = (smoothed_grid > 0.5).astype(np.uint8)

    return binary_grid


def gen_linear_img(shape, dim):
    """
    Generates an image that varies from 0-N-1 along dim
    """
    n = shape[dim]
    linear_values = np.linspace(0, n - 1, n, endpoint=True)

    # Reshape for broadcasting
    shape_expanded = [1, 1, 1]  # Start with a single value for each axis
    shape_expanded[dim] = shape[dim]  # Expand only the chosen axis
    linear_values = linear_values.reshape(shape_expanded)

    return linear_values * np.ones(shape)


def gen_pm_spheres_domain(subdomain, spheres, kd=False):
    """
    Generate binary porous media (pm) domain from sphere data that contains radii
    """
    _spheres = particles.initialize_spheres(subdomain, spheres)

    img = _domain_generation.gen_pm_sphere(subdomain, _spheres, kd)
    pm = porousmedia.gen_pm(subdomain, img)
    pm.img = communication.update_buffer(subdomain, pm.img)
    pm.img = subdomain.set_wall_bcs(pm.img)
    utils.check_img_for_solid(subdomain, pm.img)

    return pm


def gen_pm_atom_domain(subdomain, atom_locations, atom_radii, atom_types, kd=False):
    """
    Generate binary porous media (pm) domain from atom data, types and cutoff
    """
    _atoms = particles.initialize_atoms(
        subdomain, atom_locations, atom_radii, atom_types
    )

    img = _domain_generation.gen_pm_atom(subdomain, _atoms, kd=False)
    pm = porousmedia.gen_pm(subdomain, img)
    pm.img = communication.update_buffer(subdomain, pm.img)

    utils.check_img_for_solid(subdomain, pm.img)

    return pm


def gen_pm_atom_file(subdomain, lammps_file, atom_radii, add_periodic=False, kd=False):
    """
    Generate binary porous media (pm) domain from atom data, types and cutoff
    """

    positions, types, _, _ = data_read.read_lammps_atoms(lammps_file)

    _atoms = particles.initialize_atoms(
        subdomain=subdomain,
        atom_coordinates=positions,
        atom_radii=atom_radii,
        atom_ids=types,
        add_periodic=add_periodic,
        trim_intersecting=True,
    )

    img = _domain_generation.gen_pm_atom(subdomain, _atoms, kd=False)
    pm = porousmedia.gen_pm(subdomain, img)
    pm.img = communication.update_buffer(subdomain, pm.img)

    utils.check_img_for_solid(subdomain, pm.img)

    return pm


def gen_pm_inkbottle(subdomain):
    """
    Generate an inkbottle with reservoirs
    """

    _img = _domain_generation.gen_inkbottle(
        subdomain.coords[0], subdomain.coords[1], subdomain.coords[2]
    )
    pm = porousmedia.gen_pm(subdomain, _img)
    utils.check_img_for_solid(subdomain, pm.img)
    if subdomain.domain.num_subdomains > 1:
        pm.img = communication.update_buffer(subdomain, pm.img)

    pm.img = subdomain.update_reservoir(pm.img, 1)

    return pm


def gen_mp_constant(porous_media, fluid_phase=1):
    """
    Set the pore space to be a specific fluid phase
    """
    img = np.where(porous_media.img == 1, fluid_phase, 0).astype(np.uint8)
    mp = multiphase.Multiphase(porous_media, img, 2)

    return mp


def gen_mp_from_grid(mp, grid):
    """
    Set the multiphase pore space accoring to input grid
    """
    mp.mp_grid = grid

    return mp
