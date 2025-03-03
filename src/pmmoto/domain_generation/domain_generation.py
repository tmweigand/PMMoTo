import numpy as np
from scipy.ndimage import gaussian_filter
from . import _domain_generation
from ..core import communication
from ..core import utils
from ..core import porousmedia
from ..core import subdomain_features


__all__ = [
    "gen_random_binary_grid",
    "gen_smoothed_random_binary_grid",
    "gen_linear_img",
    "gen_pm_spheres_domain",
    "gen_pm_atom_domain",
    "gen_pm_verlet_spheres_domain",
    "gen_pm_verlet_atom_domain",
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


def gen_pm_spheres_domain(subdomain, spheres):
    """
    Generate binary domain (pm) from sphere data that contains radii
    """
    img = _domain_generation.gen_pm_sphere(subdomain, spheres)
    pm = porousmedia.gen_pm(subdomain, img)
    pm.img = communication.update_buffer(subdomain, pm.img)
    pm.img = subdomain.set_wall_bcs(pm.img)

    utils.check_grid(subdomain, pm.img)

    return pm


def gen_pm_atom_domain(subdomain, atom_locations, atom_types, atom_cutoff):
    """
    Generate binary domain (pm) from atom data, types and cutoff
    """
    _img = _domain_generation.gen_pm_atom(
        subdomain.coords[0],
        subdomain.coords[1],
        subdomain.coords[2],
        atom_locations,
        atom_types,
        atom_cutoff,
    )

    pm = porousmedia.gen_pm(subdomain, _img)
    pm.img = communication.update_buffer(subdomain, pm.img)

    utils.check_grid(subdomain, pm.img)

    return pm


def gen_pm_verlet_spheres_domain(subdomain, spheres):
    """
    Generate binary domain (pm) from sphere data that contains radii
       using verlet domains
    """
    _img = np.ones(subdomain.voxels, dtype=np.uint8)
    for n in range(subdomain.num_verlet):
        verlet_spheres = _domain_generation.gen_verlet_list(
            subdomain.max_diameters[n],
            subdomain.centroids[n, 0],
            subdomain.centroids[n, 1],
            subdomain.centroids[n, 2],
            spheres,
        )

        print(verlet_spheres)

        # _img = _domain_generation.gen_pm_verlet_sphere(
        #     _img,
        #     subdomain.verlet_loop[n],
        #     subdomain.coords[0],
        #     subdomain.coords[1],
        #     subdomain.coords[2],
        #     verlet_spheres,
        # )

    # pm = porousmedia.gen_pm(subdomain, _img)
    # pm.img = communication.update_buffer(subdomain, pm.img)

    # # utils.check_grid(subdomain, pm.img)

    # return pm


def gen_pm_verlet_atom_domain(
    subdomain, atom_locations, atom_types, atom_cutoff, verlet=[1, 1, 1], res_size=0
):
    """
    Generate binary domain (pm) from atom data, types and cutoff
       using verlet domains
    """
    _img = _domain_generation.gen_pm_verlet_atom(
        verlet,
        subdomain.coords[0],
        subdomain.coords[1],
        subdomain.coords[2],
        atom_locations,
        atom_types,
        atom_cutoff,
    )

    pm = porousmedia.gen_pm(subdomain, _img)
    pm.img = communication.update_buffer(subdomain, pm.img)

    utils.check_grid(subdomain, pm.img)

    return pm


def gen_pm_inkbottle(subdomain, domain_data, res_size=0):
    """ """
    subdomain.update_domain_size(domain_data)
    _img = _domain_generation.gen_pm_inkbottle(
        subdomain.coords[0], subdomain.coords[1], subdomain.coords[2]
    )
    pm = porousmedia.gen_pm(subdomain, _img)
    utils.check_grid(subdomain, pm.img)
    pm.img = communication.update_buffer(subdomain, pm.img)

    return pm


def gen_mp_constant(mp, fluid_phase=1):
    """
    Set the pore space to be a specific fluid phase
    """
    mp.grid = np.where(mp.pm_grid == 1, fluid_phase, 0).astype(np.uint8)

    return mp


def gen_mp_from_grid(mp, grid):
    """
    Set the multiphase pore space accoring to input grid
    """
    mp.mp_grid = grid

    return mp
