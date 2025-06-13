"""domain_generation.py

Functions for generating random, smoothed, and structured porous media images,
as well as initializing PorousMedia and Multiphase objects for PMMoTo.
"""

from __future__ import annotations
from typing import TypeVar, Any
import numpy as np
from numpy.typing import NDArray
from scipy.ndimage import gaussian_filter
from . import _domain_generation
from . import porousmedia
from . import multiphase
from ..io import data_read
from ..particles import particles
from ..core import communication
from ..core import utils
from ..core.subdomain import Subdomain
from ..core.subdomain_padded import PaddedSubdomain
from ..core.subdomain_verlet import VerletSubdomain

T = TypeVar("T", bound=np.generic)

__all__ = [
    "gen_img_random_binary",
    "gen_img_smoothed_random_binary",
    "gen_img_linear",
    "gen_pm_spheres_domain",
    "gen_pm_atom_domain",
    "gen_pm_atom_file",
    "gen_pm_inkbottle",
    "gen_mp_constant",
]


def gen_img_random_binary(
    shape: tuple[int, ...], p_zero: float = 0.5, seed: None | int = None
) -> NDArray[np.uint8]:
    """Generate a random binary grid with specified probability for zeros.

    Args:
        shape (tuple): Shape of the binary grid (e.g., (depth, rows, columns)).
        p_zero (float): Probability of a 0 occurring. Probability of a 1 is 1 - p_zero.
        seed (int, optional): Seed for the random number generator.

    Returns:
        np.ndarray: Random binary grid.

    """
    if not (0 <= p_zero <= 1):
        raise ValueError("Probability p_zero must be between 0 and 1.")

    # Create a random number generator with the given seed
    rng = np.random.default_rng(seed)

    return rng.choice([0, 1], size=shape, p=[p_zero, 1 - p_zero]).astype(np.uint8)


def gen_img_smoothed_random_binary(
    shape: tuple[int, ...],
    p_zero: float = 0.5,
    smoothness: float = 1.0,
    seed: None | int = None,
) -> NDArray[np.uint8]:
    """Generate a smoothed random binary grid.

    Args:
        shape (tuple): Shape of the binary grid.
        p_zero (float): Probability of a 0 occurring.
        smoothness (float): Controls the smoothness of the output grid.
        seed (int, optional): Seed for the random number generator.

    Returns:
        np.ndarray: Smoothed random binary grid.

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


def gen_img_linear(shape: tuple[int, ...], dim: int) -> NDArray[np.float64]:
    """Generate an image that varies linearly from 0 to N-1 along a given dimension.

    Args:
        shape (tuple): Shape of the output image.
        dim (int): Dimension along which to vary.

    Returns:
        np.ndarray: Linear image.

    """
    n = shape[dim]
    linear_values = np.linspace(0, n - 1, n, endpoint=True)

    # Reshape for broadcasting
    shape_expanded = [1, 1, 1]  # Start with a single value for each axis
    shape_expanded[dim] = shape[dim]  # Expand only the chosen axis
    linear_values = linear_values.reshape(shape_expanded)

    return linear_values * np.ones(shape)


def gen_pm_spheres_domain(
    subdomain: Subdomain | PaddedSubdomain | VerletSubdomain,
    spheres: NDArray[np.floating[Any]],
    kd: bool = False,
) -> porousmedia.PorousMedia:
    """Generate binary porous media domain from sphere data.

    Args:
        subdomain: Subdomain object.
        spheres: Sphere data array.
        kd (bool, optional): Use KD-tree for efficiency.

    Returns:
        PorousMedia: Initialized porous media object.

    """
    _spheres = particles.initialize_spheres(subdomain, spheres)

    img = _domain_generation.gen_pm_sphere(subdomain, _spheres, kd)
    pm = porousmedia.gen_pm(subdomain, img)
    pm.img = communication.update_buffer(subdomain, pm.img)
    pm.img = subdomain.set_wall_bcs(pm.img)
    utils.check_img_for_solid(subdomain, pm.img)

    return pm


def gen_pm_atom_domain(
    subdomain: Subdomain | PaddedSubdomain | VerletSubdomain,
    atom_locations: NDArray[np.floating[Any]],
    atom_radii: NDArray[np.floating[Any]],
    atom_types: NDArray[np.integer[Any]],
    kd: bool = False,
) -> porousmedia.PorousMedia:
    """Generate binary porous media domain from atom data.

    Args:
        subdomain: Subdomain object.
        atom_locations: Atom coordinates.
        atom_radii: Atom radii.
        atom_types: Atom types.
        kd (bool, optional): Use KD-tree for efficiency.

    Returns:
        PorousMedia: Initialized porous media object.

    """
    _atoms = particles.initialize_atoms(
        subdomain, atom_locations, atom_radii, atom_types
    )

    img = _domain_generation.gen_pm_atom(subdomain, _atoms, kd=False)
    pm = porousmedia.gen_pm(subdomain, img)
    pm.img = communication.update_buffer(subdomain, pm.img)

    utils.check_img_for_solid(subdomain, pm.img)

    return pm


def gen_pm_atom_file(
    subdomain: Subdomain | PaddedSubdomain | VerletSubdomain,
    lammps_file: str,
    atom_radii: NDArray[np.floating[Any]],
    type_map: None | dict[tuple[int, float], int] = None,
    add_periodic: bool = False,
    kd: bool = False,
) -> porousmedia.PorousMedia:
    """Generate binary porous media domain from a LAMMPS atom file.

    Args:
        subdomain: Subdomain object.
        lammps_file (str): Path to LAMMPS atom file.
        atom_radii: Atom radii.
        type_map (dict, optional): Mapping of atom types.
        add_periodic (bool, optional): Add periodic atoms.
        kd (bool, optional): Use KD-tree for efficiency.

    Returns:
        PorousMedia: Initialized porous media object.

    """
    positions, types, _, _ = data_read.read_lammps_atoms(lammps_file, type_map)

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


def gen_pm_inkbottle(
    subdomain: Subdomain | PaddedSubdomain | VerletSubdomain,
    r_y: float = 1.0,
    r_z: float = 1.0,
) -> porousmedia.PorousMedia:
    """Generate an inkbottle-shaped porous media with reservoirs.

    Args:
        subdomain: Subdomain object.
        r_y (float, optional): Elliptical scaling in y.
        r_z (float, optional): Elliptical scaling in z.

    Returns:
        PorousMedia: Initialized porous media object.

    """
    _img = _domain_generation.gen_inkbottle(
        subdomain.coords[0], subdomain.coords[1], subdomain.coords[2], r_y, r_z
    )
    pm = porousmedia.gen_pm(subdomain, _img)
    utils.check_img_for_solid(subdomain, pm.img)
    if subdomain.domain.num_subdomains > 1:
        pm.img = communication.update_buffer(subdomain, pm.img)

    if isinstance(subdomain, PaddedSubdomain | VerletSubdomain):
        pm.img = subdomain.update_reservoir(pm.img, np.uint8(1))

    return pm


def gen_mp_constant(
    porous_media: porousmedia.PorousMedia, fluid_phase: int = 1
) -> multiphase.Multiphase[np.uint8]:
    """Set the pore space to a specific fluid phase.

    Args:
        porous_media: PorousMedia object.
        fluid_phase (int, optional): Fluid phase to assign.

    Returns:
        Multiphase: Initialized multiphase object.

    """
    img = np.where(porous_media.img == 1, fluid_phase, 0).astype(np.uint8)
    mp = multiphase.Multiphase(porous_media, img, 2)

    return mp
