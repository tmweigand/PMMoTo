"""particles.py"""

import numpy as np
from typing import Dict, Union

from ._particles import _initialize_atoms, _initialize_spheres

__all__ = [
    "initialize_atoms",
    "initialize_spheres",
]


def initialize_atoms(
    subdomain,
    atom_coordinates: np.ndarray,
    atom_radii: Union[Dict[int, float], np.ndarray],
    atoms_ids: np.ndarray,
    by_type: bool = False,
    add_periodic: bool = False,
    set_own: bool = True,
    trim_intersecting: bool = False,
    trim_within: bool = False,
) -> "AtomMap":
    """
    Initialize a list of particles efficiently with memory management.

    Args:
        subdomain: Domain subdivision object
        atom_coordinates: Array of shape (n_atoms, 3) with xyz coordinates
        atom_radii: Dictionary mapping atom types to radii or array of radii
        atoms_ids: Array of atom type IDs
        by_type: Whether to organize atoms by type
        add_periodic: Add periodic images at boundaries
        set_own: Mark particles owned by this subdomain
        trim_intersecting: Remove particles intersecting boundary
        trim_within: Remove particles fully within boundary

    Returns:
        AtomMap: Container of initialized particles
    """
    # Convert inputs to contiguous arrays for better memory efficiency
    if not atom_coordinates.flags["C_CONTIGUOUS"]:
        atom_coordinates = np.ascontiguousarray(atom_coordinates)

    if isinstance(atom_radii, np.ndarray) and not atom_radii.flags["C_CONTIGUOUS"]:
        atom_radii = np.ascontiguousarray(atom_radii)

    if not atoms_ids.flags["C_CONTIGUOUS"]:
        atoms_ids = np.ascontiguousarray(atoms_ids)

    # Initialize particles with memory-efficient arrays
    particles = _initialize_atoms(atom_coordinates, atom_radii, atoms_ids, by_type)

    # Apply operations in optimal order to minimize memory usage
    if trim_within:
        particles.trim_within(subdomain)
    if trim_intersecting:
        particles.trim_intersecting(subdomain)
    if add_periodic:
        particles.add_periodic(subdomain)
    if set_own:
        particles.set_own(subdomain)

    return particles


def initialize_spheres(
    subdomain,
    spheres,
    radii=None,
    add_periodic=False,
    set_own=True,
    trim_intersecting=False,
    trim_within=False,
):
    """
    Initialize a list of spheres.
    Particles that do not cross the subdomain boundary are deleted
    If add_periodic: particles that cross the domain boundary will be add.
    If set_own: particles owned by a subdomain will be identified
    """

    if not radii:
        _spheres = spheres[:, 0:3]
        radii = spheres[:, 3]
    else:
        _spheres = spheres

    particles = _initialize_spheres(_spheres, radii)

    if trim_intersecting:
        particles.trim_intersecting(subdomain)

    if trim_within:
        particles.trim_within(subdomain)

    if add_periodic:
        particles.add_periodic(subdomain)

    if set_own:
        particles.set_own(subdomain)

    return particles
