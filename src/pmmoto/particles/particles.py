"""particles.py

Particle initialization and utility functions for PMMoTo.
"""

from __future__ import annotations
from typing import TYPE_CHECKING, Any
import numpy as np
from numpy.typing import NDArray

from ._particles import _initialize_atoms
from ._particles import _initialize_spheres
from ._particles import _initialize_cylinders
from ._particles import PySphereList, AtomMap, PyCylinderList
from .atom_universal_force_field import atom_universal_force_field

if TYPE_CHECKING:
    from ..core.subdomain import Subdomain
    from ..core.subdomain_padded import PaddedSubdomain
    from ..core.subdomain_verlet import VerletSubdomain

__all__ = [
    "convert_atoms_elements_to_ids",
    "uff_radius",
    "initialize_atoms",
    "initialize_spheres",
    "initialize_cylinders",
]


def convert_atoms_elements_to_ids(atom_elements: list[str]) -> NDArray[np.integer[Any]]:
    """Convert a list of atom names (C, H, N, O, etc.) to atomic IDs.

    Args:
        atom_elements (List[str]): List of atom element names.

    Returns:
        np.ndarray: Array of atomic numbers.

    """
    element_table = atom_universal_force_field()

    atom_ids = np.zeros(len(atom_elements), dtype=int)  # Initialize array with zeros

    for n, element in enumerate(atom_elements):
        if element in element_table:
            atom_ids[n] = element_table[element][0]  # Store atomic number

    return atom_ids


def _load_uff_data(file_name: None | str = None) -> dict[str | int, tuple[int, float]]:
    """Read universal force field file for atom radius lookup.

    Can query the dictionary based on:
        - Atom Name
        - Atomic Number (i.e. 1 = H, 6 = C)

    Args:
        file_name (str, optional): Path to UFF data file.

    Returns:
        dict: Mapping of atom names/numbers to (atomic_number, radius).

    """
    if file_name is None:
        element_table = atom_universal_force_field()
    else:
        element_table = {}
        with open(file_name, "r") as file:
            next(file)  # skip header
            for line in file:
                parts = line.split()
                if len(parts) < 3:
                    continue
                atomic_number = int(parts[0])
                name = parts[1]
                radius = float(parts[2]) / 2.0  # Convert diameter to radius
                element_table[name] = atomic_number, radius
                element_table[atomic_number] = atomic_number, radius

    return element_table


def uff_radius(
    atom_names: None | list[str] = None, atomic_numbers: None | list[int] = None
) -> dict[int, float]:
    """Collect the radius by Atom Name or Atomic Number, but not both.

    Units of radii are Angstroms!

    Args:
        atom_names (List[str], optional): List of element names.
        atomic_numbers (List[int], optional): List of atomic numbers.

    Returns:
        Dict[int, float]: Dictionary mapping atomic numbers to their radii.

    Raises:
        ValueError: If both or neither atom_names and atomic_numbers are provided.

    """
    if (atom_names is None) == (atomic_numbers is None):
        raise ValueError(
            "Provide either 'atom_names' or 'atomic_numbers', but not both."
        )

    all_uff_radii = _load_uff_data()
    radii: dict[int, float] = {}

    if atom_names is not None:
        for name in atom_names:
            if name in all_uff_radii:
                atomic_number, radius = all_uff_radii[name]
                radii[atomic_number] = radius
    elif atomic_numbers is not None:
        for num in atomic_numbers:
            if num in all_uff_radii:
                atomic_number, radius = all_uff_radii[num]
                radii[atomic_number] = radius
    else:
        raise ValueError("Input data must be a list of atom names or atomic numbers.")

    return radii


def initialize_atoms(
    subdomain: Subdomain | PaddedSubdomain | VerletSubdomain,
    atom_coordinates: NDArray[np.floating[Any]],
    atom_radii: NDArray[np.floating[Any]],
    atom_ids: NDArray[np.integer[Any]],
    atom_masses: None | NDArray[np.integer[Any]] = None,
    by_type: bool = False,
    add_periodic: bool = False,
    set_own: bool = True,
    trim_intersecting: bool = False,
    trim_within: bool = False,
) -> AtomMap:
    """Initialize a list of particles efficiently with memory management.

    Args:
        subdomain: Domain subdivision object.
        atom_coordinates (np.ndarray): Array of shape (n_atoms, 3) with xyz coordinates.
        atom_radii (dict or np.ndarray): Dictionary of atom types to radii or array.
        atom_ids (np.ndarray): Array of atom type IDs.
        atom_masses (dict, optional): Dictionary of atom masses.
        by_type (bool, optional): Whether to organize atoms by type.
        add_periodic (bool, optional): Add periodic images at boundaries.
        set_own (bool, optional): Mark particles owned by this subdomain.
        trim_intersecting (bool, optional): Remove particles intersecting boundary.
        trim_within (bool, optional): Remove particles fully within boundary.

    Returns:
        object: Python Class that wraps atoms. See _particles.pyx.

    """
    if not atom_coordinates.flags["C_CONTIGUOUS"]:
        atom_coordinates = np.ascontiguousarray(atom_coordinates)

    if isinstance(atom_radii, np.ndarray) and not atom_radii.flags["C_CONTIGUOUS"]:
        atom_radii = np.ascontiguousarray(atom_radii)

    if not atom_ids.flags["C_CONTIGUOUS"]:
        atom_ids = np.ascontiguousarray(atom_ids)

    # Initialize particles
    particles = _initialize_atoms(
        atom_coordinates, atom_radii, atom_ids, atom_masses, by_type
    )

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
    subdomain: Subdomain | PaddedSubdomain | VerletSubdomain,
    spheres: NDArray[np.floating[Any]],
    radii: None | NDArray[np.floating[Any]] = None,
    add_periodic: bool = False,
    set_own: bool = True,
    trim_intersecting: bool = False,
    trim_within: bool = False,
) -> PySphereList:
    """Initialize a list of spheres.

    Particles that do not cross the subdomain boundary are deleted.
    If add_periodic: particles that cross the domain boundary will be added.
    If set_own: particles owned by a subdomain will be identified.

    Args:
        subdomain: Domain subdivision object.
        spheres (np.ndarray): Array of sphere positions and radii.
        radii (np.ndarray, optional): Array of radii if not included in spheres.
        add_periodic (bool, optional): Add periodic images at boundaries.
        set_own (bool, optional): Mark particles owned by this subdomain.
        trim_intersecting (bool, optional): Remove particles intersecting boundary.
        trim_within (bool, optional): Remove particles fully within boundary.

    Returns:
        AtomMap: Python Class that wraps spheres. See _particles.pyx.

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


def initialize_cylinders(
    subdomain: Subdomain | PaddedSubdomain | VerletSubdomain,
    cylinders: NDArray[np.floating[Any]],
) -> PyCylinderList:
    """Initialize a list of cylindersd

    Args:
        subdomain (Subdomain | PaddedSubdomain | VerletSubdomain): pmmoto subdomain
        cylinders (NDArray[np.floating[Any]]): aray of cylinder data

    Returns:
        PyCylinderList: _description_

    """
    particles = _initialize_cylinders(cylinders)

    return particles
