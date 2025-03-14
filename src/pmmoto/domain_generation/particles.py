"""particles.py"""

from ._particles import _initialize_atoms, _initialize_spheres

__all__ = [
    "initialize_atoms",
    "initialize_spheres",
]


def initialize_atoms(
    subdomain,
    atom_coordinates,
    atom_radii,
    atoms_ids,
    by_type=False,
    add_periodic=False,
    set_own=True,
    trim_intersecting=False,
    trim_within=False,
):
    """
    Initialize a list of particles (i.e. atoms, spheres).
    Particles that do not cross the subdomain boundary are deleted
    If add_periodic: particles that cross the domain boundary will be add.
    If set_own: particles owned by a subdomain will be identified
    """

    particles = _initialize_atoms(atom_coordinates, atom_radii, atoms_ids, by_type)

    if trim_intersecting:
        particles.trim_intersecting(subdomain)

    if trim_within:
        particles.trim_within(subdomain)

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
