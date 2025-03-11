"""particles.py"""

import numpy as np
from ..core import subdomain_features
from . import _particles

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
    trim=False,
):
    """
    Initialize a list of particles (i.e. atoms, spheres).
    Particles that do not cross the subdomain boundary are deleted
    If add_periodic: particles that cross the domain boundary will be add.
    If set_own: particles owned by a subdomain will be identified
    """

    particles = _particles._initialize_atoms(
        atom_coordinates, atom_radii, atoms_ids, by_type
    )

    if trim:
        particles.trim(subdomain)

    if add_periodic:
        particles.add_periodic(subdomain)

    if set_own:
        particles.set_own(subdomain)

    return particles


def initialize_spheres(
    subdomain, spheres, radii=None, add_periodic=False, set_own=True, trim=False
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

    particles = _particles.initialize_spheres(_spheres, radii)

    if trim:
        particles.trim(subdomain)

    if add_periodic:
        particles.add_periodic(subdomain)

    if set_own:
        particles.set_own(subdomain)

    return particles


def collect_boundary_crossings(sphere_data, domain_box):
    """
    Determine if a sphere crosses the domain boundaries.

    Args:
        sphere_data (list or tuple): Sphere data as (x, y, z, radius).
        domain_box (numpy array): Domain boundaries as a 2D array [[x_min, x_max], [y_min, y_max], [z_min, z_max]].

    Returns:
        numpy.ndarray: Array of size 6 indicating boundary crossings:
                       [x_min_cross, x_max_cross, y_min_cross, y_max_cross, z_min_cross, z_max_cross].
    """
    from ..core import orientation

    boundary_features = []
    features = orientation.get_features()
    for feature_id in features:
        boundary = [0, 0, 0]
        for n, val in enumerate(feature_id):
            if val == -1:
                if sphere_data[n] - sphere_data[3] <= domain_box[n][0]:
                    boundary[n] = 1
            elif val == 1:
                if sphere_data[n] + sphere_data[3] >= domain_box[n][1]:
                    boundary[n] = 1

        if sum(boundary) == sum(abs(_id) for _id in feature_id):
            boundary_features.append(feature_id)

    return boundary_features


def is_inside_domain(sphere_coordinates, domain_box):
    """
    Determine if a sphere is within the domain boundaries.

    Args:
        sphere_data (list or tuple): Coordinates of the sphere (x, y, z).
        domain_box (list of tuples): Domain boundaries [(x_min, x_max), (y_min, y_max), (z_min, z_max)].

    Returns:
        bool: True if the sphere is within the domain boundaries, False otherwise.
    """
    return all(
        dim_min <= coord <= dim_max
        for coord, (dim_min, dim_max) in zip(sphere_coordinates, domain_box)
    )


def reflect_boundary_sphere(
    sphere_data,
    boundary_feature,
    domain_length,
    periodic_features,
    periodic_corrections,
):
    """
    Add spheres that cross periodic boundaries by using periodic_correction form features
    """
    periodic_spheres = np.zeros([len(boundary_feature), 4])
    periodic_spheres[:, 3] = sphere_data[3]

    for n, feature_id in enumerate(boundary_feature):
        if feature_id in periodic_features:

            periodic_spheres[n][0:3] = sphere_data[0:3] + [
                corr * len
                for corr, len in zip(periodic_corrections[feature_id], domain_length)
            ]

    return periodic_spheres


def gen_periodic_spheres(subdomain, sphere_data):
    """
    Add spheres that extend pass boundary and are periodic
    """

    periodic_features = subdomain_features.collect_periodic_features(subdomain.features)

    periodic_corrections = subdomain_features.collect_periodic_corrections(
        subdomain.features
    )

    num_spheres = sphere_data.shape[0]
    all_periodic_spheres = []

    for n_sphere in range(num_spheres):
        inside_domain = is_inside_domain(sphere_data[n_sphere, :], subdomain.domain.box)
        if inside_domain:
            boundary_features = collect_boundary_crossings(
                sphere_data[n_sphere, :], subdomain.domain.box
            )

            # Ignore internal spheres
            if boundary_features:
                periodic_spheres = reflect_boundary_sphere(
                    sphere_data[n_sphere, :],
                    boundary_features,
                    subdomain.domain.length,
                    periodic_features,
                    periodic_corrections,
                )

                all_periodic_spheres.extend(periodic_spheres)

    if all_periodic_spheres:
        sphere_data = np.concatenate((sphere_data, np.array(all_periodic_spheres)))

    return sphere_data


def gen_periodic_atoms(subdomain, atom_locations, atom_types, atom_cutoff):
    """
    Add atoms that extend pass boundary and are periodic
    """

    periodic_features = subdomain_features.collect_periodic_features(subdomain.features)

    periodic_corrections = subdomain_features.collect_periodic_corrections(
        subdomain.features
    )

    atom = np.zeros(4)
    periodic_atom_locations = []
    periodic_atom_types = []
    for atom_location, atom_type in zip(atom_locations, atom_types):

        atom[0:3] = atom_location
        atom[3] = atom_cutoff[atom_type]

        inside_domain = is_inside_domain(atom, subdomain.domain.box)
        if inside_domain:
            boundary_features = collect_boundary_crossings(atom, subdomain.domain.box)
            if boundary_features:

                periodic_atoms = reflect_boundary_sphere(
                    atom,
                    boundary_features,
                    subdomain.domain.length,
                    periodic_features,
                    periodic_corrections,
                )
                periodic_atom_locations.extend(periodic_atoms[:, 0:3])
                periodic_atom_types.extend([atom_type] * periodic_atoms.shape[1])

    if periodic_atom_locations:
        atom_locations = np.concatenate(
            (atom_locations, np.array(periodic_atom_locations), atom_locations)
        )
        atom_types = np.concatenate(
            (atom_types, np.array(periodic_atom_types, dtype=int))
        )
    return atom_locations, atom_types
