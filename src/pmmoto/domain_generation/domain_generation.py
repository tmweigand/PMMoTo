import numpy as np
from mpi4py import MPI
from pmmoto.domain_generation import _domain_generation
from pmmoto.core import communication
from pmmoto.core import utils
from pmmoto.core import porousmedia
from pmmoto.core import orientation
from pmmoto.core import subdomain_features


__all__ = [
    "gen_pm_spheres_domain",
    "gen_pm_atom_domain",
    "gen_pm_verlet_spheres_domain",
    "gen_pm_verlet_atom_domain",
    "gen_pm_inkbottle",
    "gen_mp_constant",
    "gen_mp_from_grid",
    "gen_periodic_spheres",
    "gen_periodic_atoms",
    "is_inside_domain",
    "collect_boundary_crossings",
    "reflect_boundary_sphere",
]


def gen_pm_spheres_domain(subdomain, spheres, res_size=0):
    """
    Generate binary domain (pm) from sphere data that contains radii
    """
    _grid = _domain_generation.gen_pm_sphere(
        subdomain.coords[0], subdomain.coords[1], subdomain.coords[2], spheres
    )
    pm = porousmedia.gen_pm(subdomain, _grid, res_size)
    pm.grid = communication.update_buffer(subdomain, pm.grid)

    utils.check_grid(subdomain, pm.grid)

    return pm


def gen_pm_atom_domain(subdomain, atom_locations, atom_types, atom_cutoff, res_size=0):
    """
    Generate binary domain (pm) from atom data, types and cutoff
    """
    _grid = _domain_generation.gen_pm_atom(
        subdomain.coords[0],
        subdomain.coords[1],
        subdomain.coords[2],
        atom_locations,
        atom_types,
        atom_cutoff,
    )

    pm = porousmedia.gen_pm(subdomain, _grid, res_size)
    pm.grid = communication.update_buffer(subdomain, pm.grid)

    utils.check_grid(subdomain, pm.grid)

    return pm


def gen_pm_verlet_spheres_domain(subdomain, spheres, verlet=[1, 1, 1], res_size=0):
    """
    Generate binary domain (pm) from sphere data that contains radii
       using verlet domains
    """
    _grid = _domain_generation.gen_pm_verlet_sphere(
        verlet, subdomain.coords[0], subdomain.coords[1], subdomain.coords[2], spheres
    )
    pm = porousmedia.gen_pm(subdomain, _grid, res_size)
    pm.grid = communication.update_buffer(subdomain, pm.grid)

    utils.check_grid(subdomain, pm.grid)

    return pm


def gen_pm_verlet_atom_domain(
    subdomain, atom_locations, atom_types, atom_cutoff, verlet=[1, 1, 1], res_size=0
):
    """
    Generate binary domain (pm) from atom data, types and cutoff
       using verlet domains
    """
    _grid = _domain_generation.gen_pm_verlet_atom(
        verlet,
        subdomain.coords[0],
        subdomain.coords[1],
        subdomain.coords[2],
        atom_locations,
        atom_types,
        atom_cutoff,
    )

    pm = porousmedia.gen_pm(subdomain, _grid, res_size)
    pm.grid = communication.update_buffer(subdomain, pm.grid)

    utils.check_grid(subdomain, pm.grid)

    return pm


def gen_pm_inkbottle(subdomain, domain_data, res_size=0):
    """ """
    subdomain.update_domain_size(domain_data)
    _grid = _domain_generation.gen_pm_inkbottle(
        subdomain.coords[0], subdomain.coords[1], subdomain.coords[2]
    )
    pm = porousmedia.gen_pm(subdomain, _grid, res_size)
    utils.check_grid(subdomain, pm.grid)
    pm.grid = communication.update_buffer(subdomain, pm.grid)

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


def reflect_boundary_sphere(sphere_data, boundary_feature, subdomain):
    """
    Add spheres that cross periodic boundaries
    """

    periodic_features = subdomain_features.collect_periodic_features(subdomain.features)

    periodic_corrections = subdomain_features.collect_periodic_corrections(
        subdomain.features
    )

    periodic_spheres = []
    for feature_id in boundary_feature:
        if feature_id in periodic_features:
            periodic_spheres.append(
                sphere_data * subdomain.feature[feature_id].periodic_correction
            )
    # for f_index in orientation.faces:
    #     face = orientation.faces[f_index]
    #     index = face["argOrder"][0]
    #     if crosses_boundary[index] and boundary_types[index] == 2:
    #         shift_sphere = sphere_data[index] + face["dir"] * domain_length[index]
    #         add_sphere = list(sphere_data)
    #         add_sphere[index] = shift_sphere
    #         periodic_spheres.append(add_sphere)

    # for e_index in orientation.edges:
    #     edge = orientation.edges[e_index]
    #     add_sphere = list(sphere_data)
    #     periodic = [False, False]
    #     for n_face, f_index in enumerate(edge["faceIndex"]):
    #         face = orientation.faces[f_index]
    #         index = face["argOrder"][0]
    #         if crosses_boundary[index] and boundary_types[index] == 2:
    #             shift_sphere = sphere_data[index] + face["dir"] * domain_length[index]
    #             add_sphere[index] = shift_sphere
    #             periodic[n_face] = True
    #     if all(periodic):
    #         periodic_spheres.append(add_sphere)

    # for c_index in orientation.corners:
    #     corner = orientation.corners[c_index]
    #     add_sphere = list(sphere_data)
    #     periodic = [False, False, False]
    #     for n_face, f_index in enumerate(corner["faceIndex"]):
    #         face = orientation.faces[f_index]
    #         index = face["argOrder"][0]
    #         if crosses_boundary[index] and boundary_types[index] == 2:
    #             shift_sphere = sphere_data[index] + face["dir"] * domain_length[index]
    #             add_sphere[index] = shift_sphere
    #             periodic[n_face] = True
    #     if all(periodic):
    #         periodic_spheres.append(add_sphere)

    return periodic_spheres


def gen_periodic_spheres(subdomain, sphere_data):
    """
    Add spheres that extend pass boundary and are periodic
    """
    domain = subdomain.domain.size_domain
    domain_length = subdomain.domain.length_domain
    boundaries = np.array(subdomain.domain.boundaries).flatten()

    num_spheres = sphere_data.shape[0]

    all_periodic_spheres = []
    for n_sphere in range(num_spheres):
        inside_domain = is_inside_domain(sphere_data[n_sphere, :], domain)
        if inside_domain:
            crosses_boundary = collect_boundary_crossings(
                sphere_data[n_sphere, :], domain
            )

            # Pass internal spheres
            if np.sum(crosses_boundary) == 0:
                continue
            periodic_spheres = reflect_boundary_sphere(
                sphere_data[n_sphere, :], crosses_boundary, domain_length, boundaries
            )

            all_periodic_spheres.extend(periodic_spheres)

    sphere_data = np.concatenate((np.array(all_periodic_spheres), sphere_data))
    return sphere_data


def gen_periodic_atoms(subdomain, atom_locations, atom_types, atom_cutoff):
    """
    Add atoms that extend pass boundary and are periodic
    """
    domain = subdomain.domain.size_domain
    res = subdomain.domain.voxel
    domain_length = subdomain.domain.length_domain
    boundaries = np.array(subdomain.domain.boundaries).flatten()

    num_atom = atom_locations.shape[0]

    sphere = np.zeros(4)
    periodic_atom_locations = []
    periodic_atom_types = []
    for n_atom in range(num_atom):

        sphere[0:3] = atom_locations[n_atom]
        sphere[3] = atom_cutoff[atom_types[n_atom]]

        inside_domain = is_inside_domain(sphere, domain)
        if inside_domain:
            crosses_boundary = is_boundary_sphere(sphere, res, domain)

            # Pass internal spheres
            if np.sum(crosses_boundary) == 0:
                continue
            add_periodic_atoms = reflect_boundary_sphere(
                sphere, crosses_boundary, domain_length, boundaries
            )

            for atom in add_periodic_atoms:
                periodic_atom_locations.extend([atom[0:3]])
                periodic_atom_types.extend([atom_types[n_atom]])

    if periodic_atom_locations:
        atom_locations = np.concatenate(
            (np.array(periodic_atom_locations), atom_locations)
        )
        atom_types = np.concatenate(
            (np.array(periodic_atom_types, dtype=int), atom_types)
        )
    return atom_locations, atom_types
