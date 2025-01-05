### Core Utility Functions ###
import sys
import numpy as np

# from mpi4py import MPI
from mpi4py import MPI

comm = MPI.COMM_WORLD


__all__ = ["phase_exists"]


def raise_error():
    """Exit gracefuully."""
    MPI.Finalize()
    sys.exit()


def check_grid(subdomain, grid):
    """Esure solid voxel on each subprocess"""
    if np.sum(grid) == np.prod(subdomain.voxels):
        print(
            "This code requires at least 1 solid voxel in each subdomain. Please reorder processors!"
        )
        raise_error()


def check_inputs(mpi_size, subdomains, nodes, boundaries, inlet, outlet):
    """
    Ensure Input Parameters are Valid
    """
    check_input_nodes(nodes)
    check_subdomain_size(mpi_size, subdomains)
    check_boundaries(boundaries)
    check_inlet_outlet(boundaries, inlet, outlet)


def check_input_nodes(nodes):
    """Check Nodes are Positive"""
    error = False

    for n in nodes:
        if n <= 0:
            error = True
            print("Error: Nodes must be positive integer!")

    if error:
        raise_error()


def check_subdomain_size(mpi_size, subdomains):
    """
    Check subdomain size and ensure mpi size is equal to num_subdomains
    """
    error = False
    for n in subdomains:
        if n <= 0:
            error = True
            print("Error: Number of Subdomains must be positive integer!")

    num_subdomains = np.prod(subdomains)

    if mpi_size != num_subdomains:
        error = True
        print("Error: Number of MPI processes must equal number of subdomains!")

    if error:
        raise_error()


def check_boundaries(boundaries):
    """
    Check boundaries and boundary pairs
    """
    error = False
    for d in boundaries:
        for n in d:
            if n < 0 or n > 2:
                error = True
                print(
                    "Error: Allowable Boundary IDs are (0) None (1) Walls (2) Periodic"
                )
    if error:
        raise_error()


def check_inlet_outlet(boundaries, inlet, outlet):
    """
    Check inlet and outlet conditions
    """
    error = False

    # Inlet
    n_sum = 0
    for d_in, d_bound in zip(inlet, boundaries):
        for n_in, n_bound in zip(d_in, d_bound):
            if n_in != 0:
                n_sum = n_sum + 1
                if n_bound != 0:
                    error = True
                    print("Error: Boundary must be type (0) None at Inlet")
    if n_sum > 1:
        error = True
        print("Error: Only 1 Inlet Allowed")

    # Outlet
    n_sum = 0
    for d_in, d_bound in zip(outlet, boundaries):
        for n_in, n_bound in zip(d_in, d_bound):
            if n_in != 0:
                n_sum = n_sum + 1
                if n_bound != 0:
                    error = True
                    print("Error: Boundary must be type (0) None at Outlet")
    if n_sum > 1:
        error = True
        print("Error: Only 1 Outlet Allowed")

    if error:
        raise_error()


def check_padding(mpi_size, boundaries) -> bool:
    """
    Determine if padding needs to be added to the domain/subdomain

    Args:
        mpi_size (int): number of mpi processes
        boundaries (tuple): boundary conditions
    """
    pad = False
    if mpi_size > 1:
        pad = True

    for bound in boundaries:
        if bound[0] != 0:
            pad = True
        if bound[1] != 0:
            pad = True

    return pad


def unpad(grid, pad):
    """
    Unpad a padded array
    """
    _dim = grid.shape
    grid_out = grid[
        pad[0] : _dim[0] - pad[1], pad[2] : _dim[1] - pad[3], pad[4] : _dim[2] - pad[5]
    ]
    return np.ascontiguousarray(grid_out)


def constant_pad(grid, pad, pad_value):
    """
    Pad a grid with a constant value
    """
    grid = np.pad(
        grid,
        ((pad[0], pad[1]), (pad[2], pad[3]), (pad[4], pad[5])),
        "constant",
        constant_values=pad_value,
    )
    return grid


def own_grid(grid, own):
    """
    Pass array with only nodes owned py that process
    """
    grid_out = grid[own[0] : own[1], own[2] : own[3], own[4] : own[5]]

    return np.ascontiguousarray(grid_out)


def phase_exists(grid, phase):
    """
    Determine if phase exists in grid
    """
    phase_exists = False
    local_count = np.count_nonzero(grid == phase)
    global_count = comm.allreduce(local_count, op=MPI.SUM)

    if global_count > 0:
        phase_exists = True

    return phase_exists


def global_grid(grid, index, local_grid):
    """Take local grid from each process and combine into global grid"""
    grid[index[0] : index[1], index[2] : index[3], index[4] : index[5]] = local_grid
    return grid


def decompose_img(img, start, shape):
    """
    Decompose an image

    Parameters:
    - img: np.ndarray, the input array.
    - start: tuple, the starting index for the slice.
    - shape: tuple, the shape of the slice.

    Returns:
    - local_img: np.ndarray, the resulting wrapped slice.
    """
    z_max, y_max, x_max = img.shape
    dz, dy, dx = shape
    start_z, start_y, start_x = start

    # Create indices with wrapping
    z_indices = np.arange(start_z, start_z + dz) % z_max
    y_indices = np.arange(start_y, start_y + dy) % y_max
    x_indices = np.arange(start_x, start_x + dx) % x_max

    # Use advanced indexing to extract the slice
    return np.ascontiguousarray(img[np.ix_(z_indices, y_indices, x_indices)])


# def reconstruct_grid_to_root(subdomain,grid):
#     """This function (re)constructs a grid from all proccesses to root
#     """

#     if subdomain.ID == 0:
#         sd_all = np.empty((subdomain.domain.num_subdomains), dtype = object)
#         grid_all = np.empty((subdomain.domain.num_subdomains), dtype = object)
#         sd_all[0] = subdomain
#         grid_all[0] = grid
#         for neigh in range(1,subdomain.domain.num_subdomains):
#             sd_all[neigh] = comm.recv(source=neigh)
#             grid_all[neigh] = comm.recv(source=neigh)

#     if subdomain.ID > 0:
#         comm.send(subdomain,dest=0)
#         comm.send(grid,dest=0)

#     if subdomain.ID == 0:
#         grid_out = np.zeros(subdomain.domain.nodes)
#         for n in range(0,subdomain.domain.num_subdomains):
#             _own_grid = own_grid(grid_all[n],sd_all[n].index_own_nodes)
#             grid_out = global_grid(grid_out,sd_all[n].index_global,_own_grid)

#         return grid_out

#     return 0
