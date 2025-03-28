### Core Utility Functions ###
import sys
import logging
import numpy as np

# from mpi4py import MPI
from mpi4py import MPI
from . import communication

comm = MPI.COMM_WORLD

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)


__all__ = ["phase_exists", "constant_pad_img", "unpad", "determine_maximum"]


def raise_error():
    """Exit gracefully."""
    MPI.Finalize()
    sys.exit()


def check_grid(subdomain, grid):
    """Ensure solid voxel on each subprocess"""
    if np.sum(grid) == np.prod(subdomain.voxels):
        logging.warning(
            f"Many functions in pmmoto require at least 1 solid voxel in each subdomain. Process with rank: {subdomain.rank} is all pores."
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


def unpad(img, pad):
    """
    Removes padding from a NumPy array.

    Parameters:
        img (np.ndarray): The padded array.
        pad (list or tuple): Padding amounts in the format [[before_0, after_0], [before_1, after_1], ...].

    Returns:
        np.ndarray: The unpadded array.
    """
    slices = tuple(slice(p[0], img.shape[i] - p[1]) for i, p in enumerate(pad))
    return np.ascontiguousarray(img[slices])


def constant_pad_img(img, pad, pad_value):
    """
    Pad a grid with a constant value
    """

    img = np.pad(
        img,
        ((pad[0][0], pad[0][1]), (pad[1][0], pad[1][1]), (pad[2][0], pad[2][1])),
        "constant",
        constant_values=pad_value,
    )
    return img


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


def determine_maximum(img):
    """
    Determine the global maximum of an input image
    """
    local_max = np.amax(img)

    proc_local_max = communication.all_gather(local_max)

    return np.amax(proc_local_max)


def global_grid(grid, index, local_grid):
    """Take local grid from each process and combine into global grid"""
    grid[index[0] : index[1], index[2] : index[3], index[4] : index[5]] = local_grid
    return grid


def decompose_img(img, start, shape, padded_img=False):
    """
    Decompose an image

    Parameters:
    - img: np.ndarray, the input array.
    - start: tuple, the starting index for the slice.
    - shape: tuple, the shape of the slice.

    Returns:
    - local_img: np.ndarray, the resulting wrapped slice.
    """

    # Create indices with wrapping
    index = [None, None, None]
    for n, (_start, _shape) in enumerate(zip(start, shape)):
        index[n] = np.arange(_start, _start + _shape) % img.shape[n]

    # Use advanced indexing to extract the slice
    return img[np.ix_(index[0], index[1], index[2])]


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
