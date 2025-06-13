"""Example: Generate the pore size distribution of a sphere packing."""

from mpi4py import MPI
import pmmoto

comm = MPI.COMM_WORLD
rank = comm.Get_rank()


def sphere_pack_psd() -> None:
    """Generate the pore size distribution of a sphere pack.

    To run this file:
        python examples/sphere_pack_psd.py
    or to run in parallel:
        set subdomains = (2,2,2)
        mpirun -np 8 python examples/sphere_pack_psd.py
    """
    voxels = (300, 300, 300)
    subdomains = (1, 1, 1)

    sd = pmmoto.initialize(voxels, rank=rank, subdomains=subdomains)
    sphere_pack_file = "examples/sphere_pack_psd/sphere_pack.in"
    pm = pmmoto.io.data_read.read_sphere_pack_xyzr_domain(sphere_pack_file)
    psd = pmmoto.filters.porosimetry.pore_size_distribution(sd, pm)
