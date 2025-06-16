"""Example: Generate the pore size distribution of a sphere packing."""

from mpi4py import MPI
import pmmoto

comm = MPI.COMM_WORLD
rank = comm.Get_rank()


def md_porous_media() -> None:
    """Generate the pore size distribution of a sphere pack.

    To run this file:
        python examples/md_porous_media.py
    or to run in parallel:
        set subdomains = (2,2,2)
        mpirun -np 8 python examples/md_porous_media.py
    """
    voxels = (300, 300, 300)
    subdomains = (1, 1, 1)

    sd = pmmoto.initialize(voxels, rank=rank, subdomains=subdomains)
