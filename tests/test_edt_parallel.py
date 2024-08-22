import numpy as np
from mpi4py import MPI

import pmmoto
import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)
import matplotlib.pyplot as plt


def my_function():

    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()

    subdomain_map = [2, 2, 2]  # Specifies how domain is broken among processes
    voxels = [300, 300, 300]  # Total Number of Nodes in Domain

    box = [[0, 3.945410e-01], [0, 3.945410e-01], [0, 3.945410e-01]]

    ## Ordering for Inlet/Outlet ( (-x,+x) , (-y,+y) , (-z,+z) )
    boundaries = [[2, 2], [2, 2], [2, 2]]  # 0: Nothing Assumed  1: Walls 2: Periodic
    inlet = [[0, 0], [0, 0], [0, 0]]
    outlet = [[0, 0], [0, 0], [0, 0]]

    file = "tests/testDomains/50pack.out"

    testSerial = True
    testAlgo = True

    sd, domain = pmmoto.initialize(
        box=box,
        subdomain_map=subdomain_map,
        voxels=voxels,
        boundaries=boundaries,
        inlet=inlet,
        outlet=outlet,
        rank=rank,
        mpi_size=size,
        reservoir_voxels=2,
    )

    sphere_data, domain_data = pmmoto.io.read_sphere_pack_xyzr_domain(file)
    pm = pmmoto.domain_generation.gen_pm_spheres_domain(sd, sphere_data, domain_data)

    edt = pmmoto.filters.calc_edt(domain, sd, pm.grid)

    ### Save Grid Data where kwargs are used for saving other grid data (i.e. EDT, Medial Axis)
    pmmoto.io.save_grid_data("dataOut/parallel_grid", sd, pm.grid, dist=edt)


if __name__ == "__main__":
    my_function()
    MPI.Finalize()
