import numpy as np
from mpi4py import MPI
import pmmoto
import time


def test_porous_media():

    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()

    subdomain_map = [1, 1, 1]
    voxels = [100, 100, 100]

    box = [[0, 3.945410e-01], [0, 3.945410e-01], [0, 3.945410e-01]]
    file = "tests/testDomains/50pack.out"
    boundaries = [[2, 2], [2, 2], [2, 2]]
    inlet = [[0, 0], [0, 0], [0, 0]]
    outlet = [[0, 0], [0, 0], [0, 0]]

    save_data = True

    sd, domain = pmmoto.initialize(
        box=box,
        subdomain_map=subdomain_map,
        voxels=voxels,
        boundaries=boundaries,
        inlet=inlet,
        outlet=outlet,
        rank=rank,
        mpi_size=size,
        reservoir_voxels=0,
    )

    sphere_data, domain_data = pmmoto.io.read_sphere_pack_xyzr_domain(file)
    pm = pmmoto.domain_generation.gen_pm_spheres_domain(
        sd,
        sphere_data,
    )

    pm.get_porosity()
    print(pm.porosity)

    if save_data:

        kwargs = {}
        pmmoto.io.save_grid_data("dataOut/test_porous_media", sd, pm.grid, **kwargs)


if __name__ == "__main__":
    test_porous_media()
    MPI.Finalize()
