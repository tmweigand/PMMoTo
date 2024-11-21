import numpy as np
from mpi4py import MPI
import pmmoto

import pytest


@pytest.mark.skip(reason="TBD")
def test_connect_sets():

    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()

    subdomains = [1, 1, 1]  # Specifies how Domain is broken among procs
    voxels = [100, 100, 100]  # Total Number of Nodes in Domain

    box = [[0, 3.945410e-01], [0, 3.945410e-01], [0, 3.945410e-01]]

    ## Ordering for Inlet/Outlet ( (-x,+x) , (-y,+y) , (-z,+z) )
    boundary_types = [
        [0, 0],
        [0, 0],
        [0, 0],
    ]  # 0: Nothing Assumed  1: Walls 2: Periodic
    inlet = [[0, 0], [0, 0], [0, 0]]
    outlet = [[0, 0], [0, 0], [0, 0]]

    file = "tests/testDomains/50pack.out"

    save_data = True

    sd = pmmoto.initialize(
        box=box,
        subdomains=subdomains,
        voxels=voxels,
        boundary_types=boundary_types,
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

    connected_sets = pmmoto.filters.connect_components(pm.grid, sd)

    if save_data:

        kwargs = {"sets": connected_sets}
        pmmoto.io.save_grid_data_serial(
            "data_out/test_connects_sets_grid", sd, pm.grid, **kwargs
        )


if __name__ == "__main__":
    test_connect_sets()
    MPI.Finalize()
