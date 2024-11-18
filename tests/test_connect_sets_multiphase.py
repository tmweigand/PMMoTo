import numpy as np
from mpi4py import MPI
import pmmoto
import time

import pytest


@pytest.mark.skip(reason="TBD")
def test_connected_sets_multiphase():

    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()

    subdomain_map = [1, 1, 1]
    voxels = [100, 100, 100]

    box = [[0, 3.945410e-01], [0, 3.945410e-01], [0, 3.945410e-01]]
    file = "tests/testDomains/50pack.out"
    boundaries = [[0, 0], [0, 0], [0, 0]]
    inlet = [[1, 0], [0, 0], [0, 0]]
    outlet = [[0, 0], [0, 0], [0, 0]]

    # Multiphase
    num_fluid_phases = 2

    w_inlet = [[1, 0], [0, 0], [0, 0]]
    nw_inlet = [[1, 0], [0, 0], [0, 0]]
    mp_inlet = {0: w_inlet, 1: nw_inlet}

    w_outlet = [[0, 0], [0, 0], [0, 0]]
    nw_outlet = [[0, 0], [0, 0], [0, 0]]
    mp_outlet = {0: w_outlet, 1: nw_outlet}

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

    mp = pmmoto.core.initialize_multiphase(
        porous_media=pm,
        num_phases=num_fluid_phases,
        inlets=mp_inlet,
        outlets=mp_outlet,
    )

    mp = pmmoto.domain_generation.gen_mp_constant(mp, fluid_phase=1)

    mp.grid[4:20, 4:20, 4:20] = 2

    connected_sets = pmmoto.filters.connect_all_phases(
        mp, return_grid=True, return_set=True
    )

    if save_data:

        kwargs = {"sets": connected_sets["grid"]}
        pmmoto.io.save_grid_data(
            "dataOut/test_connect_sets_multiphase_parallel_grid", sd, mp.grid, **kwargs
        )

        kwargs = {
            "inlet": "subdomain_data.inlet",
            "outlet": "subdomain_data.outlet",
            "proc": "proc_ID",
        }
        pmmoto.io.save_set_data(
            "dataOut/test_connect_sets_parallel_multiphase",
            sd,
            connected_sets["sets"],
            **kwargs,
        )


if __name__ == "__main__":
    test_connected_sets_multiphase()
    MPI.Finalize()
