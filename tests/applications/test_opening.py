import numpy as np
from mpi4py import MPI
import pmmoto
import time

import pytest


@pytest.mark.skip(reason="TBD")
def test_opening():

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

    # Multiphase
    num_fluid_phases = 2

    w_inlet = [[0, 0], [0, 0], [0, 0]]
    nw_inlet = [[0, 0], [0, 0], [0, 0]]
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

    mp = pmmoto.domain_generation.gen_mp_constant(mp, fluid_phase=2)

    out_saturations = [0.80]  # ,0.65,0.50,0.35,0.20]

    mp = pmmoto.filters.multiphase.calcOpenSW(domain, sd, mp, out_saturations, 0.9, 10)

    if save_data:

        kwargs = {}
        pmmoto.io.save_grid_data("dataOut/test_opening", sd, mp.grid, **kwargs)


if __name__ == "__main__":
    test_opening()
    MPI.Finalize()
