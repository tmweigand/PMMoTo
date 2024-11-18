import numpy as np
from mpi4py import MPI
import pmmoto

import pytest


@pytest.mark.skip(reason="TBD")
def test_connect_sets_parallel():

    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()

    subdomain_map = [2, 2, 2]  # Specifies how Domain is broken among procs
    voxels = [100, 100, 100]  # Total Number of Nodes in Domain

    box = [[0, 3.945410e-01], [0, 3.945410e-01], [0, 3.945410e-01]]

    ## Ordering for Inlet/Outlet ( (-x,+x) , (-y,+y) , (-z,+z) )
    boundaries = [[0, 0], [0, 0], [0, 0]]  # 0: Nothing Assumed  1: Walls 2: Periodic
    inlet = [[1, 0], [0, 0], [0, 0]]
    outlet = [[0, 0], [0, 0], [0, 0]]

    file = "tests/testDomains/50pack.out"

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

    sphere_data, _ = pmmoto.io.read_sphere_pack_xyzr_domain(file)
    pm = pmmoto.domain_generation.gen_pm_spheres_domain(
        sd,
        sphere_data,
    )

    grid = np.ones_like(pm.grid)
    grid[2:-2, 2:-2, 2:-2] = sd.rank
    grid = pmmoto.core.communication.update_buffer(sd, grid)

    connected_grid = pmmoto.filters.connect_components(grid, sd)
    label_phase_map = pmmoto.filters.gen_grid_to_label_map(grid, connected_grid)
    inlet_label_map = pmmoto.filters.gen_inlet_label_map(sd, connected_grid)
    outlet_label_map = pmmoto.filters.gen_outlet_label_map(sd, connected_grid)

    print(label_phase_map, inlet_label_map, outlet_label_map)

    if save_data:
        kwargs = {"cc": connected_grid}
        pmmoto.io.save_grid_data_parallel(
            "data_out/test_connects_sets_parallel_grid", sd, domain, pm.grid, **kwargs
        )

    # kwargs = {
    #     "inlet": "subdomain_data.inlet",
    #     "outlet": "subdomain_data.outlet",
    #     "proc": "proc_ID",
    # }

    # pmmoto.io.save_set_data(
    #     "dataOut/test_connect_sets_parallel", sd, connected_sets["sets"], **kwargs
    # )


if __name__ == "__main__":
    test_connect_sets_parallel()
    MPI.Finalize()
