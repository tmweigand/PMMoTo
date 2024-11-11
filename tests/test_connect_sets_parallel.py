import numpy as np
from mpi4py import MPI
import pmmoto


def test_connect_sets_parallel():

    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()

    subdomain_map = [2, 2, 2]  # Specifies how Domain is broken among procs
    voxels = [100, 100, 100]  # Total Number of Nodes in Domain

    box = [[0, 3.945410e-01], [0, 3.945410e-01], [0, 3.945410e-01]]

    ## Ordering for Inlet/Outlet ( (-x,+x) , (-y,+y) , (-z,+z) )
    boundaries = [[0, 0], [0, 0], [0, 0]]  # 0: Nothing Assumed  1: Walls 2: Periodic
    inlet = [[0, 0], [0, 0], [0, 0]]
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

    connected_sets = pmmoto.filters.connect_all_phases(pm, sd)

    if save_data:
        kwargs = {"sets": connected_sets}
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
