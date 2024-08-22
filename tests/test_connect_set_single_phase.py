import numpy as np
from mpi4py import MPI
import pmmoto
import time


def test_connected_sets():

    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()

    subdomains = [1, 1, 1]  # Specifies how Domain is broken among procs
    nodes = [100, 100, 100]  # Total Number of Nodes in Domain

    ## Ordering for Inlet/Outlet ( (-x,+x) , (-y,+y) , (-z,+z) )
    boundaries = [[2, 2], [2, 2], [0, 0]]  # 0: Nothing Assumed  1: Walls 2: Periodic
    inlet = [[0, 0], [0, 0], [1, 0]]
    outlet = [[0, 0], [0, 0], [0, 1]]

    file = "./testDomains/50pack.out"

    save_data = True

    sd = pmmoto.initialize(rank, size, subdomains, nodes, boundaries, inlet, outlet)
    sphere_data, domain_data = pmmoto.io.read_sphere_pack_xyzr_domain(file)
    pm = pmmoto.domain_generation.gen_pm_spheres_domain(sd, sphere_data, domain_data)

    pm.grid[0:20, 0:20, 0:20] = 2
    pm.grid[-10:, -10:, -10:] = 2

    connected_sets = pmmoto.filters.connect_single_phase(
        pm, pm.inlet, pm.outlet, phase=1
    )

    for set in connected_sets["sets"].sets:
        _set = connected_sets["sets"].sets[set]
        # print(_set.subdomain_data.inlet,_set.subdomain_data.outlet)
        print(
            _set.global_ID,
            _set.phase,
            _set.subdomain_data.inlet,
            _set.subdomain_data.outlet,
        )

    if save_data:

        pmmoto.io.save_grid_data("dataOut/test_sets", sd, pm.grid)

        kwargs = {
            "inlet": "subdomain_data.inlet",
            "outlet": "subdomain_data.outlet",
            "proc": "proc_ID",
        }
        pmmoto.io.save_set_data(
            "dataOut/test_connect_set_single_phase",
            sd,
            connected_sets["sets"],
            **kwargs,
        )


if __name__ == "__main__":
    test_connected_sets()
    MPI.Finalize()
