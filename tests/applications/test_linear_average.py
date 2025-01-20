import numpy as np
from mpi4py import MPI
import pmmoto


def my_function():

    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()

    subdomains = [1, 1, 1]  # Specifies how domain is broken among processes
    nodes = [300, 300, 300]  # Total Number of Nodes in Domain

    ## Ordering for Inlet/Outlet ( (-x,+x) , (-y,+y) , (-z,+z) )
    boundaries = [[0, 0], [0, 0], [0, 0]]  # 0: Nothing Assumed  1: Walls 2: Periodic
    inlet = [[0, 0], [0, 0], [0, 0]]
    outlet = [[0, 0], [0, 0], [0, 0]]

    file = "./testDomains/50pack.out"

    sd = pmmoto.initialize(rank, size, subdomains, nodes, boundaries, inlet, outlet)
    sphere_data, domain_data = pmmoto.io.read_sphere_pack_xyzr_domain(file)
    pm = pmmoto.domain_generation.gen_pm_spheres_domain(sd, sphere_data, domain_data)

    average_porosity = pmmoto.analysis.average.linear(sd, pm.grid, direction=2)

    print(average_porosity)
    # plt.plot(average_porosity)
    # plt.show()


if __name__ == "__main__":
    my_function()
    MPI.Finalize()
