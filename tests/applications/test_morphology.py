import numpy as np
from mpi4py import MPI
import pmmoto


def my_function():

    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()

    subdomain_map = [1, 1, 1]  # Specifies how Domain is broken among procs
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

    sphere_data, domain_data = pmmoto.io.read_sphere_pack_xyzr_domain(file)
    pm = pmmoto.domain_generation.gen_pm_spheres_domain(
        sd,
        sphere_data,
    )

    edt = pmmoto.filters.calc_edt(domain, sd, pm.grid)

    radius = sd.resolution[0] * 4 + sd.resolution[0] * 1.0e-6
    edt_closing = pmmoto.filters.closing(sd, pm.grid, radius=radius, fft=False)
    fft_closing = pmmoto.filters.closing(sd, pm.grid, radius=radius, fft=True)
    print(f"Closing methods equal: {np.array_equal(edt_closing, fft_closing)}")

    edt_opening = pmmoto.filters.opening(sd, pm.grid, radius=radius, fft=False)
    fft_opening = pmmoto.filters.opening(sd, pm.grid, radius=radius, fft=True)
    print(f"Opening methods equal: {np.array_equal(edt_opening, fft_opening)}")

    if save_data:
        ### Save Grid Data where kwargs are used for saving other grid data (i.e. EDT, Medial Axis)
        pmmoto.io.save_grid_data(
            "dataOut/test_morphology",
            sd,
            pm.grid,
            edt=edt,
            edt_closing=edt_closing,
            fft_closing=fft_closing,
            edt_opening=edt_opening,
            fft_opening=fft_opening,
        )


if __name__ == "__main__":
    my_function()
    MPI.Finalize()
