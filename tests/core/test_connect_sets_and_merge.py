import numpy as np
import pmmoto


# def test_connect_sets_and_merge():

#     comm = MPI.COMM_WORLD
#     size = comm.Get_size()
#     rank = comm.Get_rank()

#     subdomain_map = [1, 1, 1]
#     voxels = [100, 100, 100]

#     box = [[0, 3.945410e-01], [0, 3.945410e-01], [0, 3.945410e-01]]
#     file = "tests/testDomains/50pack.out"
#     boundaries = [[2, 2], [2, 2], [2, 2]]
#     inlet = [[0, 0], [0, 0], [0, 0]]
#     outlet = [[0, 0], [0, 0], [0, 0]]

#     # Multiphase
#     num_fluid_phases = 2

#     w_inlet = [[0, 0], [0, 0], [0, 0]]
#     nw_inlet = [[0, 0], [0, 0], [0, 0]]
#     mp_inlet = {0: w_inlet, 1: nw_inlet}

#     w_outlet = [[0, 0], [0, 0], [0, 0]]
#     nw_outlet = [[0, 0], [0, 0], [0, 0]]
#     mp_outlet = {0: w_outlet, 1: nw_outlet}

#     save_data = True

#     sd, domain = pmmoto.initialize(
#         box=box,
#         subdomain_map=subdomain_map,
#         voxels=voxels,
#         boundaries=boundaries,
#         inlet=inlet,
#         outlet=outlet,
#         rank=rank,
#         mpi_size=size,
#         reservoir_voxels=0,
#     )

#     sphere_data, domain_data = pmmoto.io.read_sphere_pack_xyzr_domain(file)
#     pm = pmmoto.domain_generation.gen_pm_spheres_domain(
#         sd,
#         sphere_data,
#     )

#     mp = pmmoto.core.initialize_multiphase(
#         porous_media=pm,
#         num_phases=num_fluid_phases,
#         inlets=mp_inlet,
#         outlets=mp_outlet,
#     )

#     mp = pmmoto.domain_generation.gen_mp_constant(mp, fluid_phase=2)

#     # all_sets, local_global_map = pmmoto.core.sets.create_sets_and_merge(
#     #     img, set_count, label_count, label_grid, phase_map, inlet, outlet
#     # )

#     # print(all_sets, local_global_map)

#     if save_data:

#         kwargs = {}
#         pmmoto.io.save_grid_data(
#             "dataOut/test_connect_sets_and_merge", sd, mp.grid, **kwargs
#         )


# if __name__ == "__main__":
#     test_connect_sets_and_merge()
#     MPI.Finalize()


def test_connect_sets():
    """
    Test  subdomain features
    """

    # subdomain_map = [1, 1, 1]
    subdomain_map = [1, 1, 1]

    voxels = (10, 10, 10)

    box = [[0, 10], [0, 10], [0, 10]]
    boundaries = [[0, 0], [0, 0], [0, 0]]
    # boundaries = [[2, 2], [2, 2], [2, 2]]
    inlet = [[0, 0], [0, 0], [0, 0]]
    outlet = [[0, 0], [0, 0], [0, 0]]

    save_data = True

    sd = {}
    domain = {}
    grid = {}

    size = np.prod(subdomain_map)

    for rank in range(size):
        sd[rank], domain[rank] = pmmoto.initialize(
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

        grid = np.zeros(sd[rank].voxels, dtype=np.uint64)
        grid[0, :, :] = 1
        pmmoto.core.sets.create_sets_and_merge(
            img=grid, subdomain=sd[rank], label_count=2
        )
