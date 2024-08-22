import numpy as np

# from mpi4py import MPI
from scipy.ndimage import distance_transform_edt
from edt import edt3d
import time
import pmmoto


def my_function():

    # comm = MPI.COMM_WORLD
    # size = comm.Get_size()
    # rank = comm.Get_rank()

    subdomain_map = [1, 1, 1]  # Specifies how domain is broken among processes
    voxels = [300, 300, 300]  # Total Number of Nodes in Domain

    box = [[0, 3.945410e-01], [0, 3.945410e-01], [0, 3.945410e-01]]

    ## Ordering for Inlet/Outlet ( (-x,+x) , (-y,+y) , (-z,+z) )
    boundaries = [[2, 2], [2, 2], [2, 2]]  # 0: Nothing Assumed  1: Walls 2: Periodic
    inlet = [[0, 0], [0, 0], [0, 0]]
    outlet = [[0, 0], [0, 0], [0, 0]]

    file = "tests/testDomains/50pack.out"

    testSerial = True
    testAlgo = True

    startTime = time.time()
    sd, domain = pmmoto.initialize(
        box=box,
        subdomain_map=subdomain_map,
        voxels=voxels,
        boundaries=boundaries,
        inlet=inlet,
        outlet=outlet,
        rank=0,
        mpi_size=1,
        reservoir_voxels=2,
    )
    sphere_data, domain_data = pmmoto.io.read_sphere_pack_xyzr_domain(file)
    pm = pmmoto.domain_generation.gen_pm_spheres_domain(
        sd,
        sphere_data,
    )

    edt = pmmoto.filters.calc_edt(domain, sd, pm.grid)

    ### Save Grid Data where kwargs are used for saving other grid data (i.e. EDT, Medial Axis)
    pmmoto.io.save_grid_data("dataOut/grid", sd, pm.grid, dist=edt)

    # refactor_procs = [2, 2, 2]
    # sd_all, local_grid = pmmoto.utils.deconstruct_grid(sd, pm.grid, refactor_procs)
    # file_out = "dataOut/decomposed_grid"
    # pmmoto.io.save_grid_data_proc(file_out, sd_all, local_grid)

    # if testSerial:

    #     edt_all = pmmoto.utils.reconstruct_grid_to_root(sd, edt)
    #     pm_grid_all = pmmoto.utils.reconstruct_grid_to_root(sd, pm.grid)

    #     if rank == 0:
    #         if testAlgo:

    #             startTime = time.time()

    #             sphere_data, _ = pmmoto.io.read_sphere_pack_xyzr_domain(file)

    #             ##### To GENERATE SINGLE PROC TEST CASE ######
    #             x = np.linspace(
    #                 sd.domain.size_domain[0, 0] + sd.domain.voxel[0] / 2,
    #                 sd.domain.size_domain[0, 1] - sd.domain.voxel[0] / 2,
    #                 nodes[0],
    #             )
    #             y = np.linspace(
    #                 sd.domain.size_domain[1, 0] + sd.domain.voxel[1] / 2,
    #                 sd.domain.size_domain[1, 1] - sd.domain.voxel[1] / 2,
    #                 nodes[1],
    #             )
    #             z = np.linspace(
    #                 sd.domain.size_domain[2, 0] + sd.domain.voxel[2] / 2,
    #                 sd.domain.size_domain[2, 1] - sd.domain.voxel[2] / 2,
    #                 nodes[2],
    #             )

    #             grid_out = (
    #                 pmmoto.domain_generation._domainGeneration.gen_domain_sphere_pack(
    #                     x, y, z, sphere_data
    #                 )
    #             )
    #             grid_out = np.ascontiguousarray(grid_out)

    #             pG = [0, 0, 0]
    #             pgSize = int(nodes[0] / 2)

    #             if boundaries[0][0] == 1:
    #                 pG[0] = 1
    #             if boundaries[1][0] == 1:
    #                 pG[1] = 1
    #             if boundaries[2][0] == 1:
    #                 pG[2] = 1

    #             periodic = [False, False, False]
    #             if boundaries[0][0] == 2:
    #                 periodic[0] = True
    #                 pG[0] = pgSize
    #             if boundaries[1][0] == 2:
    #                 periodic[1] = True
    #                 pG[1] = pgSize
    #             if boundaries[2][0] == 2:
    #                 periodic[2] = True
    #                 pG[2] = pgSize

    #             grid_out = np.pad(
    #                 grid_out, ((pG[0], pG[0]), (pG[1], pG[1]), (pG[2], pG[2])), "wrap"
    #             )

    #             if boundaries[0][0] == 1:
    #                 grid_out[0, :, :] = 0
    #             if boundaries[0][1] == 1:
    #                 grid_out[-1, :, :] = 0
    #             if boundaries[1][0] == 1:
    #                 grid_out[:, 0, :] = 0
    #             if boundaries[1][1] == 1:
    #                 grid_out[:, -1, :] = 0
    #             if boundaries[2][0] == 1:
    #                 grid_out[:, :, 0] = 0
    #             if boundaries[2][1] == 1:
    #                 grid_out[:, :, -1] = 0

    #             realDT = edt3d(grid_out, anisotropy=sd.domain.voxel)
    #             edtV, _ = distance_transform_edt(
    #                 grid_out, sampling=sd.domain.voxel, return_indices=True
    #             )
    #             endTime = time.time()

    #             print("Serial Time:", endTime - startTime)

    #             arrayList = [grid_out, realDT, edtV]

    #             for i, arr in enumerate(arrayList):
    #                 if pG[0] > 0 and pG[1] == 0 and pG[2] == 0:
    #                     arrayList[i] = arr[pG[0] : -pG[0], :, :]
    #                 elif pG[0] == 0 and pG[1] > 0 and pG[2] == 0:
    #                     arrayList[i] = arr[:, pG[1] : -pG[1], :]
    #                 elif pG[0] == 0 and pG[1] == 0 and pG[2] > 0:
    #                     arrayList[i] = arr[:, :, pG[2] : -pG[2]]
    #                 elif pG[0] > 0 and pG[1] == 0 and pG[2] > 0:
    #                     arrayList[i] = arr[pG[0] : -pG[0], :, pG[2] : -pG[2]]
    #                 elif pG[0] > 0 and pG[1] > 0 and pG[2] == 0:
    #                     arrayList[i] = arr[pG[0] : -pG[0], pG[1] : -pG[1], :]
    #                 elif pG[0] == 0 and pG[1] > 0 and pG[2] > 0:
    #                     arrayList[i] = arr[:, pG[1] : -pG[1], pG[2] : -pG[2]]
    #                 elif pG[0] > 0 and pG[1] > 0 and pG[2] > 0:
    #                     arrayList[i] = arr[
    #                         pG[0] : -pG[0], pG[1] : -pG[1], pG[2] : -pG[2]
    #                     ]
    #             ####################################################################

    #             grid_out = arrayList[0]
    #             realDT = arrayList[1]
    #             edtV = arrayList[2]

    #             print("L2 Grid Error Norm", np.linalg.norm(grid_out - pm_grid_all))

    #             diffEDT = np.abs(realDT - edt_all)
    #             diffEDT2 = np.abs(edtV - edt_all)

    #             print("L2 EDT Error Norm", np.linalg.norm(diffEDT))
    #             print("L2 EDT Error Norm 2", np.linalg.norm(diffEDT2))
    #             print("L2 EDT Error Norm 2", np.linalg.norm(realDT - edtV))

    #             print("LI EDT Error Norm", np.max(diffEDT))
    #             print("LI EDT Error Norm 2", np.max(diffEDT2))
    #             print("LI EDT Error Norm 2", np.max(realDT - edtV))


if __name__ == "__main__":
    my_function()
    MPI.Finalize()
