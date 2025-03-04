"""test_inkbottle.py"""

from mpi4py import MPI
import numpy as np
import pmmoto

capillary_pressure = [
    7.69409,
    7.69409,
    7.69393,
    7.60403,
    7.36005,
    6.9909,
    6.53538,
    6.03352,
    5.52008,
    5.02123,
    4.55421,
    4.12854,
    3.74806,
    3.41274,
    3.12024,
    2.86704,
    2.64914,
    2.4625,
    2.30332,
    2.16814,
    2.05388,
    1.95783,
    1.87764,
    1.81122,
    1.75678,
    1.7127,
    1.67755,
    1.65002,
    1.62893,
    1.61322,
    1.60194,
    1.5943,
    1.58965,
    1.58964,
    1.5872,
    0.0,
]

w_saturation_analytical = [
    0.0,
    0.808933,
    0.808965,
    0.809662,
    0.810321,
    0.810973,
    0.811641,
    0.81235,
    0.813122,
    0.813978,
    0.814941,
    0.816034,
    0.817281,
    0.818703,
    0.820323,
    0.822162,
    0.824238,
    0.826565,
    0.829154,
    0.832009,
    0.835131,
    0.838511,
    0.842134,
    0.845978,
    0.850016,
    0.854214,
    0.858539,
    0.862957,
    0.867443,
    0.871985,
    0.876591,
    0.881302,
    0.886197,
    1.0,
    1.0,
    1.0,
]


def my_function():

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    # voxels = (140, 30, 30)
    voxels = (560, 120, 120)
    # voxels = (1120, 240, 240)
    voxels = (1680, 360, 360)  ##res = 120
    reservoir_voxels = 120
    box = ((0.0, 14.0), (-1.5, 1.5), (-1.5, 1.5))

    inlet = ((1, 0), (0, 0), (0, 0))

    subdomains = (8, 1, 1)

    sd = pmmoto.initialize(
        voxels,
        box=box,
        subdomains=subdomains,
        inlet=inlet,
        reservoir_voxels=reservoir_voxels,
        rank=rank,
    )

    print(sd.rank, sd.voxels)

    pm = pmmoto.domain_generation.gen_pm_inkbottle(sd)

    mp = pmmoto.core.multiphase.Multiphase(pm, np.copy(pm.img), 2)
    # capillary_pressure = 1.58965

    nn = 1  # len(capillary_pressure)
    for n in range(0, nn):
        morph, w_saturation = pmmoto.filters.equilibrium_distribution.drainage(
            capillary_pressure[n], mp
        )

        if rank == 0:
            print(f"pmmoto: {w_saturation} \nanalytical: {w_saturation_analytical[n]}")

    edt = pmmoto.filters.distance.edt(pm.img, sd)
    pmmoto.io.save_img_data_parallel(
        "data_out/test_inkbottle",
        sd,
        pm.img,
        additional_img={"morph": morph, "edt": edt},
    )


if __name__ == "__main__":
    my_function()
