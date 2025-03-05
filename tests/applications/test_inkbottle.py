"""test_inkbottle.py"""

from mpi4py import MPI
import numpy as np
import pmmoto

# import matplotlib.pyplot as plt

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
    # 0.0,
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
    # 1.0,
]

# capillary_pressure.sort(reverse=False)
# w_saturation_analytical.sort(reverse=False)


def initialize_ink_bottle(parallel=False):

    if parallel:
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        subdomains = (8, 1, 1)
    else:
        rank = 0
        subdomains = (1, 1, 1)

    # voxels = (140, 30, 30)  # res = 10
    # reservoir_voxels = 30

    voxels = (560, 120, 120)  ##res = 40
    reservoir_voxels = 40

    # voxels = (1120, 240, 240)  ##res = 80
    # reservoir_voxels = 80

    # voxels = (1680, 360, 360)  ##res = 120
    # reservoir_voxels = 120

    box = ((0.0, 14.0), (-1.5, 1.5), (-1.5, 1.5))

    inlet = ((0, 1), (0, 0), (0, 0))
    outlet = ((1, 0), (0, 0), (0, 0))

    sd = pmmoto.initialize(
        voxels,
        box=box,
        subdomains=subdomains,
        inlet=inlet,
        outlet=outlet,
        reservoir_voxels=reservoir_voxels,
        rank=rank,
    )

    pm = pmmoto.domain_generation.gen_pm_inkbottle(sd)
    mp = pmmoto.domain_generation.gen_mp_constant(pm, 2)

    return sd, pm, mp


def test_drainage():

    sd, pm, mp = initialize_ink_bottle()

    nn = len(capillary_pressure)
    w_saturation = np.zeros(nn)
    n = 25

    mp_img, w_saturation[n] = pmmoto.filters.equilibrium_distribution.calcDrainage(
        capillary_pressure[n], mp
    )

    #     # # morph, w_saturation[n] = pmmoto.filters.equilibrium_distribution.drainage(
    #     # #     capillary_pressure[n], mp
    #     # # )

    # if rank == 0:
    # print(
    #     f"pmmoto: {w_saturation[n]} \nanalytical: {w_saturation_analytical[n]} \nradius: {pmmoto.core.multiphase.Multiphase.get_probe_radius(capillary_pressure[n])}\n p_c: {capillary_pressure[n]}\n"
    # )

    # radius = pmmoto.core.multiphase.Multiphase.get_probe_radius(1.5872)
    # edt = pmmoto.filters.distance.edt(pm.img, sd)
    # ind = np.where(
    #     (edt >= radius) & (mp.pm_img == 1),
    #     1,
    #     0,
    # ).astype(np.uint8)

    # ind = mp.subdomain.update_reservoir(ind, 1)
    # ind = mp.subdomain.set_wall_bcs(ind)
    # ind2 = pmmoto.filters.connected_components.inlet_connected_img(mp.subdomain, ind)
    # ind2 = mp.subdomain.update_reservoir(ind2, 0)
    # morph = pmmoto.filters.morphological_operators.addition(
    #     mp.subdomain, ind2, radius, fft=False
    # )

    # _grid_distance = pmmoto.filters.distance.edt3d(
    #     np.logical_not(ind2),
    #     resolution=sd.domain.resolution,
    #     squared=True,
    # )

    # w_connected = pmmoto.filters.connected_components.outlet_connected_img(
    #     mp.subdomain, mp.img, 2
    # )

    # n_connected = pmmoto.filters.connected_components.inlet_connected_img(
    #     mp.subdomain, mp.img, 1
    # )

    # pmmoto.io.save_img_data_parallel(
    #     "data_out/test_inkbottle",
    #     sd,
    #     pm.img,
    #     additional_img={
    #         "mp_img": mp.img,
    #         #         "edt": edt,
    #         "w_connected": w_connected,
    #         "n_connected": n_connected,
    #         #         "ind": ind,
    #         #         "ind2": ind2,
    #         #         "mp": mp.img,
    #         #         "_grid_distance": _grid_distance,
    #     },
    # )

    # if sd.rank == 0:
    #     plt.plot(w_saturation_analytical, capillary_pressure, label="Analytic")
    #     plt.plot(w_saturation, capillary_pressure, label="pmmoto")
    #     plt.legend()
    #     plt.show()


if __name__ == "__main__":
    # new_function()
    test_drainage()
