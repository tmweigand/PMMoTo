"""drainage_inkbottle.py"""

from mpi4py import MPI
import numpy as np
import matplotlib.pyplot as plt
import pmmoto

comm = MPI.COMM_WORLD
rank = comm.Get_rank()

capillary_pressure = [
    0.0,
    0.75,
    1.25,
    1.265,
    1.3,
    1.35,
    1.4,
    1.45,
    1.5,
    1.55,
    1.58965,
    1.5943,
    1.60194,
    1.61322,
    1.62893,
    1.65002,
    1.67755,
    1.7127,
    1.75678,
    1.81122,
    1.87764,
    1.95783,
    2.05388,
    2.16814,
    2.30332,
    2.4625,
    2.64914,
    2.86704,
    3.12024,
    3.41274,
    3.74806,
    4.12854,
    4.55421,
    5.02123,
    5.52008,
    6.03352,
    6.53538,
    6.9909,
    7.36005,
    7.60403,
    7.69393,
    7.69409,
]


def drain_ink_bottle():
    """
    This is an example that simulates morphological drainage of an inkbottle.

    To run this file:
        mpirun -np 8 python examples/drainage_inkbottle.py

    """

    voxels = (560, 120, 120)
    reservoir_voxels = 40
    subdomains = (8, 1, 1)

    box = ((0.0, 14.0), (-1.5, 1.5), (-1.5, 1.5))

    inlet = ((0, 1), (0, 0), (0, 0))
    outlet = ((1, 0), (0, 0), (0, 0))

    sd = pmmoto.initialize(
        voxels,
        rank=rank,
        box=box,
        subdomains=subdomains,
        inlet=inlet,
        outlet=outlet,
        reservoir_voxels=reservoir_voxels,
    )

    pm = pmmoto.domain_generation.gen_pm_inkbottle(sd)
    mp = pmmoto.domain_generation.gen_mp_constant(pm, 2)

    w_saturation_standard = pmmoto.filters.equilibrium_distribution.drainage(
        mp, capillary_pressure, gamma=1, method="standard", save=True
    )

    # Reset to fully wetted domain
    mp = pmmoto.domain_generation.gen_mp_constant(pm, 2)

    w_saturation_contact_angle = pmmoto.filters.equilibrium_distribution.drainage(
        mp, capillary_pressure, gamma=1, contact_angle=20, method="contact_angle"
    )

    # Save final state of multiphase image
    pmmoto.io.output.save_img_data_parallel(
        file_name="examples/drainage_inkbottle_img",
        subdomain=sd,
        img=pm.img,
        additional_img={"mp_img": mp.img},
    )

    if rank == 0:
        plt.plot(
            w_saturation_standard, capillary_pressure, ".", label="Standard Method"
        )
        plt.plot(
            w_saturation_contact_angle,
            capillary_pressure,
            ".",
            label="Contact Angle Method",
        )
        plt.xlabel("Wetting Phase Saturation")
        plt.ylabel("Capillary Pressure")
        plt.savefig("examples/drainage_inkpottle.pdf")
        plt.close()


if __name__ == "__main__":
    drain_ink_bottle()
