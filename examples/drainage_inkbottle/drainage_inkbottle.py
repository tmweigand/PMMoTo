"""Example: Morphological drainage simulation in an ink bottle geometry using PMMoTo.

Run with:
    mpirun -np 2 python examples/drainage_inkbottle/drainage_inkbottle.py
"""

from mpi4py import MPI
import numpy as np
import matplotlib.pyplot as plt
import pmmoto

comm = MPI.COMM_WORLD
rank = comm.Get_rank()


def drain_ink_bottle() -> None:
    """Simulate morphological drainage of an ink bottle with two methods."""
    # Domain voxel resolution.
    # Must be a 3-tuple as only 3D is currently supported.
    voxels = (560, 120, 120)

    # Number of voxels allocated for the inlet reservoir region.
    reservoir_voxels = 20

    # Domain decomposition across MPI ranks.
    # The product of subdomain counts must match the number of MPI processes.
    subdomains = (2, 1, 1)

    # Physical extent of the domain in each dimension: (min, max)
    box = (
        (0.0, 14.0),  # x-dimension
        (-1.5, 1.5),  # y-dimension
        (-1.5, 1.5),  # z-dimension
    )

    # Boundary conditions for each face  (−, +) per axis.
    # Options:
    #   - END:     Nothing assumed
    #   - WALL:    Solid, impermeable boundary added to image
    #   - PERIODIC: Wraparound — both faces of the axis must be periodic
    boundary_types = (
        (pmmoto.BoundaryType.END, pmmoto.BoundaryType.END),  # x
        (pmmoto.BoundaryType.END, pmmoto.BoundaryType.END),  # y
        (pmmoto.BoundaryType.END, pmmoto.BoundaryType.END),  # z
    )

    # Inlet boundary condition: (−, +) per axis
    # Used to specify where fluid enters the domain
    inlet = (
        (False, True),  # x: fluid enters from the +x face
        (False, False),  # y: no inlet
        (False, False),  # z: no inlet
    )

    # Outlet boundary condition: (−, +) per axis
    # Used to specify where fluid exits the domain
    outlet = (
        (True, False),  # x: fluid exits from the −x face
        (False, False),  # y: no outlet
        (False, False),  # z: no outlet
    )

    # Initialize the simulation domain with number of voxels and MPI parameters,
    # specifying the decomposition (subdomains), boundary conditions and inlet/outlet,
    # reservoir size, and global domain size for this MPI rank.
    sd = pmmoto.initialize(
        voxels=voxels,
        box=box,
        boundary_types=boundary_types,
        rank=rank,
        subdomains=subdomains,
        inlet=inlet,
        outlet=outlet,
        reservoir_voxels=reservoir_voxels,
    )

    # Initialize the ink bottle as a porous media
    # Set to 1 for traditional ink bottle
    pm = pmmoto.domain_generation.gen_pm_inkbottle(sd, 1, 1)

    # Fill the pore space with fluid 2 (wetting phase) creating a multiphase system
    mp = pmmoto.domain_generation.gen_mp_constant(pm, 2)

    # Set the capillary pressures
    capillary_pressure = 0.1 + np.linspace(0, 1, 41) ** 1.5 * 7.6

    # Perform a classical morphological drainage with a surface tension (gamma) of 1
    # The output of this function is the predicted equilibrium saturation at a
    # given capillary pressure
    w_saturation_standard = pmmoto.filters.equilibrium_distribution.drainage(
        mp,
        capillary_pressure,
        gamma=1,
        method="standard",
    )

    # Save final state
    pmmoto.io.output.save_img(
        file_name="examples/drainage_inkbottle/image",
        subdomain=sd,
        img=pm.img,
        additional_img={"mp_img": mp.img},
    )

    # Reset to fully wet domain
    mp = pmmoto.domain_generation.gen_mp_constant(pm, 2)

    # Perform a morphological drainage with a contact angle of 20 degrees.
    # and a surface tension (gamma) of 1
    w_saturation_contact_angle = pmmoto.filters.equilibrium_distribution.drainage(
        mp, capillary_pressure, gamma=1, contact_angle=20, method="contact_angle"
    )

    # Generate plot to compare results between the two methos
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
        plt.legend()
        plt.savefig("examples/drainage_inkbottle/saturation_pressure_plot.png")
        plt.close()


if __name__ == "__main__":
    drain_ink_bottle()
