"""Example: Generate the pore size distribution of a sphere packing."""

from mpi4py import MPI
import numpy as np
import pmmoto

comm = MPI.COMM_WORLD
rank = comm.Get_rank()


def connected_pathways() -> None:
    """Generate the pore size distribution of a sphere pack.

    To run this file:
        python examples/connected_pathways.py
    or to run in parallel:
        set subdomains = (2,2,2)
        mpirun -np 8 python examples/connected_pathways.py
    """
    voxels = (2000, 2000, 10)
    box = ((0, 2000), (0, 2000), (0, 30))
    inlet = ((True, False), (False, False), (False, False))
    outlet = ((False, True), (False, False), (False, False))

    subdomains = (2, 1, 1)

    sd = pmmoto.initialize(
        voxels, rank=rank, subdomains=subdomains, box=box, inlet=inlet, outlet=outlet
    )
    img = pmmoto.domain_generation.gen_img_smoothed_random_binary(
        sd.domain.voxels, p_zero=0.5, smoothness=10, seed=3
    )

    sd, img_sd = pmmoto.core.pmmoto.deconstruct_grid(sd, img, subdomains, rank)
    morph_img = pmmoto.filters.morphological_operators.subtraction(sd, img_sd, radius=5)

    cc, label_count = pmmoto.filters.connected_components.connect_components(img_sd, sd)

    connections = pmmoto.filters.connected_components.inlet_outlet_connections(sd, cc)
    inlet_img = pmmoto.filters.connected_components.inlet_connected_img(sd, cc)
    outlet_img = pmmoto.filters.connected_components.outlet_connected_img(sd, cc)

    isolated_internal = np.where(
        (inlet_img == 0) & (outlet_img == 0) & (img_sd == 1), 1, 0
    )

    # pmmoto.io.output.save_img_data_parallel(
    #     "examples/connected_pathways/images",
    #     sd,
    #     img_sd,
    #     additional_img={
    #         "cc": cc,
    #         "morph_img": morph_img,
    #         "inlet_img": inlet_img,
    #         "outlet_img": outlet_img,
    #         "isolated_internal": isolated_internal,
    #     },
    # )


if __name__ == "__main__":
    connected_pathways()
