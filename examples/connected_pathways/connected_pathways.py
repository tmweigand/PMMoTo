"""Example: Generate the pore size distribution of a sphere packing."""

from mpi4py import MPI
import pmmoto


def connected_pathways() -> None:
    """Examine the connections of a thin porous structure.

    To run this file:
        mpirun -np 4 python examples/connected_pathways/connected_pathways.py
    """
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    voxels = (2000, 2000, 10)
    box = ((0, 2000), (0, 2000), (0, 30))
    inlet = ((True, False), (False, False), (False, False))
    outlet = ((False, True), (False, False), (False, False))

    subdomains = (2, 2, 1)

    sd = pmmoto.initialize(
        voxels, rank=rank, subdomains=subdomains, box=box, inlet=inlet, outlet=outlet
    )

    img = pmmoto.domain_generation.gen_img_smoothed_random_binary(
        sd.domain.voxels, p_zero=0.5, smoothness=10, seed=8
    )

    sd, img_sd = pmmoto.domain_generation.deconstruct_img(sd, img, subdomains, rank)

    cc, label_count = pmmoto.filters.connected_components.connect_components(img_sd, sd)

    inlet_img = pmmoto.filters.connected_components.inlet_connected_img(sd, img_sd)
    outlet_img = pmmoto.filters.connected_components.outlet_connected_img(sd, img_sd)
    inlet_outlet_img = pmmoto.filters.connected_components.inlet_outlet_connected_img(
        sd, img_sd
    )
    isolated_img = pmmoto.filters.connected_components.isolated_img(sd, img_sd)

    pmmoto.io.output.save_img(
        "examples/connected_pathways/image",
        sd,
        img_sd,
        additional_img={
            "cc": cc,
            "inlet_img": inlet_img,
            "outlet_img": outlet_img,
            "inlet_outlet_img": inlet_outlet_img,
            "isolated_img": isolated_img,
        },
    )


if __name__ == "__main__":
    connected_pathways()
