"""Example: Generate the pore size distribution of a sphere packing."""

from mpi4py import MPI
import pmmoto


def sphere_pack_psd() -> None:
    """Generate the pore size distribution of a sphere pack.

    To run this file:
        mpirun -np 8 python examples/sphere_pack_psd/sphere_pack_psd.py
    """
    sphere_pack_file = "examples/sphere_pack_psd/sphere_pack.in"
    spheres, domain_box = pmmoto.io.data_read.read_sphere_pack_xyzr_domain(
        sphere_pack_file
    )

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    voxels = (401, 401, 401)
    boundary = pmmoto.BoundaryType.PERIODIC
    boundary_types = (
        (boundary, boundary),
        (boundary, boundary),
        (boundary, boundary),
    )
    subdomains = (2, 2, 2)

    sd = pmmoto.initialize(
        voxels=voxels,
        rank=rank,
        subdomains=subdomains,
        box=domain_box,
        boundary_types=boundary_types,
    )

    pm = pmmoto.domain_generation.gen_pm_spheres_domain(sd, spheres, invert=False)

    # Perform edt
    dist = pmmoto.filters.distance.edt(pm.img, sd)
    dist = pm.distance

    # Collect pore size distribution image
    # aka largest radius of a sphere that fits in a given voxel
    psd = pmmoto.filters.porosimetry.pore_size_distribution(
        sd, pm, num_radii=25, inlet=False
    )

    # Generate a pdf of the pore sizes
    pmmoto.filters.porosimetry.plot_pore_size_distribution(
        "examples/sphere_pack_psd/pm", sd, psd, plot_type="pdf"
    )

    # Invert the porous structure
    invert_pm = pmmoto.domain_generation.gen_pm_spheres_domain(sd, spheres, invert=True)

    invert_distance = invert_pm.distance

    # Collect pore size distribution image on the inverted image
    invert_psd = pmmoto.filters.porosimetry.pore_size_distribution(
        sd, invert_pm, num_radii=25, inlet=False
    )

    # Generate a pdf of the inverted pore sizes
    pmmoto.filters.porosimetry.plot_pore_size_distribution(
        "examples/sphere_pack_psd/inverted_pm",
        sd,
        invert_psd,
    )

    pmmoto.io.output.save_img(
        file_name="examples/sphere_pack_psd/image",
        subdomain=sd,
        img=pm.img,
        additional_img={
            "psd": psd,
            "dist": dist,
            "invert_pm": invert_pm.img,
            "invert_dist": invert_distance,
            "invert_psd": invert_psd,
        },
    )


if __name__ == "__main__":
    sphere_pack_psd()
