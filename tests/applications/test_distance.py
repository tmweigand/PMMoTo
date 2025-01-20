import numpy as np
from mpi4py import MPI
import pytest
import pmmoto


@pytest.mark.mpi(min_size=8)
def test_edt_bcc():

    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()

    bcc_file = "tests/test_domains/bcc.in"
    bcc_spheres, domain_box = pmmoto.io.data_read.read_sphere_pack_xyzr_domain(bcc_file)

    boundary_types = ((2, 2), (2, 2), (2, 2))
    voxels = (300, 300, 300)

    subdomains = (2, 2, 2)
    sd = pmmoto.initialize(
        box=domain_box,
        subdomains=subdomains,
        voxels=voxels,
        boundary_types=boundary_types,
        rank=rank,
        mpi_size=size,
    )

    porous_media = pmmoto.domain_generation.gen_pm_spheres_domain(
        subdomain=sd, spheres=bcc_spheres
    )

    _edt = pmmoto.filters.distance.edt(porous_media.img, sd)

    np.testing.assert_approx_equal(np.max(_edt), 0.059628483)


@pytest.mark.mpi(min_size=8)
def test_edt_single_sphere():

    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()

    bcc_file = "tests/test_domains/single_sphere.in"
    bcc_spheres, domain_box = pmmoto.io.data_read.read_sphere_pack_xyzr_domain(bcc_file)

    boundary_types = ((1, 1), (1, 1), (1, 1))
    voxels = (300, 300, 300)

    subdomains = (2, 2, 2)
    sd = pmmoto.initialize(
        box=domain_box,
        subdomains=subdomains,
        voxels=voxels,
        boundary_types=boundary_types,
        rank=rank,
        mpi_size=size,
    )

    porous_media = pmmoto.domain_generation.gen_pm_spheres_domain(
        subdomain=sd, spheres=bcc_spheres
    )

    _edt = pmmoto.filters.distance.edt(porous_media.img, sd)

    pmmoto.io.output.save_grid_data_parallel(
        "data_out/test_edt_single_sphere", sd, porous_media.img, **{"edt": _edt}
    )

    np.testing.assert_approx_equal(np.max(_edt), 0.3695041)
