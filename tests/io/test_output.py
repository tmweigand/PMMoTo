"""test_output.py"""

import pmmoto
import pytest
import numpy as np
from mpi4py import MPI


def test_save_particle_data(tmp_path):
    """Tests for saving particle data"""
    sd = pmmoto.initialize((10, 10, 10))
    particles = np.array([[0, 0, 0], [1, 1, 1]])
    file_path = tmp_path / "particles"

    pmmoto.io.output.save_particle_data(str(file_path), sd, particles)
    expected_file = file_path / "particlesProc.0.vtu"

    assert expected_file.exists()

    # Check for ptvu file
    expected_file = file_path / "particles.pvtu"
    assert expected_file.exists()

    sd.rank = 1
    particles = np.array([[0, 0, 0, 1], [1, 1, 1, 1]])
    file_path = tmp_path / "particles"

    pmmoto.io.output.save_particle_data(str(file_path), sd, particles)
    expected_file = file_path / "particlesProc.1.vtu"

    assert expected_file.exists()

    sd.rank = 2
    particles = np.array([[0, 0, 0, 1, 2], [1, 1, 1, 1, 2]])
    file_path = tmp_path / "particles"

    pmmoto.io.output.save_particle_data(str(file_path), sd, particles)
    expected_file = file_path / "particlesProc.2.vtu"

    assert expected_file.exists()

    sd.rank = 3
    particles = np.array([[0, 0, 0, 1, 2, 3], [1, 1, 1, 1, 2, 3]])
    file_path = tmp_path / "particles"

    pmmoto.io.output.save_particle_data(str(file_path), sd, particles)
    expected_file = file_path / "particlesProc.3.vtu"

    assert expected_file.exists()


def test_save_img(tmp_path):
    """Tests saving of img"""
    sd = pmmoto.initialize((10, 10, 10))
    img = np.zeros(sd.voxels)
    file_path = tmp_path / "test_img"

    pmmoto.io.output.save_img(str(file_path), sd, img)

    expected_file = tmp_path / "test_img.vti"
    assert expected_file.exists()

    # Img is not shape of sd.voxels
    err_img = np.zeros((5, 5, 5))
    with pytest.raises(ValueError):
        pmmoto.io.output.save_img(
            str(file_path), sd, img, additional_img={"err_img": err_img}
        )


def test_save_img_additional(tmp_path):
    """Tests saving of img"""
    sd = pmmoto.initialize((10, 10, 10))
    img = np.zeros(sd.voxels)
    img2 = np.zeros(sd.voxels)
    file_path = tmp_path / "test_img"

    pmmoto.io.output.save_img(str(file_path), sd, img, additional_img={"img2": img2})

    expected_file = tmp_path / "test_img.vti"
    assert expected_file.exists()

    pmmoto.io.output.save_img(str(file_path), sd, img, additional_img={"img2": img2})

    expected_file = tmp_path / "test_img.vti"
    assert expected_file.exists()

    with pytest.raises(TypeError):
        pmmoto.io.output.save_img(str(file_path), sd, img, additional_img={"img2": 4.0})

    img3 = np.zeros((2, 2, 2))
    with pytest.raises(ValueError):
        pmmoto.io.output.save_img(
            str(file_path), sd, img, additional_img={"img3": img3}
        )


def test_save_imgs(tmp_path):
    """Tests saving of img"""
    sd_0 = pmmoto.initialize((10, 10, 10), subdomains=(2, 1, 1), rank=0)
    sd_1 = pmmoto.initialize((10, 10, 10), subdomains=(2, 1, 1), rank=1)
    img_0 = np.zeros(sd_0.voxels)
    img_1 = np.zeros(sd_1.voxels)

    sd = {0: sd_0, 1: sd_1}
    img = {0: img_0, 1: img_1}

    file_path = tmp_path / "test_imgs"

    pmmoto.io.output.save_img(str(file_path), sd, img)

    expected_file = tmp_path / "test_imgs_proc/test_imgs_proc_0.vti"
    assert expected_file.exists()

    expected_file = tmp_path / "test_imgs_proc/test_imgs_proc_1.vti"
    assert expected_file.exists()

    expected_file = tmp_path / "test_imgs.pvti"
    assert expected_file.exists()


def test_save_extended_img(tmp_path):
    """Tests saving of extedned img"""
    sd = pmmoto.initialize((10, 10, 10))
    img = np.zeros(sd.voxels)
    extended_img = np.zeros((15, 15, 15))

    file_path = tmp_path / "test_imgs"

    with pytest.raises(ValueError):
        pmmoto.io.output.save_extended_img_data_parallel(str(file_path), sd, img)

    pmmoto.io.output.save_extended_img_data_parallel(
        str(file_path), sd, extended_img, extension=((1, 4), (1, 4), (1, 4))
    )

    expected_file = tmp_path / "test_imgs/test_imgs_proc_0.vti"
    assert expected_file.exists()

    expected_file = tmp_path / "test_imgs.pvti"
    assert expected_file.exists()


@pytest.mark.mpi(min_size=2)
def test_save_img_parallel_2(tmp_path):
    """Tests saving of img"""
    comm = MPI.COMM_WORLD

    subdomains = (2, 1, 1)
    sd = pmmoto.initialize((10, 10, 10), subdomains=subdomains, rank=comm.Get_rank())
    img = np.zeros(sd.voxels)
    img2 = np.zeros(sd.voxels)
    file_path = tmp_path / "test_img"

    pmmoto.io.output.save_img(str(file_path), sd, img, additional_img={"img2": img2})

    if sd.rank == 0:
        expected_file = tmp_path / "test_img_proc/test_img_proc_0.vti"
        assert expected_file.exists()
    if sd.rank == 1:
        expected_file = tmp_path / "test_img_proc/test_img_proc_1.vti"
        assert expected_file.exists()

    # Img is not shape of sd.voxels
    err_img = np.zeros((5, 5, 5))
    with pytest.raises(ValueError):
        pmmoto.io.output.save_img(str(file_path), sd, err_img)
