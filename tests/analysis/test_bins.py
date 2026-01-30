"""test_bins.py"""

import numpy as np
import pmmoto
from mpi4py import MPI
import pytest


def test_bin_centers() -> None:
    """Dummy check for bins"""
    start = 0
    end = 1
    num_bins = 1

    bin = pmmoto.analysis.bins.Bin(start=start, end=end, num_bins=num_bins)

    assert (
        bin.centers[0] == 0.5
    )  # Only one bin, so center is halfway between start and end

    start = -1
    bin = pmmoto.analysis.bins.Bin(start=start, end=end, num_bins=num_bins)

    assert (
        bin.centers[0] == 0.0
    )  # Only one bin, so center is halfway between start and end

    num_bins = 2
    bin = pmmoto.analysis.bins.Bin(start=start, end=end, num_bins=num_bins)

    np.testing.assert_array_equal(bin.centers, np.array([-0.5, 0.5]))


def test_bin() -> None:
    """Test bins"""
    start = 0
    end = 3
    num_bins = 25

    bin = pmmoto.analysis.bins.Bin(start=start, end=end, num_bins=num_bins)

    assert bin.width == 0.12

    assert len(bin.centers) == num_bins

    np.testing.assert_array_equal(bin.values, np.zeros(num_bins))

    ones = np.ones(num_bins)
    bin.set_values(ones)
    np.testing.assert_array_equal(bin.values, ones)

    bin = pmmoto.analysis.bins.Bin(start=start, end=end, num_bins=num_bins, values=ones)

    np.testing.assert_array_equal(bin.values, ones)


def test_bin_volume():
    """Tests volume calculation"""
    bin = pmmoto.analysis.bins.Bin(start=0, end=1, num_bins=1)
    bin_volume = bin.calculate_volume(radial_volume=True)

    assert bin_volume == pmmoto.analysis.bins.radial_bin_volume(1)


def test_bin_rdf():
    """Tests rdf generation"""
    num_bins = 10
    bin = pmmoto.analysis.bins.Bin(start=0, end=1, num_bins=num_bins)
    rdf = bin.generate_rdf()

    np.testing.assert_array_equal(rdf, np.zeros(num_bins))

    bin = pmmoto.analysis.bins.Bin(
        start=0, end=1, num_bins=num_bins, values=np.ones(num_bins)
    )
    rdf = bin.generate_rdf()

    np.testing.assert_array_almost_equal(
        rdf,
        np.array(
            [
                100.0,
                14.28571429,
                5.26315789,
                2.7027027,
                1.63934426,
                1.0989011,
                0.78740157,
                0.59171598,
                0.46082949,
                0.36900369,
            ]
        ),
    )


def test_bin_save(tmp_path):
    """Tests bin save"""
    bin = pmmoto.analysis.bins.Bin(start=0, end=1, num_bins=5, name="test_bin")

    sd = pmmoto.initialize((5, 5, 5))
    bin.save_bin(sd, str(tmp_path) + "/")

    # Check file was created
    expected_file = tmp_path / "bin_test_bin.txt"
    assert expected_file.exists()

    # Check file content
    data = np.loadtxt(str(expected_file))
    assert data.shape == (5, 2)


def test_bin_save_not_rank_1(tmp_path):
    sd = pmmoto.initialize((5, 5, 5))
    sd.rank = 1
    bin = pmmoto.analysis.bins.Bin(start=0, end=1, num_bins=5, name="test_bin")
    bin.save_bin(sd, str(tmp_path) + "/")

    # Check file was not created
    expected_file = tmp_path / "bin_test_bin.txt"
    assert not expected_file.exists()


def test_bins() -> None:
    """Test bins"""
    start = [0, 1]
    end = [3, 2.8]
    num_bins = [25, 50]
    labels = [1, 5]

    bins = pmmoto.analysis.bins.Bins(
        starts=start, ends=end, num_bins=num_bins, labels=labels
    )

    assert bins.bins[1].width == 0.12
    assert bins.bins[5].width == 0.036

    values = {1: np.zeros(num_bins[0]), 5: np.ones(num_bins[1])}
    bins.update_bins(values)
    np.testing.assert_array_equal(bins.bins[1].values, np.zeros(num_bins[0]))
    np.testing.assert_array_equal(bins.bins[5].values, np.ones(num_bins[1]))


def test_bins_save(tmp_path):
    """Tests bin save"""
    start = [0, 1]
    end = [3, 2.8]
    num_bins = [25, 50]
    labels = [1, 5]

    bins = pmmoto.analysis.bins.Bins(
        starts=start,
        ends=end,
        num_bins=num_bins,
        labels=labels,
        names=["test_bin_1", "test_bin_2"],
    )
    sd = pmmoto.initialize((5, 5, 5))
    bins.save_bins(sd, str(tmp_path) + "/")

    # Check files was created
    expected_file_1 = tmp_path / "bins_test_bin_1.txt"
    assert expected_file_1.exists()

    expected_file_2 = tmp_path / "bins_test_bin_2.txt"
    assert expected_file_2.exists()

    # Check files content
    data = np.loadtxt(str(expected_file_1))
    assert data.shape == (25, 2)

    data = np.loadtxt(str(expected_file_2))
    assert data.shape == (50, 2)


def test_bins_save_not_rank_1(tmp_path):
    """Tests bin save"""
    start = [0, 1]
    end = [3, 2.8]
    num_bins = [25, 50]
    labels = [1, 5]

    bins = pmmoto.analysis.bins.Bins(
        starts=start,
        ends=end,
        num_bins=num_bins,
        labels=labels,
        names=["test_bin_1", "test_bin_2"],
    )
    sd = pmmoto.initialize((5, 5, 5))
    sd.rank = 1
    bins.save_bins(sd, str(tmp_path) + "/")

    # Check files was created
    expected_file_1 = tmp_path / "bins_test_bin_1.txt"
    assert not expected_file_1.exists()

    expected_file_2 = tmp_path / "bins_test_bin_2.txt"
    assert not expected_file_2.exists()


def test_count_locations():
    """Test binning coordinates"""
    start = 0
    end = 3
    num_bins = 25

    bin = pmmoto.analysis.bins.Bin(start, end, num_bins)

    coordinates = np.array([[0.1, 0.0, 0.0], [1.1, 0.0, 0.0], [2.1, 0.0, 0.0]])

    pmmoto.analysis.bins.count_locations(coordinates=coordinates, dimension=0, bin=bin)

    assert np.sum(bin.values) == 3

    start = -1
    end = 3
    num_bins = 25

    bin = pmmoto.analysis.bins.Bin(start, end, num_bins)

    pmmoto.analysis.bins.count_locations(coordinates=coordinates, dimension=0, bin=bin)

    assert np.sum(bin.values) == 3

    start = -1
    end = 2
    num_bins = 25

    bin = pmmoto.analysis.bins.Bin(start, end, num_bins)

    pmmoto.analysis.bins.count_locations(coordinates=coordinates, dimension=0, bin=bin)

    assert np.sum(bin.values) == 2


def test_count_membrane() -> None:
    """Test if membrane atoms are being counted."""
    membrane_file = "tests/test_data/LAMMPS/membranedata.gz"
    positions, types, domain = pmmoto.io.data_read.py_read_lammps_atoms(membrane_file)

    dimension = 2
    start = domain[dimension, 0]
    end = domain[dimension, 1]
    num_bins = 100
    bin = pmmoto.analysis.bins.Bin(start, end, num_bins)

    pmmoto.analysis.bins.count_locations(
        coordinates=positions, dimension=dimension, bin=bin
    )

    assert np.sum(bin.values) == positions.shape[0]


@pytest.mark.mpi(min_size=8)
def test_count_membrane_parallel() -> None:
    """Test if membrane atoms are being counted correctly in parallel"""
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    box = [
        [0.0, 176.0],
        [0.0, 176.0],
        [-287, 237],
    ]

    membrane_file = "tests/test_data/LAMMPS/membranedata.gz"
    positions, types, _ = pmmoto.io.data_read.py_read_lammps_atoms(membrane_file)

    sd = pmmoto.initialize(
        voxels=(10, 10, 10), box=box, rank=rank, subdomains=(2, 2, 2)
    )

    dimension = 0
    start = box[dimension][0]
    end = box[dimension][1]
    num_bins = 100
    bin = pmmoto.analysis.bins.Bin(start, end, num_bins)

    unique_types = np.unique(types)
    atom_radii = {}
    atom_masses = {}
    for _type in unique_types:
        atom_radii[_type] = 1e-12
        atom_masses[_type] = 0.1

    membrane = pmmoto.particles.initialize_atoms(
        sd,
        positions,
        atom_radii,
        types,
        atom_masses=atom_masses,
        by_type=False,
        trim_within=True,
    )

    coords = membrane.return_coordinates()

    pmmoto.analysis.bins.count_locations(
        coordinates=coords, dimension=dimension, bin=bin
    )

    assert np.sum(bin.values) == positions.shape[0]


def test_sum_masses() -> None:
    """Tests summing masses of atoms based on location."""
    start = 0
    end = 3
    num_bins = 25

    bin = pmmoto.analysis.bins.Bin(start, end, num_bins)

    coordinates = np.array([[0.1, 0.0, 0.0], [1.1, 0.0, 0.0], [2.1, 0.0, 0.0]])

    masses = np.array([3, 12.5, 256])

    _ = pmmoto.analysis.bins.sum_masses(
        coordinates=coordinates, dimension=0, bin=bin, masses=masses
    )

    assert np.sum(bin.values) == 3 + 12.5 + 256  # Total mass of all atoms

    bin = pmmoto.analysis.bins.Bin(start, end, num_bins)

    coordinates = np.array([[0.1, 0.0, 0.0], [0.1, 0.0, 0.0], [0.1, 0.0, 0.0]])

    pmmoto.analysis.bins.sum_masses(
        coordinates=coordinates, dimension=0, bin=bin, masses=masses
    )

    assert bin.values[0] == 3 + 12.5 + 256  # All atoms are in the first bin

    start = -5
    end = 3
    num_bins = 25

    bin = pmmoto.analysis.bins.Bin(start, end, num_bins)

    coordinates = np.array([[-4.9, 0.0, 0.0], [1.1, 0.0, 0.0], [2.1, 0.0, 0.0]])

    pmmoto.analysis.bins.sum_masses(
        coordinates=coordinates, dimension=0, bin=bin, masses=masses
    )


def test_sum_membrane_mass() -> None:
    """Test membrane mass counting."""
    membrane_file = "tests/test_data/LAMMPS/membranedata.gz"
    positions, types, masses, domain = pmmoto.io.data_read.py_read_lammps_atoms(
        membrane_file, include_mass=True
    )

    dimension = 2
    start = domain[dimension, 0]
    end = domain[dimension, 1]
    num_bins = 100
    bin = pmmoto.analysis.bins.Bin(start, end, num_bins)

    pmmoto.analysis.bins.sum_masses(
        coordinates=positions, dimension=2, bin=bin, masses=masses
    )

    # Calculate bin volume
    area = (domain[0, 1] - domain[0, 0]) * (domain[1, 1] - domain[1, 0])

    bin.calculate_volume(area=area)

    np.testing.assert_approx_equal(np.sum(bin.values), np.sum(masses))


@pytest.mark.mpi(min_size=8)
def test_sum_membrane_mass_parallel() -> None:
    """Test membrane mass counting."""
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    # Full domain with reservoirs
    box = [
        [0.0, 176.0],
        [0.0, 176.0],
        [-287, 237],
    ]

    atom_id_mass_map = {
        1: 12.01,
        3: 12.01,
        4: 16,
        5: 1.008,
        7: 14.01,
        8: 1.008,
        12: 16,
        14: 1.008,
    }

    membrane_file = "tests/test_data/LAMMPS/membranedata.gz"
    positions, types, masses, _ = pmmoto.io.data_read.py_read_lammps_atoms(
        membrane_file, include_mass=True
    )

    all_masses = np.sum(masses)

    dimension = 2
    start = box[dimension][0]
    end = box[dimension][1]
    num_bins = 100
    bin = pmmoto.analysis.bins.Bin(start, end, num_bins)

    sd = pmmoto.initialize(
        voxels=(10, 10, 10), box=box, rank=rank, subdomains=(2, 2, 2)
    )

    unique_types = np.unique(types)
    print(unique_types)
    atom_radii = {}
    atom_masses = {}
    for _type in unique_types:
        atom_radii[_type] = 1
        atom_masses[_type] = atom_id_mass_map[_type]

    membrane = pmmoto.particles.initialize_atoms(
        sd,
        positions,
        atom_radii,
        types,
        atom_masses=atom_masses,
        by_type=False,
        trim_within=True,
    )

    coords = membrane.return_coordinates()
    masses = membrane.return_masses()

    pmmoto.analysis.bins.sum_masses(
        coordinates=coords, dimension=2, bin=bin, masses=masses
    )

    np.testing.assert_approx_equal(np.sum(bin.values), all_masses)
