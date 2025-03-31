"""test_bins.py"""

import numpy as np
import pmmoto
from mpi4py import MPI
import matplotlib.pyplot as plt
import pytest


def test_bin_centers():
    """
    Dummy check for bins
    """
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


def test_bin():
    """
    Test bins
    """

    start = 0
    end = 3
    num_bins = 25

    bin = pmmoto.analysis.bins.Bin(start=start, end=end, num_bins=num_bins)

    assert bin.width == 0.12

    assert len(bin.centers) == num_bins

    np.testing.assert_array_equal(bin.values, np.zeros(num_bins))

    ones = np.ones(num_bins)
    bin = pmmoto.analysis.bins.Bin(start=start, end=end, num_bins=num_bins, values=ones)

    np.testing.assert_array_equal(bin.values, ones)


def test_bins():
    """
    Test bins
    """

    start = [0, 1]
    end = [3, 2.8]
    num_bins = [25, 50]
    labels = [1, 5]

    bins = pmmoto.analysis.bins.Bins(
        starts=start, ends=end, num_bins=num_bins, labels=labels
    )

    assert bins.bins[1].width == 0.12
    assert bins.bins[5].width == 0.036


def test_count_locations():
    """
    Test binning coordinates
    """

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


def test_count_membrane():
    """
    Test if membrane atoms are being counted.
    """

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
def test_count_membrane_parallel():
    """
    Test if membrane atoms are being counted correctly in parallel
    """
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
        coordinates=coords, dimension=dimension, bin=bin, subdomain=sd
    )

    assert np.sum(bin.values) == positions.shape[0]


def test_sum_masses():
    """
    Tests summing masses of atoms based on location.
    """

    start = 0
    end = 3
    num_bins = 25

    bin = pmmoto.analysis.bins.Bin(start, end, num_bins)

    coordinates = np.array([[0.1, 0.0, 0.0], [1.1, 0.0, 0.0], [2.1, 0.0, 0.0]])

    masses = np.array([3, 12.5, 256])

    mass_counts = pmmoto.analysis.bins.sum_masses(
        coordinates=coordinates, dimension=0, bin=bin, masses=masses
    )

    assert np.sum(mass_counts) == 3 + 12.5 + 256  # Total mass of all atoms

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


def test_sum_membrane_mass():
    """
    Test membrane mass counting.
    """

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
    mass_density = bin.values / bin.volume

    np.testing.assert_approx_equal(np.sum(bin.values), np.sum(masses))


@pytest.mark.mpi(min_size=8)
def test_sum_membrane_mass_parallel():
    """
    Test membrane mass counting.
    """
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    # Full domain with reservoirs
    box = [
        [0.0, 176.0],
        [0.0, 176.0],
        [-287, 237],
    ]

    membrane_file = "tests/test_data/LAMMPS/membranedata.gz"
    positions, types, masses, _ = pmmoto.io.data_read.py_read_lammps_atoms(
        membrane_file, include_mass=True
    )

    num_positions = positions.shape[0]

    dimension = 2
    start = box[dimension][0]
    end = box[dimension][1]
    num_bins = 100
    bin = pmmoto.analysis.bins.Bin(start, end, num_bins)

    sd = pmmoto.initialize(
        voxels=(10, 10, 10), box=box, rank=rank, subdomains=(2, 2, 2)
    )

    unique_types = np.unique(types)
    atom_radii = {}
    atom_masses = {}
    for _type in unique_types:
        atom_radii[_type] = 1
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
    masses = membrane.return_masses()

    pmmoto.analysis.bins.sum_masses(
        coordinates=coords, dimension=2, bin=bin, masses=masses, subdomain=sd
    )

    # Calculate bin volume
    area = (box[0][1] - box[0][0]) * (box[1][1] - box[1][0])

    bin.calculate_volume(area=area)
    mass_density = bin.values / bin.volume

    np.testing.assert_approx_equal(np.sum(bin.values), num_positions * 0.1)
