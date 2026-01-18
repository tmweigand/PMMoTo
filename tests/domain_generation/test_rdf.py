"""test_rdf.py"""

import pmmoto
import pytest
import numpy as np


def test_rdf():
    """Test for checking rdf values"""
    atom_folder = "tests/test_data/atom_data/"
    atom_map, rdf = pmmoto.io.data_read.read_rdf(atom_folder)

    assert atom_map == {1: {"element": "H", "name": "BH1"}}

    assert rdf[1].interpolate_rdf(3) == 1.1235851734374689


def test_bounded_rdf():
    """Test for checking rdf values"""
    atom_folder = "tests/test_data/atom_data/"
    atom_map, rdf = pmmoto.io.data_read.read_rdf(atom_folder)
    assert atom_map == {
        1: {"element": "H", "name": "BH1"},
    }

    bounded_rdf = pmmoto.domain_generation.rdf.BoundedRDF.from_rdf(rdf[1], eps=1.0e-3)

    assert bounded_rdf.interpolate_radius_from_pmf(5.0) == 2.386833861580646

    assert bounded_rdf.interpolate_radius_from_pmf(
        0.0
    ) == bounded_rdf.interpolate_radius_from_pmf(-100.0)


def test_bounded_rdf_find_radii():
    """Test for checking rdf values"""
    atom_folder = "tests/test_data/atom_data/"
    atom_map, rdf = pmmoto.io.data_read.read_rdf(atom_folder)
    assert atom_map == {
        1: {"element": "H", "name": "BH1"},
    }

    bounded_rdf = pmmoto.domain_generation.rdf.BoundedRDF(
        name="BH1", atom_id=1, radii=rdf[1].radii, rdf=rdf[1].rdf, eps=1.0e-3
    )

    min_radius = bounded_rdf.find_min_radius(rdf[1].rdf, 1.0e-3)
    assert min_radius == 0

    with pytest.raises(ValueError):
        _ = bounded_rdf.find_min_radius(rdf[1].rdf, -1.0e-3)

    max_radius = bounded_rdf.find_max_radius(0.5, rdf[1].rdf)
    assert max_radius == 428

    max_radius = bounded_rdf.find_max_radius(1.0, rdf[1].rdf)
    assert max_radius == 469

    with pytest.raises(ValueError):
        _ = bounded_rdf.find_max_radius(10.0, rdf[1].rdf)


def test_bounded_rdf_bounds():
    """Test for checking rdf values"""
    atom_folder = "tests/test_data/atom_data/"
    atom_map, rdf = pmmoto.io.data_read.read_rdf(atom_folder)
    assert atom_map == {
        1: {"element": "H", "name": "BH1"},
    }

    bounded_rdf = pmmoto.domain_generation.rdf.BoundedRDF(
        name="BH1", atom_id=1, radii=rdf[1].radii, rdf=rdf[1].rdf, eps=1.0e-3
    )

    bounds = bounded_rdf.determine_bounds(rdf[1].rdf, 1.0e-3)
    assert bounds == [0, 469]

    bounds = bounded_rdf.determine_bounds(rdf[1].rdf[0:400], 1.0e-3)
    assert bounds == [0, 399]


def test_bin_distance():
    """Test of bin_distances"""
    sd = pmmoto.initialize((10, 10, 10))
    probe_atom_coordinates = np.array([[0.0, 0.0, 0.0]])
    probe_atom_radii = {1: 2}
    probe_atom_ids = np.array([1])

    atom_coordinates = np.array([[0.5, 0.5, 0.5], [1, 1, 1]])
    atom_radii = {2: 10, 3: 5}
    atom_ids = np.array([2, 3])

    probe_atoms = pmmoto.particles.initialize_atoms(
        sd, probe_atom_coordinates, probe_atom_radii, probe_atom_ids, by_type=True
    )
    atom_list = probe_atoms.return_list(1)

    atoms = pmmoto.particles.initialize_atoms(
        sd, atom_coordinates, atom_radii, atom_ids, by_type=True
    )

    bins = pmmoto.analysis.bins.Bins(
        starts=[0, 0],
        ends=[2, 2],
        num_bins=[10, 10],
        labels=[2, 3],
        names=["test_bin_1", "test_bin_2"],
    )

    pmmoto.domain_generation.rdf.bin_distances(sd, atom_list, atoms, bins)

    np.testing.assert_array_equal(
        bins.bins[2].values,
        np.array([0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
    )

    np.testing.assert_array_equal(
        bins.bins[3].values,
        np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0]),
    )


@pytest.mark.mpi(min_size=8)
def test_bin_distance_parallel():
    """Test of bin_distances

    TODO: edge cases where no atoms in subdomain
    """
    subdomains = (2, 2, 2)
    sd = pmmoto.initialize((10, 10, 10), subdomains=subdomains)
    probe_atom_coordinates = np.array(
        [
            [0.15, 0.15, 0.15],
            [0.85, 0.15, 0.15],
            [0.15, 0.85, 0.15],
            [0.15, 0.15, 0.85],
            [0.85, 0.85, 0.15],
            [0.85, 0.15, 0.85],
            [0.15, 0.85, 0.85],
            [0.85, 0.85, 0.85],
        ]
    )
    probe_atom_radii = {1: 2.0}
    probe_atom_ids = np.array([1, 1, 1, 1, 1, 1, 1, 1])

    atom_coordinates = np.array(
        [
            [0.25, 0.25, 0.25],
            [0.75, 0.25, 0.25],
            [0.25, 0.75, 0.25],
            [0.25, 0.25, 0.75],
            [0.75, 0.75, 0.25],
            [0.75, 0.25, 0.75],
            [0.25, 0.75, 0.75],
            [0.75, 0.75, 0.75],
        ]
    )
    atom_radii = {2: 10.0}
    atom_ids = np.array([2, 2, 2, 2, 2, 2, 2, 2])

    probe_atoms = pmmoto.particles.initialize_atoms(
        sd,
        probe_atom_coordinates,
        probe_atom_radii,
        probe_atom_ids,
        by_type=True,
        trim_within=True,
    )
    atom_list = probe_atoms.return_list(1)

    atoms = pmmoto.particles.initialize_atoms(
        sd,
        atom_coordinates,
        atom_radii,
        atom_ids,
        by_type=True,
        trim_within=True,
    )

    bins = pmmoto.analysis.bins.Bins(
        starts=[0],
        ends=[2],
        num_bins=[10],
        labels=[2],
        names=["test_bin_1", "test_bin_2"],
    )

    pmmoto.domain_generation.rdf.bin_distances(sd, atom_list, atoms, bins)

    np.testing.assert_array_equal(
        bins.bins[2].values,
        np.array([8.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
    )
