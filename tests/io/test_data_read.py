"""test_data_read.py"""

import pmmoto
import pytest
import numpy as np


def test_read_sphere_pack():
    """Test reading of a sphere pack"""
    file_in = "tests/test_data/sphere_packs/bcc.out"
    spheres, domain = pmmoto.io.data_read.read_sphere_pack_xyzr_domain(file_in)

    np.testing.assert_array_equal(
        spheres,
        [
            [0.0, 0.0, 0.0, 0.25],
            [0.0, 0.0, 1.0, 0.25],
            [0.0, 1.0, 0.0, 0.25],
            [1.0, 0.0, 0.0, 0.25],
            [0.0, 1.0, 1.0, 0.25],
            [1.0, 0.0, 1.0, 0.25],
            [1.0, 1.0, 0.0, 0.25],
            [1.0, 1.0, 1.0, 0.25],
        ],
    )

    np.testing.assert_array_equal(domain, ((0.0, 1.0), (0.0, 1.0), (0.0, 1.0)))


def test_read_sphere_pack_xyzr_domain_bad_sphere_line(tmp_path):
    bad_file = tmp_path / "bad.xyzr"

    bad_file.write_text("0 1\n" "0 1\n" "0 1\n" "0.5 0.5 not_a_number 0.1\n")

    with pytest.raises(ValueError):
        pmmoto.io.data_read.read_sphere_pack_xyzr_domain(str(bad_file))


def test_read_sphere_pack_xyzr_domain_invalid_sphere_data(tmp_path):
    bad_file = tmp_path / "bad.xyzr"

    bad_file.write_text("0 1\n" "0 1\n" "0 1\n" "0.5 0.5 0.2 0.2 0.1\n")

    with pytest.raises(ValueError):
        pmmoto.io.data_read.read_sphere_pack_xyzr_domain(str(bad_file))


def test_read_atom_map():
    """Test behavior of read atom map"""
    atom_map_file = "tests/test_data/atom_data/atom_map.txt"

    atom_map = pmmoto.io.data_read.read_atom_map(atom_map_file)

    assert atom_map == {
        1: {"element": "H", "name": "BH1"},
    }


def test_lammps_file_read():
    """Tests reading of LAMMPS files"""
    membrane_file = "tests/test_data/LAMMPS/membranedata.gz"
    positions, types, domain = pmmoto.io.data_read.py_read_lammps_atoms(membrane_file)

    assert positions.ndim == 2
    assert positions.shape[1] == 3
    assert len(types) == positions.shape[0]
    assert domain.shape == (3, 2)

    c_positions, c_types, c_domain, time = pmmoto.io.data_read.read_lammps_atoms(
        membrane_file
    )

    np.testing.assert_array_equal(positions, c_positions)
    np.testing.assert_array_equal(types, c_types)
    np.testing.assert_array_equal(domain, c_domain)


def test_read_binned_distances(tmp_path):
    """Tests for binned distance reads"""
    input_folder = tmp_path / "binned_distances"
    input_folder.mkdir(exist_ok=True)

    # Create a fake atom_map.txt
    atom_map_file = input_folder / "atom_map.txt"
    atom_map_file.write_text("1 H BH1\n" "2 O Oxygen\n")

    for atom_name in ["BH1", "Oxygen"]:
        atom_file = input_folder / f"{atom_name}.rdf"
        # Two columns: radial distance, rdf(r)
        np.savetxt(atom_file, np.array([[0.0, 0.5], [1.0, 1.5]]).T)

    atom_map, rdf_out = pmmoto.io.data_read.read_binned_distances_rdf(str(input_folder))

    # Basic assertions
    assert set(atom_map.keys()) == {1, 2}
    assert set(rdf_out.keys()) == {1, 2}

    for _id, rdf_obj in rdf_out.items():
        assert isinstance(rdf_obj, pmmoto.domain_generation.rdf.RDF)
        assert rdf_obj.name in ["BH1", "Oxygen"]
        assert len(rdf_obj.radii) == 2
        assert len(rdf_obj.rdf) == 2
