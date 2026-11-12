"""test_data_read.py"""

import gzip
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

    data, c_domain, _ = pmmoto.io.data_read.read_lammps_atoms(
        membrane_file,
        {"type": np.uint8, "x": float, "y": float, "z": float},
    )

    np.testing.assert_array_equal(
        positions, np.column_stack([data["x"], data["y"], data["z"]])
    )
    np.testing.assert_array_equal(types, data["type"])
    np.testing.assert_array_equal(domain, c_domain)
    assert data["type"].dtype == np.uint8


def test_read_lammps_atoms_extra_columns():
    """Columns outside the old fixed layout can be requested by name"""
    membrane_file = "tests/test_data/LAMMPS/membranedata.gz"
    data, domain, timestep = pmmoto.io.data_read.read_lammps_atoms(
        membrane_file,
        {
            "id": np.uint64,
            "mass": float,
            "q": float,
            "v_peratompress": float,
            "c_peratomvol[1]": float,
        },
    )

    assert list(data) == ["id", "mass", "q", "v_peratompress", "c_peratomvol[1]"]
    assert data["id"].dtype == np.uint64
    assert data["id"][0] == 290944
    assert data["mass"][0] == 16.0
    assert data["q"][0] == -0.587509
    assert data["v_peratompress"][0] == -1.34184e06
    assert data["c_peratomvol[1]"][0] == 14.5779
    assert len(data["id"]) == 465815
    assert timestep == 100005000
    assert domain.shape == (3, 2)


def write_lammps(path, header_columns, rows, time=10, n_atoms=None, newline="\n"):
    """Write a minimal LAMMPS dump file"""
    n_atoms = len(rows) if n_atoms is None else n_atoms
    lines = [
        "ITEM: TIMESTEP",
        str(time),
        "ITEM: NUMBER OF ATOMS",
        str(n_atoms),
        "ITEM: BOX BOUNDS pp pp pp",
        "0 1",
        "-2 2.5",
        "0 10",
        "ITEM: ATOMS " + " ".join(header_columns),
    ] + [" ".join(str(v) for v in row) for row in rows]
    path.write_text(newline.join(lines) + newline)
    return str(path)


def test_read_lammps_atoms_arbitrary_layout(tmp_path):
    """Column order, extra columns and unwrapped coordinates are handled"""
    file_in = write_lammps(
        tmp_path / "dump.lammps",
        ["id", "foo", "xu", "yu", "zu", "type", "x", "y", "z"],
        [
            [3, "a", 1.5, 2.5, 3.5, 2, 0.5, 0.25, 0.125],
            [1, "b", -1e-3, 2e2, 0, 7, 1, 2, 3],
        ],
    )

    data, domain, timestep = pmmoto.io.data_read.read_lammps_atoms(
        file_in,
        {"zu": float, "id": int, "type": np.uint8, "x": np.float64},
    )

    assert set(data) == {"zu", "id", "type", "x"}
    np.testing.assert_array_equal(data["id"], [3, 1])
    np.testing.assert_array_equal(data["type"], [2, 7])
    np.testing.assert_array_equal(data["zu"], [3.5, 0.0])
    np.testing.assert_array_equal(data["x"], [0.5, 1.0])
    assert data["id"].dtype == np.int64
    assert data["type"].dtype == np.uint8
    assert data["zu"].dtype == np.float64
    np.testing.assert_array_equal(domain, [[0, 1], [-2, 2.5], [0, 10]])
    assert timestep == 10


def test_read_lammps_atoms_gzip_and_crlf(tmp_path):
    """Gzipped files and Windows line endings are read"""
    plain = write_lammps(
        tmp_path / "dump.lammps",
        ["id", "q"],
        [[1, 0.5], [2, -0.5]],
        newline="\r\n",
    )
    gz_file = str(tmp_path / "dump.lammps.gz")
    with open(plain, "rb") as f_in, gzip.open(gz_file, "wb") as f_out:
        f_out.write(f_in.read())

    for file_in in (plain, gz_file):
        data, _, _ = pmmoto.io.data_read.read_lammps_atoms(
            file_in, {"id": int, "q": float}
        )
        np.testing.assert_array_equal(data["id"], [1, 2])
        np.testing.assert_array_equal(data["q"], [0.5, -0.5])


def test_read_lammps_atoms_zero_atoms(tmp_path):
    """A file with no atoms returns empty arrays"""
    file_in = write_lammps(tmp_path / "dump.lammps", ["id", "x"], [])
    data, _, _ = pmmoto.io.data_read.read_lammps_atoms(file_in, {"id": int, "x": float})
    assert data["id"].shape == (0,)
    assert data["x"].shape == (0,)


def test_read_lammps_atoms_timestep_conversion(tmp_path):
    """Timestep is converted to simulation time when requested"""
    file_in = write_lammps(tmp_path / "dump.lammps", ["id"], [[1]], time=1500)
    _, _, timestep = pmmoto.io.data_read.read_lammps_atoms(
        file_in, {"id": int}, initial_iteration=500, simulation_timestep=0.002
    )
    assert timestep == pytest.approx(2.0)

    with pytest.raises(ValueError):
        pmmoto.io.data_read.read_lammps_atoms(
            file_in, {"id": int}, initial_iteration=500
        )


def test_read_lammps_atoms_errors(tmp_path):
    """Bad requests and malformed files raise informative errors"""
    read = pmmoto.io.data_read.read_lammps_atoms
    good = write_lammps(tmp_path / "good.lammps", ["id", "x"], [[1, 0.5], [2, 1.5]])

    with pytest.raises(ValueError, match="missing"):
        read(good, {"missing": float})

    with pytest.raises(ValueError):
        read(good, {})

    with pytest.raises(TypeError):
        read(good, {"id": str})

    with pytest.raises(TypeError):
        read(good, {1: int})

    # Non-integer value requested as int
    bad_int = write_lammps(tmp_path / "bad_int.lammps", ["id", "x"], [[1, 0.5]])
    with pytest.raises(ValueError, match="'x'"):
        read(bad_int, {"x": int})

    # Non-numeric value requested as float
    bad_float = write_lammps(tmp_path / "bad_float.lammps", ["id", "x"], [[1, "abc"]])
    with pytest.raises(ValueError, match="'x'"):
        read(bad_float, {"x": float})

    # Fewer atom lines than declared
    short = write_lammps(tmp_path / "short.lammps", ["id", "x"], [[1, 0.5]], n_atoms=2)
    with pytest.raises(ValueError, match="atoms"):
        read(short, {"id": int})

    # Atom line with fewer columns than the header
    narrow = write_lammps(tmp_path / "narrow.lammps", ["id", "x"], [[1]])
    with pytest.raises(ValueError, match="columns"):
        read(narrow, {"x": float})

    # Missing ATOMS header
    no_header = tmp_path / "no_header.lammps"
    no_header.write_text("ITEM: TIMESTEP\n0\n" * 5)
    with pytest.raises(ValueError, match="ATOMS"):
        read(str(no_header), {"id": int})

    # Empty file
    empty = tmp_path / "empty.lammps"
    empty.write_text("")
    with pytest.raises(ValueError):
        read(str(empty), {"id": int})


def test_map_types_by_charge():
    """Types are remapped using (type, charge) pairs"""
    types = np.array([1, 1, 3, 3], dtype=np.uint8)
    charges = np.array([0.5, -0.5, 0.25, 0.25])
    type_map = {(1, 0.5): 10, (1, -0.5): 11, (3, 0.25): 12}

    mapped = pmmoto.io.data_read.map_types_by_charge(types, charges, type_map)

    np.testing.assert_array_equal(mapped, [10, 11, 12, 12])
    assert mapped.dtype == np.uint8

    with pytest.raises(ValueError, match="No mapping"):
        pmmoto.io.data_read.map_types_by_charge(types, charges, {(1, 0.5): 10})


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
