"""test_io_utils.py"""

import os
import pytest
from pmmoto.io import io_utils


def test_check_file_exists(tmp_path):
    """Test check_file with an existing file"""
    test_file = tmp_path / "test.txt"
    test_file.write_text("test content")

    # Should not raise error
    io_utils.check_file(str(test_file))


def test_check_file_not_exists(tmp_path):
    """Test check_file with a non-existent file"""
    test_file = tmp_path / "nonexistent.txt"

    # Should raise error
    with pytest.raises(SystemExit):
        io_utils.check_file(str(test_file))


def test_check_file_path_creates_directory(tmp_path):
    """Test check_file_path creates directory structure"""
    file_path = tmp_path / "subdir1" / "subdir2" / "test.txt"

    io_utils.check_file_path(str(file_path), create_folder=False)

    # Directory should be created
    assert (tmp_path / "subdir1" / "subdir2").exists()
    assert (tmp_path / "subdir1" / "subdir2").is_dir()


def test_check_file_path_creates_folder_with_filename(tmp_path):
    """Test check_file_path creates folder with file name"""
    file_path = tmp_path / "test_output"

    io_utils.check_file_path(str(file_path), create_folder=True)

    # Folder with name should be created
    assert (tmp_path / "test_output").exists()
    assert (tmp_path / "test_output").is_dir()


def test_check_file_path_with_extra_info(tmp_path):
    """Test check_file_path with extra_info parameter"""
    file_path = tmp_path / "test_output"
    extra = "_results"

    io_utils.check_file_path(str(file_path), create_folder=True, extra_info=extra)

    # Folder with name + extra_info should be created
    assert (tmp_path / "test_output_results").exists()
    assert (tmp_path / "test_output_results").is_dir()


def test_check_file_path_no_create_folder(tmp_path):
    """Test check_file_path with create_folder=False"""
    file_path = tmp_path / "subdir" / "test.txt"

    io_utils.check_file_path(str(file_path), create_folder=False)

    # Only parent directory should be created, not folder with filename
    assert (tmp_path / "subdir").exists()
    assert not (tmp_path / "subdir" / "test.txt").exists()


def test_check_file_path_existing_directory(tmp_path):
    """Test check_file_path with existing directory"""
    subdir = tmp_path / "existing_dir"
    subdir.mkdir()
    file_path = subdir / "test.txt"

    # Should not raise error
    io_utils.check_file_path(str(file_path), create_folder=False)

    assert subdir.exists()


def test_check_file_path_no_directory_component(tmp_path):
    """Test check_file_path with just a filename (no directory)"""
    os.chdir(tmp_path)
    file_name = "simple_file.txt"

    io_utils.check_file_path(file_name, create_folder=True)

    # Should create folder with filename
    assert (tmp_path / "simple_file.txt").exists()
    assert (tmp_path / "simple_file.txt").is_dir()


def test_check_file_path_nested_with_extra_info(tmp_path):
    """Test check_file_path with nested path and extra_info"""
    file_path = tmp_path / "level1" / "level2" / "output"
    extra = "_data"

    io_utils.check_file_path(str(file_path), create_folder=True, extra_info=extra)

    # Parent directories should exist
    assert (tmp_path / "level1" / "level2").exists()
    # Folder with name + extra should exist
    assert (tmp_path / "level1" / "level2" / "output_data").exists()


def test_check_file_path_extra_info_no_create_folder(tmp_path):
    """Test check_file_path with extra_info but create_folder=False"""
    file_path = tmp_path / "subdir" / "test.txt"
    extra = "_extra"

    io_utils.check_file_path(str(file_path), create_folder=False, extra_info=extra)

    # Parent directory should exist
    assert (tmp_path / "subdir").exists()
    # But folder with name + extra should NOT be created
    assert not (tmp_path / "subdir" / "test.txt_extra").exists()


def test_check_num_files_matching():
    """Test check_num_files with matching counts"""
    # Should not raise error
    io_utils.check_num_files(4, 4)


def test_check_num_files_not_matching():
    """Test check_num_files with non-matching counts"""
    with pytest.raises(SystemExit):
        io_utils.check_num_files(4, 8)


def test_check_num_files_zero():
    """Test check_num_files with zero values"""
    # Should not raise error when both are zero
    io_utils.check_num_files(0, 0)


def test_check_num_files_one_zero():
    """Test check_num_files with one zero value"""
    with pytest.raises(SystemExit):
        io_utils.check_num_files(0, 4)


def test_check_folder_exists(tmp_path):
    """Test check_folder with existing folder"""
    test_folder = tmp_path / "test_folder"
    test_folder.mkdir()

    # Should not raise error
    io_utils.check_folder(str(test_folder))


def test_check_folder_not_exists(tmp_path):
    """Test check_folder with non-existent folder"""
    test_folder = tmp_path / "nonexistent_folder"

    with pytest.raises(SystemExit):
        io_utils.check_folder(str(test_folder))


def test_check_folder_is_file(tmp_path):
    """Test check_folder with a file instead of folder"""
    test_file = tmp_path / "test.txt"
    test_file.write_text("content")

    with pytest.raises(SystemExit):
        io_utils.check_folder(str(test_file))


def test_check_file_path_complex_scenario(tmp_path):
    """Test check_file_path with complex nested structure"""
    file_path = tmp_path / "a" / "b" / "c" / "d" / "output.dat"

    io_utils.check_file_path(str(file_path), create_folder=True, extra_info="_final")

    # All parent directories should exist
    assert (tmp_path / "a" / "b" / "c" / "d").exists()
    # Folder with filename + extra should exist
    assert (tmp_path / "a" / "b" / "c" / "d" / "output.dat_final").exists()


def test_check_file_path_idempotent(tmp_path):
    """Test check_file_path is idempotent (can be called multiple times)"""
    file_path = tmp_path / "subdir" / "test.txt"

    # Call multiple times
    io_utils.check_file_path(str(file_path), create_folder=True)
    io_utils.check_file_path(str(file_path), create_folder=True)
    io_utils.check_file_path(str(file_path), create_folder=True)

    # Should still work and directory should exist
    assert (tmp_path / "subdir").exists()
    assert (tmp_path / "subdir" / "test.txt").exists()


def test_check_file_path_relative_path(tmp_path):
    """Test check_file_path with relative path"""
    os.chdir(tmp_path)
    file_path = "relative/path/file.txt"

    io_utils.check_file_path(file_path, create_folder=True)

    assert (tmp_path / "relative" / "path").exists()
    assert (tmp_path / "relative" / "path" / "file.txt").exists()
