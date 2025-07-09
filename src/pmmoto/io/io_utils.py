"""Ineput/Output Utility Functions"""

import os
from ..core import utils
from ..core.logging import get_logger, USE_LOGGING

if USE_LOGGING:
    logger = get_logger()


def check_file(file_name: str) -> None:
    """Check file name"""
    if not os.path.isfile(file_name):
        logger.error(
            "%s does not exist!",
            file_name,
        )
        utils.raise_error()


def check_file_path(
    file_name: str, create_folder: bool = True, extra_info: None | str = None
) -> None:
    """Ensure the directory path(s) for the given file_name exist, creating them .

    Parameters
    ----------
    file_name : str
        The full file path or just a file name. If it contains directories,
        those directories will be checked and created if missing.
    create_folder : bool, optional
        If True, also creates a folder with the name formed by concatenating `file_name`
        and `extra_info` (if provided), defaults to True.
    extra_info : str or None, optional
        Optional string appended to `file_name` to form an additional directory path
        to create.

    Notes
    -----
    - If `file_name` contains no directory component (i.e., just a file name),
      no directory will be created for the file itself, but if `create_folder` is True
      and `extra_info` is provided, a directory named `file_name + extra_info` will be
      created.

    """
    # Extract directory path from file_name
    dir_path = os.path.dirname(file_name)

    # Create directories if path exists and does not already exist
    if dir_path and not os.path.isdir(dir_path):
        os.makedirs(dir_path)

    # Handle additional folder creation if extra_info is provided
    if extra_info is not None:
        _file_name = file_name + extra_info
    else:
        _file_name = file_name

    if create_folder and not os.path.isdir(_file_name):
        os.makedirs(_file_name)


def check_num_files(num_files: int, size: int) -> None:
    """Check makesure num_files is equal to mpi.size"""
    if num_files != size:
        print(f"Error: Number of Procs {(size)} Must Be Same As When Written")
        utils.raise_error()


def check_folder(folder_name: str) -> None:
    """Check to make sure folder exists"""
    if os.path.isdir(folder_name):
        pass
    else:
        print(f"Warning. {folder_name} Does Not Exist")
        utils.raise_error()
