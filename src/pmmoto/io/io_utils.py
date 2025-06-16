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


def check_file_path(file_name: str) -> None:
    """Ensure pathways exists, if not make it"""
    paths = file_name.split("/")[0:-1]
    pathway = ""
    for p in paths:
        pathway = pathway + p + "/"
    if not os.path.isdir(pathway):
        os.makedirs(pathway)

    # Same for individual procs data
    if not os.path.isdir(file_name):
        os.makedirs(file_name)


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
