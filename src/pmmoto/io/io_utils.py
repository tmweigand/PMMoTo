"""Ineput/Output Utility Functions"""

import os
from ..core import utils


def check_file(file_name):
    """
    Check file name
    """
    if os.path.isfile(file_name):
        return True
    else:
        print(f"Warning. {file_name} Does Not Exist")
        utils.raise_error()


def check_file_path(file_name):
    """
    Ensure pathways exists, if not make it
    """
    paths = file_name.split("/")[0:-1]
    pathway = ""
    for p in paths:
        pathway = pathway + p + "/"
    if not os.path.isdir(pathway):
        os.makedirs(pathway)

    # Same for individual procs data
    if not os.path.isdir(file_name):
        os.makedirs(file_name)


def check_num_files(num_files, size):
    """
    Check makesure num_files is equal to mpi.size
    """

    if num_files != size:
        print(f"Error: Number of Procs {(size)} Must Be Same As When Written")
        utils.raise_error()


def check_folder(folder_name):
    """
    Check to make sure folder exists
    """
    if os.path.isdir(folder_name):
        return True
    else:
        print(f"Warning. {folder_name} Does Not Exist")
        utils.raise_error()
