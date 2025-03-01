"""test_generate_rdf.py"""

import numpy as np
import pytest
import pmmoto


def test_generate_rdf():
    """
    Test for generating a radial distribution function from LAMMPS data
    """

    np.random.seed(42)
    box = [[0, 1], [0, 1], [0, 1]]

    num_atoms = 100
    atoms = np.random.rand(num_atoms, 3)

    # new_atoms = pmmoto.domain_generation._domain_generation.gen_verlet_list()
