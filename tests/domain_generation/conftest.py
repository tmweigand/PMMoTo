import pytest
import numpy as np


@pytest.fixture
def atom_map():
    """
    Atom map for tests
    """
    data = {1: "A", 2: "B", 5: "aabbss"}

    return data


def atom_data():
    """
    Atom rdf data
    """
