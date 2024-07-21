import pytest
import numpy as np

@pytest.fixture
def domain():
    """
    Domain data to pass into tests
    """
    data = {
        'box':np.array([
            (0,100),
            (0,100),
            (0,100)
            ]),
        'boundaries':(
            (0,0),
            (1,1),
            (2,2)
        ),
        'inlet':(
            (1,0),
            (0,0),
            (0,0)
        ),
        'outlet':(
            (0,1),
            (0,0),
            (0,0)
        ),
        'length':np.array([
            100,
            100,
            100
            ]),
        'num_voxel':np.array([
            1,
            1,
            1
            ]),
    }

    return data

@pytest.fixture
def decomposed_domain():
    """
    Subdomain data to pass into tests
    """
    data = {
        'subdomain_map': (2,2,2)
    }

    return data

@pytest.fixture
def domain_discretization():
    """
    Domain discretization data to pass into tests.
    """
    data = {
        'num_voxels':(100,100,100)
    }

    return data


@pytest.fixture
def subdomain():
    """
    Subdomain data to pass into tests.
    """
    data = {
        'rank':0,
        'index':(0,0,0)
    }

    return data