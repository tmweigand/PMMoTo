import pytest
import numpy as np

@pytest.fixture
def domain_data():
    """
    Domain data to pass into tests
    """
    data = {
        'size_domain':np.array([
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
        'length_domain':np.array([
            100,
            100,
            100
            ]),
        'voxel':np.array([
            1,
            1,
            1
            ]),
    }

    return data
