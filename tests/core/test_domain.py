import pytest
import numpy as np
import pmmoto

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


def test_init(domain_data):
    """
    Test for checking initialization of domain values
    """
    domain = pmmoto.core.Domain(
        domain_data['size_domain'],
        domain_data['boundaries'],
        domain_data['inlet'],
        domain_data['outlet']
    )

    np.testing.assert_array_equal(domain.size_domain, domain_data['size_domain'])
    assert(domain.boundaries == domain_data['boundaries'])
    assert(domain.inlet == domain_data['inlet'])
    assert(domain.outlet == domain_data['outlet'])
    assert all(domain.length_domain == domain_data['length_domain'])
    assert (domain.periodic == True)
