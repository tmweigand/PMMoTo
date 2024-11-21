import pytest
import numpy as np
import pmmoto


@pytest.fixture
def padded_subdomain():
    """
    Generate a padded subdomain
    """
    box = ((0, 1), (0, 1), (0, 1))
    boundary_types = ((2, 2), (2, 2), (2, 2))
    voxels = (10, 10, 10)
    subdomains = (1, 1, 1)

    sd = pmmoto.initialize(
        box=box,
        subdomains=subdomains,
        voxels=voxels,
        boundary_types=boundary_types,
        rank=0,
        pad=(1, 1, 1),
    )

    return sd
