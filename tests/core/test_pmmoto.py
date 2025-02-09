"""test_pmmoto.py"""

import pmmoto
import pytest
import numpy as np


@pytest.mark.mpi(min_size=8)
def test_deconstruct_grid(generate_single_subdomain):
    """Ensure expected behavior of deconstruct_grid"""
    sd = generate_single_subdomain(0, periodic=True)

    n = sd.domain.voxels[0]
    linear_values = np.linspace(0, n - 1, n, endpoint=True)
    img = np.ones(sd.domain.voxels) * linear_values

    subdomains, local_img = pmmoto.core.pmmoto.deconstruct_grid(
        sd, img, subdomains=(2, 2, 2)
    )

    subdomains, local_img = pmmoto.core.pmmoto.deconstruct_grid(
        sd, img, subdomains=(2, 2, 2), rank=2
    )
