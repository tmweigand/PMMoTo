"""test_porous_media.py"""

import numpy as np
import pmmoto


def test_porosity():
    """
    Ensures correct calculation of porosity
    """
    sd = pmmoto.initialize(voxels=(10, 10, 10), boundary_types=((1, 1), (1, 1), (1, 1)))
    img = np.zeros(sd.voxels)
    img[3:6, 3:6, 3:6] = 1
    pm = pmmoto.domain_generation.porousmedia.gen_pm(sd, img)

    assert pm.porosity == (27 / 1000)
