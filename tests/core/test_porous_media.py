import numpy as np
import pmmoto


def test_porousmedia_walls():

    sd = pmmoto.initialize(voxels=(10, 10, 10), boundary_types=((1, 1), (1, 1), (1, 1)))

    img = np.ones(sd.voxels)
    img = sd.set_wall_bcs(img)

    assert np.all(img[0, :, :] == 0)
    assert np.all(img[-1, :, :] == 0)
    assert np.all(img[:, 0, :] == 0)
    assert np.all(img[:, -1, :] == 0)
    assert np.all(img[:, :, 0] == 0)
    assert np.all(img[:, :, -1] == 0)

    assert not np.all(img[1, :, :] == 0)
    assert not np.all(img[-2, :, :] == 0)
    assert not np.all(img[:, 1, :] == 0)
    assert not np.all(img[:, -2, :] == 0)
    assert not np.all(img[:, :, 1] == 0)
    assert not np.all(img[:, :, -2] == 0)
