"""test_subdomain_verlet.py"""

import numpy as np

import pmmoto


def test_verlet_subdomain():
    """Test for subdomain"""
    verlet_domains = (2, 2, 2)
    sd = pmmoto.initialize(voxels=(11, 11, 11), verlet_domains=verlet_domains)

    assert sd.num_verlet == 8

    assert sd.verlet_voxels == (
        (5, 5, 5),
        (5, 5, 6),
        (5, 6, 5),
        (5, 6, 6),
        (6, 5, 5),
        (6, 5, 6),
        (6, 6, 5),
        (6, 6, 6),
    )

    np.testing.assert_array_equal(sd.verlet_loop[0], ([[0, 5], [0, 5], [0, 5]]))

    np.testing.assert_array_equal(sd.verlet_loop[7], ([[5, 11], [5, 11], [5, 11]]))

    np.testing.assert_array_almost_equal(
        sd.centroids[0], [0.22727273, 0.22727273, 0.22727273]
    )

    np.testing.assert_array_almost_equal(
        sd.centroids[7], [0.727273, 0.727273, 0.727273]
    )

    np.testing.assert_allclose(
        sd.max_diameters,
        [
            0.78729582,
            0.84305623,
            0.84305623,
            0.89535071,
            0.84305623,
            0.89535071,
            0.89535071,
            0.94475499,
        ],
    )

    np.testing.assert_allclose(
        sd.verlet_box[0],
        (
            (
                (0.0, 0.4545454545454546),
                (0.0, 0.4545454545454546),
                (0.0, 0.4545454545454546),
            )
        ),
    )

    np.testing.assert_allclose(
        sd.verlet_box[7],
        (
            (
                (
                    (0.45454545454545453, 1.0),
                    (0.45454545454545453, 1.0),
                    (0.45454545454545453, 1.0),
                )
            )
        ),
    )
