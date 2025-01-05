"""test_utils.py"""

import pmmoto
import numpy as np


def test_decompose_img():
    """Ensure expected behavior of decompose_img"""
    n = 5
    img = np.arange(n * n * n).reshape(n, n, n)
    start = (0, 0, 0)
    shape = (3, 3, 3)
    result = pmmoto.core.utils.decompose_img(img, start=start, shape=shape)

    np.testing.assert_array_equal(
        result,
        np.array(
            [
                [[0, 1, 2], [5, 6, 7], [10, 11, 12]],
                [[25, 26, 27], [30, 31, 32], [35, 36, 37]],
                [[50, 51, 52], [55, 56, 57], [60, 61, 62]],
            ]
        ),
    )

    start = (-1, -1, -1)
    shape = (3, 3, 3)
    result = pmmoto.core.utils.decompose_img(img, start=start, shape=shape)

    np.testing.assert_array_equal(
        result,
        np.array(
            [
                [[124, 120, 121], [104, 100, 101], [109, 105, 106]],
                [[24, 20, 21], [4, 0, 1], [9, 5, 6]],
                [[49, 45, 46], [29, 25, 26], [34, 30, 31]],
            ]
        ),
    )
