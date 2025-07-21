"""test_edt.py"""

import pmmoto
import numpy as np


def test_neighborhood() -> None:
    """Test the Neighborhood function for medial axis"""
    img = np.ones((5, 5, 5), dtype=np.uint8)
    neighbors = pmmoto.filters.medial_axis._medial_axis._get_neighborhood(img, 2, 2, 2)
    assert (neighbors == 1).all()
    assert len(neighbors) == 27


def test_get_neighborhood_center_all_zeros():
    img = np.zeros((5, 5, 5), dtype=np.uint8)
    neighbors = pmmoto.filters.medial_axis._medial_axis._get_neighborhood(img, 2, 2, 2)
    assert (neighbors == 0).all()


def test_get_neighborhood_center_voxel():
    # 5x5x5 array filled with increasing integers for easy indexing
    img = np.arange(125, dtype=np.uint8).reshape((5, 5, 5))

    # Get neighbors at the center (2, 2, 2)
    neighbors = pmmoto.filters.medial_axis._medial_axis._get_neighborhood(img, 2, 2, 2)

    # Manually compute expected neighborhood values
    expected = np.array(
        [
            31,
            36,
            41,
            32,
            37,
            42,
            33,
            38,
            43,
            56,
            61,
            66,
            57,
            62,
            67,
            58,
            63,
            68,
            81,
            86,
            91,
            82,
            87,
            92,
            83,
            88,
            93,
        ]
    )

    assert neighbors.shape == (27,)
    assert neighbors.dtype == np.uint8
    np.testing.assert_array_equal(neighbors, expected)


def test_neighborhood_index() -> None:
    """Test the Neighborhood function for medial axis"""
    img = np.ones((5, 5, 5), dtype=np.uint8)
    neighbors = pmmoto.filters.medial_axis._medial_axis._get_neighborhood(
        img, 2, 2, 2, [0, -1, 0]
    )
    print(neighbors)
    # assert (neighbors == 1).all()
    # assert len(neighbors) == 27


# def test_find_simple_points():
#     # 5x5x5 array filled with increasing integers for easy indexing
#     voxels = (10, 10, 10)
#     img = np.zeros(voxels, dtype=np.uint8)
#     img[2:8, :, 2:8] = 1

#     sd = pmmoto.initialize(voxels)
#     edt = pmmoto.filters.distance.edt(img, sd)

#     neighbors = pmmoto.filters.medial_axis._medial_axis._get_neighborhood(img, 2, 2, 2)

#     is_endpoint = pmmoto.filters.medial_axis._medial_axis._is_endpoint(neighbors)
#     is_Euler_invariant = pmmoto.filters.medial_axis._medial_axis._is_Euler_invariant(
#         neighbors
#     )
#     is_simple_point = pmmoto.filters.medial_axis._medial_axis._is_simple_point(
#         neighbors
#     )

#     simples = pmmoto.filters.medial_axis._medial_axis._find_simple_point_candidates(
#         img, 1
#     )

#     print(neighbors, is_endpoint, is_Euler_invariant, is_simple_point)
#     sp = img.copy()
#     for s in simples:
#         sp[s["x"], s["y"], s["z"]] = 2
#         print(s)

#     ma = img.copy()
#     pmmoto.filters.medial_axis._medial_axis._compute_thin_image(ma)

#     # # Get neighbors at the center (2, 2, 2)
#     # simple_points = pmmoto.filters.medial_axis._medial_axis._find_simple_points(
#     #     img, edt
#     # )
#     # print(simple_points)

#     # ma = pmmoto.filters.medial_axis.medial_axis(img)

#     pmmoto.io.output.save_img(
#         "data_out/medial_axis", sd, img, additional_img={"edt": edt, "sp": sp, "ma": ma}
#     )
