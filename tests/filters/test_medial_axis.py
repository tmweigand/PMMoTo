"""test_edt.py"""

import pmmoto
import numpy as np
import skimage


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
            32,
            33,
            36,
            37,
            38,
            41,
            42,
            43,
            56,
            57,
            58,
            61,
            62,
            63,
            66,
            67,
            68,
            81,
            82,
            83,
            86,
            87,
            88,
            91,
            92,
            93,
        ]
    )

    assert neighbors.shape == (27,)
    assert neighbors.dtype == np.uint8
    np.testing.assert_array_equal(neighbors, expected)


def test_find_skeleton():
    # 5x5x5 array filled with increasing integers for easy indexing
    voxels = (700, 700, 10)
    # voxels = (300, 300, 5)
    img = pmmoto.domain_generation.gen_img_smoothed_random_binary(
        voxels, 0.5, 3, seed=123
    )

    boundary = pmmoto.BoundaryType.END

    boundary_types = ((boundary, boundary), (boundary, boundary), (boundary, boundary))

    sd = pmmoto.initialize(
        voxels,
        boundary_types=boundary_types,
        box=((0, voxels[0]), (0, voxels[1]), (0, voxels[2])),
    )
    edt = pmmoto.filters.distance.edt(img, sd)

    pm_ma = pmmoto.filters.medial_axis.skeleton(sd, img)

    # simples = pmmoto.filters.medial_axis.find_simple_point_candidates(
    #     sd, pm_ma, (1, 0, 0)
    # )

    # sp = pm_ma.copy()
    # for s in simples:
    #     sp[s["x"], s["y"], s["z"]] = 2
    #     # print(s)

    # # ma = img.copy()
    # # pmmoto.filters.medial_axis._medial_axis._compute_thin_image(ma)

    # # # Get neighbors at the center (2, 2, 2)
    # # simple_points = pmmoto.filters.medial_axis._medial_axis._find_simple_points(
    # #     img, edt
    # # )
    # # print(simple_points)

    # ma = pmmoto.filters.medial_axis.medial_axis(img)
    ma_skim = skimage.morphology.skeletonize(img, method="lee")

    # img_cc, img_count = pmmoto.filters.connected_components.connect_components(img, sd)

    # ma_cc, ma_count = pmmoto.filters.connected_components.connect_components(pm_ma, sd)

    # _, skim_count = pmmoto.filters.connected_components.connect_components(ma_skim, sd)

    # print(img_count, ma_count, skim_count)

    # pmmoto.io.output.save_img(
    #     "data_out/medial_axis",
    #     sd,
    #     img,
    #     additional_img={
    #         "edt": edt,
    #         "ma": ma,
    #         "ma_skim": ma_skim.astype(np.uint8),
    #         "pm_ma": pm_ma,
    #         "img_cc": img_cc,
    #         "ma_cc": ma_cc,
    #     },
    # )
