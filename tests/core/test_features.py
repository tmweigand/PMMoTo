"""test_features.py"""

import numpy as np
import pytest
import pmmoto


def test_feature():
    """Test the Feature base class"""
    feature_id = (0, 0, 0)
    neighbor_rank = 0
    boundary_type = pmmoto.BoundaryType.END

    # Non_feature because all are 0
    non_feature = pmmoto.core.features.Feature(
        dim=3,
        feature_id=feature_id,
        neighbor_rank=neighbor_rank,
        boundary_type=boundary_type,
    )

    assert non_feature.dim == 3
    assert non_feature.neighbor_rank == neighbor_rank
    assert non_feature.boundary_type == boundary_type


def test_face():
    """Test the face feature class"""
    feature_id = (1, 0, 0)
    neighbor_rank = 0
    boundary_type = pmmoto.BoundaryType.PERIODIC

    feature = pmmoto.core.subdomain_features.Face(
        feature_id=feature_id, neighbor_rank=neighbor_rank, boundary_type=boundary_type
    )

    assert feature.info.opp == (-1, 0, 0)
    assert feature.boundary_type == pmmoto.BoundaryType.PERIODIC
    assert feature.get_periodic_correction() == (-1, 0, 0)

    feature_id = (0, -1, 0)
    neighbor_rank = 0
    boundary_type = pmmoto.BoundaryType.END

    feature = pmmoto.core.subdomain_features.Face(
        feature_id=feature_id, neighbor_rank=neighbor_rank, boundary_type=boundary_type
    )

    assert feature.info.opp == (0, 1, 0)
    assert feature.boundary_type == pmmoto.BoundaryType.END
    assert feature.get_periodic_correction() == (0, 0, 0)


def test_edge():
    """Test the edge feature class"""
    feature_id = (1, 1, 0)
    neighbor_rank = 0
    boundary_type = [pmmoto.BoundaryType.INTERNAL, pmmoto.BoundaryType.PERIODIC]

    feature = pmmoto.core.subdomain_features.Edge(
        feature_id=feature_id, neighbor_rank=neighbor_rank, boundary_type=boundary_type
    )

    assert feature.info.opp == (-1, -1, 0)
    assert feature.boundary_type == (
        pmmoto.BoundaryType.INTERNAL,
        pmmoto.BoundaryType.PERIODIC,
    )
    assert feature.get_periodic_correction() == (0, 0, 0)

    feature_id = (-1, 0, -1)
    neighbor_rank = 0
    boundary_type = [pmmoto.BoundaryType.END, pmmoto.BoundaryType.END]

    feature = pmmoto.core.subdomain_features.Edge(
        feature_id=feature_id, neighbor_rank=neighbor_rank, boundary_type=boundary_type
    )

    assert feature.info.opp == (1, 0, 1)
    assert feature.boundary_type == (pmmoto.BoundaryType.END, pmmoto.BoundaryType.END)
    assert feature.get_periodic_correction() == (0, 0, 0)


def test_corner():
    """Test the corner feature class"""
    feature_id = (1, 1, -1)
    neighbor_rank = 0
    boundary_type = [
        pmmoto.BoundaryType.PERIODIC,
        pmmoto.BoundaryType.PERIODIC,
        pmmoto.BoundaryType.PERIODIC,
    ]

    feature = pmmoto.core.subdomain_features.Corner(
        feature_id=feature_id, neighbor_rank=neighbor_rank, boundary_type=boundary_type
    )

    assert feature.info.opp == (-1, -1, 1)
    assert feature.boundary_type == (
        pmmoto.BoundaryType.PERIODIC,
        pmmoto.BoundaryType.PERIODIC,
        pmmoto.BoundaryType.PERIODIC,
    )
    assert feature.periodic
    assert feature.get_periodic_correction() == (-1, -1, 1)

    feature_id = (-1, 1, -1)
    neighbor_rank = 0
    boundary_type = [
        pmmoto.BoundaryType.END,
        pmmoto.BoundaryType.WALL,
        pmmoto.BoundaryType.PERIODIC,
    ]

    feature = pmmoto.core.subdomain_features.Corner(
        feature_id=feature_id, neighbor_rank=neighbor_rank, boundary_type=boundary_type
    )

    assert feature.info.opp == (1, -1, 1)
    assert feature.boundary_type == (
        pmmoto.BoundaryType.END,
        pmmoto.BoundaryType.WALL,
        pmmoto.BoundaryType.PERIODIC,
    )
    assert feature.get_periodic_correction() == (0, 0, 0)


@pytest.mark.parametrize(
    "feature_id, boundary_type, pad, expected_no_extract, expected_extract",
    [
        # Case 1: face = (-1,0,0), WALL
        (
            (-1, 0, 0),
            pmmoto.BoundaryType.WALL,
            ((1, 1), (1, 1), (1, 1)),
            np.array([[1, 2], [1, 9], [1, 9]]),
            np.array([[1, 2], [2, 8], [2, 8]]),
        ),
        # Case 2: face = (-1,0,0), END
        (
            (-1, 0, 0),
            pmmoto.BoundaryType.END,
            ((0, 0), (0, 0), (1, 1)),
            np.array([[0, 1], [0, 10], [1, 9]]),
            np.array([[0, 1], [1, 9], [2, 8]]),
        ),
        # Case 3: face = (0,1,0), WALL
        (
            (0, 1, 0),
            pmmoto.BoundaryType.WALL,
            ((1, 1), (1, 1), (1, 1)),
            np.array([[1, 9], [8, 9], [1, 9]]),
            np.array([[2, 8], [8, 9], [2, 8]]),
        ),
        # Case 4: face = (0,1,0), END
        (
            (0, 1, 0),
            pmmoto.BoundaryType.END,
            ((0, 0), (0, 0), (1, 1)),
            np.array([[0, 10], [9, 10], [1, 9]]),
            np.array([[1, 9], [9, 10], [2, 8]]),
        ),
    ],
)
def test_get_voxels_face(
    feature_id, boundary_type, pad, expected_no_extract, expected_extract
):
    """Parameterized test for determining feature voxels"""
    voxels = (10, 10, 10)
    feature = pmmoto.core.subdomain_features.Face(
        feature_id=feature_id, neighbor_rank=0, boundary_type=boundary_type
    )

    own, _ = feature.get_voxels(voxels, pad, extract_features=False)
    np.testing.assert_array_equal(own, expected_no_extract)

    own, _ = feature.get_voxels(voxels, pad, extract_features=True)
    np.testing.assert_array_equal(own, expected_extract)


@pytest.mark.parametrize(
    "feature_id, boundary_type, pad, expected_no_extract, expected_extract",
    [
        # Case 1: edge = (-1,0,-1), WALL
        (
            (-1, 0, -1),
            (pmmoto.BoundaryType.WALL, pmmoto.BoundaryType.WALL),
            ((1, 1), (1, 1), (1, 1)),
            np.array([[1, 2], [1, 9], [1, 2]]),
            np.array([[1, 2], [2, 8], [1, 2]]),
        ),
        # Case 2: edge = (-1,0,-1), END
        (
            (-1, 0, -1),
            (pmmoto.BoundaryType.END, pmmoto.BoundaryType.END),
            ((0, 0), (0, 0), (1, 1)),
            np.array([[0, 1], [0, 10], [1, 2]]),
            np.array([[0, 1], [1, 9], [1, 2]]),
        ),
        # Case 3: edge = (0,1,1), WALL
        (
            (0, 1, 1),
            (pmmoto.BoundaryType.WALL, pmmoto.BoundaryType.WALL),
            ((1, 1), (1, 1), (1, 1)),
            np.array([[1, 9], [8, 9], [8, 9]]),
            np.array([[2, 8], [8, 9], [8, 9]]),
        ),
        # Case 4: edge = (0,1,1), END
        (
            (0, 1, 1),
            (pmmoto.BoundaryType.END, pmmoto.BoundaryType.END),
            ((0, 0), (0, 0), (1, 1)),
            np.array([[0, 10], [9, 10], [8, 9]]),
            np.array([[1, 9], [9, 10], [8, 9]]),
        ),
    ],
)
def test_get_voxels_edges(
    feature_id, boundary_type, pad, expected_no_extract, expected_extract
):
    """Parameterized test for determining feature voxels"""
    voxels = (10, 10, 10)
    feature = pmmoto.core.subdomain_features.Edge(
        feature_id=feature_id, neighbor_rank=0, boundary_type=boundary_type
    )

    own, _ = feature.get_voxels(voxels, pad, extract_features=False)
    np.testing.assert_array_equal(own, expected_no_extract)

    own, _ = feature.get_voxels(voxels, pad, extract_features=True)
    np.testing.assert_array_equal(own, expected_extract)


@pytest.mark.parametrize(
    "feature_id, boundary_type, pad, expected_no_extract, expected_extract",
    [
        # Case 1: corner = (-1,1,-1), WALL
        (
            (-1, 1, -1),
            (
                pmmoto.BoundaryType.WALL,
                pmmoto.BoundaryType.WALL,
                pmmoto.BoundaryType.WALL,
            ),
            ((1, 1), (1, 1), (1, 1)),
            np.array([[1, 2], [8, 9], [1, 2]]),
            np.array([[1, 2], [8, 9], [1, 2]]),
        ),
        # Case 2: corner = (-1,1,-1), END
        (
            (-1, 1, -1),
            (pmmoto.BoundaryType.END, pmmoto.BoundaryType.END, pmmoto.BoundaryType.END),
            ((0, 0), (0, 0), (1, 1)),
            np.array([[0, 1], [9, 10], [1, 2]]),
            np.array([[0, 1], [9, 10], [1, 2]]),
        ),
        # Case 3: corner = (1,1,1), WALL
        (
            (1, 1, 1),
            (
                pmmoto.BoundaryType.WALL,
                pmmoto.BoundaryType.WALL,
                pmmoto.BoundaryType.WALL,
            ),
            ((1, 1), (1, 1), (1, 1)),
            np.array([[8, 9], [8, 9], [8, 9]]),
            np.array([[8, 9], [8, 9], [8, 9]]),
        ),
        # Case 4: corner = (1,1,1), END
        (
            (1, 1, 1),
            (pmmoto.BoundaryType.END, pmmoto.BoundaryType.END, pmmoto.BoundaryType.END),
            ((0, 0), (0, 0), (1, 1)),
            np.array([[9, 10], [9, 10], [8, 9]]),
            np.array([[9, 10], [9, 10], [8, 9]]),
        ),
    ],
)
def test_get_voxels_corners(
    feature_id, boundary_type, pad, expected_no_extract, expected_extract
):
    """Parameterized test for determining feature voxels"""
    voxels = (10, 10, 10)
    feature = pmmoto.core.subdomain_features.Corner(
        feature_id=feature_id, neighbor_rank=0, boundary_type=boundary_type
    )

    own, _ = feature.get_voxels(voxels, pad, extract_features=False)
    np.testing.assert_array_equal(own, expected_no_extract)

    own, _ = feature.get_voxels(voxels, pad, extract_features=True)
    np.testing.assert_array_equal(own, expected_extract)


def test_get_octant_vertices():
    """Test for determining feature voxels"""
    voxels = (10, 10, 10)

    # Face
    feature_id = (1, 0, 0)
    boundary_type = [
        pmmoto.BoundaryType.END,
        pmmoto.BoundaryType.INTERNAL,
        pmmoto.BoundaryType.PERIODIC,
    ]
    for b_type in boundary_type:
        boundary_type = pmmoto.BoundaryType.END
        feature = pmmoto.core.subdomain_features.Face(
            feature_id=feature_id, neighbor_rank=0, boundary_type=b_type
        )

        vertices = feature.get_octant_vertices()
        assert vertices == [9, 10, 11, 12, 13, 14, 15, 16, 17]

    boundary_type = pmmoto.BoundaryType.WALL
    feature = pmmoto.core.subdomain_features.Face(
        feature_id=feature_id, neighbor_rank=0, boundary_type=boundary_type
    )
    vertices = feature.get_octant_vertices()
    assert vertices == list(range(18))

    # Edge
    feature_id = (0, -1, -1)
    boundary_type = [
        [pmmoto.BoundaryType.END, pmmoto.BoundaryType.END],
        [pmmoto.BoundaryType.INTERNAL, pmmoto.BoundaryType.INTERNAL],
        [pmmoto.BoundaryType.PERIODIC, pmmoto.BoundaryType.PERIODIC],
    ]
    for b_type in boundary_type:
        boundary_type = pmmoto.BoundaryType.END
        feature = pmmoto.core.subdomain_features.Edge(
            feature_id=feature_id, neighbor_rank=0, boundary_type=b_type
        )

        vertices = feature.get_octant_vertices()
        assert vertices == [4, 5, 7, 13, 14, 16, 22, 23, 25]

    boundary_type = [
        [pmmoto.BoundaryType.END, pmmoto.BoundaryType.WALL],
        [pmmoto.BoundaryType.WALL, pmmoto.BoundaryType.END],
    ]
    for b_type in boundary_type:
        feature = pmmoto.core.subdomain_features.Edge(
            feature_id=feature_id, neighbor_rank=0, boundary_type=b_type
        )
        vertices = feature.get_octant_vertices()
        assert vertices == [4, 5, 7, 8, 13, 14, 16, 17, 22, 23, 25, 26]

    # Corner
    feature_id = (-1, 1, 1)
    boundary_type = [
        [pmmoto.BoundaryType.END, pmmoto.BoundaryType.END, pmmoto.BoundaryType.END],
        [
            pmmoto.BoundaryType.INTERNAL,
            pmmoto.BoundaryType.INTERNAL,
            pmmoto.BoundaryType.INTERNAL,
        ],
        [
            pmmoto.BoundaryType.PERIODIC,
            pmmoto.BoundaryType.PERIODIC,
            pmmoto.BoundaryType.PERIODIC,
        ],
    ]
    for b_type in boundary_type:
        boundary_type = pmmoto.BoundaryType.END
        feature = pmmoto.core.subdomain_features.Corner(
            feature_id=feature_id, neighbor_rank=0, boundary_type=b_type
        )

        vertices = feature.get_octant_vertices()
        assert vertices == [9, 10, 12, 13, 19, 21, 22]

    boundary_type = [
        [
            pmmoto.BoundaryType.END,
            pmmoto.BoundaryType.WALL,
            pmmoto.BoundaryType.PERIODIC,
        ],
        [
            pmmoto.BoundaryType.WALL,
            pmmoto.BoundaryType.END,
            pmmoto.BoundaryType.INTERNAL,
        ],
    ]
    for b_type in boundary_type:
        feature = pmmoto.core.subdomain_features.Corner(
            feature_id=feature_id, neighbor_rank=0, boundary_type=b_type
        )
        vertices = feature.get_octant_vertices()
        assert vertices == [9, 10, 12, 13, 18, 19, 21, 22]
