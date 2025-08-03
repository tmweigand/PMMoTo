"""test_octants.py"""

import pmmoto


def test_octant_mapping():
    """Test for octant mapping order"""
    mapping = pmmoto.core.octants.generate_octant_mapping()

    assert len(mapping) == 8

    import itertools

    expected = list(itertools.product([-1, 1], repeat=3))
    for idx, octant in enumerate(expected):
        assert mapping[octant] == idx


def test_feature_octants():
    """Test for feature to octant mapping"""
    feature = (-1, 0, 0)
    result = pmmoto.core.octants.feature_octants(feature)
    expected = [4, 5, 6, 7]
    assert result == expected

    feature = (1, 1, 1)
    result = pmmoto.core.octants.feature_octants(feature)
    expected = [0]
    assert result == expected

    feature = (0, 0, 0)
    result = pmmoto.core.octants.feature_octants(feature)
    expected = list(range(8))
    assert result == expected

    feature = (0, -1, 1)
    result = pmmoto.core.octants.feature_octants(feature)
    expected = [2, 6]
    assert result == expected


def test_neighbor_vertices():
    feature = (0, 0, 0)
    n_v = pmmoto.core.octants.get_neighbors_vertices(feature)
    assert n_v == list(range(27))

    feature = (1, 0, 0)
    n_v = pmmoto.core.octants.get_neighbors_vertices(feature)
    assert n_v == list(range(18))

    feature = (-1, 0, 0)
    n_v = pmmoto.core.octants.get_neighbors_vertices(feature)
    assert n_v == list(range(9, 27))

    feature = (0, 0, 1)
    n_v = pmmoto.core.octants.get_neighbors_vertices(feature)
    assert n_v == [0, 1, 3, 4, 6, 7, 9, 10, 12, 13, 15, 16, 18, 19, 21, 22, 24, 25]

    feature = (0, -1, -1)
    n_v = pmmoto.core.octants.get_neighbors_vertices(feature)
    assert n_v == [4, 5, 7, 8, 13, 14, 16, 17, 22, 23, 25, 26]

    feature = (-1, 1, 1)
    n_v = pmmoto.core.octants.get_neighbors_vertices(feature)
    assert n_v == [9, 10, 12, 13, 18, 19, 21, 22]

    # Test extraction of vertices along boundaries
    feature = (0, 0, 0)
    n_v = pmmoto.core.octants.get_neighbors_vertices(feature, boundary_face=(1, 0, 0))
    assert n_v == list(range(9, 27))

    feature = (0, 0, 0)
    n_v = pmmoto.core.octants.get_neighbors_vertices(feature, boundary_face=(-1, 0, 0))
    assert n_v == list(range(18))

    feature = (1, 0, 0)
    n_v = pmmoto.core.octants.get_neighbors_vertices(feature, boundary_face=(1, 0, 0))
    assert n_v == [9, 10, 11, 12, 13, 14, 15, 16, 17]

    feature = (0, 0, -1)
    n_v = pmmoto.core.octants.get_neighbors_vertices(feature, boundary_face=(0, 0, 0))
    assert n_v == [1, 2, 4, 5, 7, 8, 10, 11, 13, 14, 16, 17, 19, 20, 22, 23, 25, 26]

    n_v = pmmoto.core.octants.get_neighbors_vertices(feature, boundary_face=(1, 0, 0))
    assert n_v == [10, 11, 13, 14, 16, 17, 19, 20, 22, 23, 25, 26]

    n_v = pmmoto.core.octants.get_neighbors_vertices(feature, boundary_face=(-1, 0, 0))
    assert n_v == [1, 2, 4, 5, 7, 8, 10, 11, 13, 14, 16, 17]

    feature = (1, 1, 0)

    n_v = pmmoto.core.octants.get_neighbors_vertices(feature, boundary_face=(1, 0, 0))
    assert n_v == [9, 10, 11, 12, 13, 14]

    n_v = pmmoto.core.octants.get_neighbors_vertices(feature, boundary_face=(0, 1, 0))
    assert n_v == [3, 4, 5, 12, 13, 14]

    feature = (-1, 1, -1)

    n_v = pmmoto.core.octants.get_neighbors_vertices(feature, boundary_face=(0, 0, 0))
    assert n_v == [10, 11, 13, 14, 19, 20, 22, 23]

    n_v = pmmoto.core.octants.get_neighbors_vertices(feature, boundary_face=(0, 1, 0))
    assert n_v == [13, 14, 22, 23]

    n_v = pmmoto.core.octants.get_neighbors_vertices(feature, boundary_face=(0, 0, -1))
    assert n_v == [10, 13, 19, 22]

    n_v = pmmoto.core.octants.get_neighbors_vertices(feature, boundary_face=(-1, 0, 0))
    assert n_v == [10, 11, 13, 14]
