"""test_features.py"""

import numpy as np
import pmmoto


def test_feature():
    """
    test the Feature base class
    """

    feature_id = (0, 0, 0)
    neighbor_rank = [0]
    boundary_type = 0

    feature = pmmoto.core.subdomain_features.Feature(
        feature_id=feature_id, neighbor_rank=neighbor_rank, boundary_type=boundary_type
    )


def test_face():
    """
    Test the face feature class
    """
    feature_id = (1, 0, 0)
    neighbor_rank = [0]
    boundary_type = "periodic"

    feature = pmmoto.core.subdomain_features.Face(
        feature_id=feature_id, neighbor_rank=neighbor_rank, boundary_type=boundary_type
    )

    assert feature.opp_info == (-1, 0, 0)
    assert feature.boundary_type == "periodic"
    assert feature.get_periodic_correction() == (-1, 0, 0)

    feature_id = (0, -1, 0)
    neighbor_rank = [0]
    boundary_type = "end"

    feature = pmmoto.core.subdomain_features.Face(
        feature_id=feature_id, neighbor_rank=neighbor_rank, boundary_type=boundary_type
    )

    assert feature.opp_info == (0, 1, 0)
    assert feature.boundary_type == "end"
    assert feature.get_periodic_correction() == (0, 0, 0)


def test_edge():
    """
    Test the edge feature class
    """
    feature_id = (1, 1, 0)
    neighbor_rank = [0]
    boundary_type = "periodic"

    feature = pmmoto.core.subdomain_features.Edge(
        feature_id=feature_id, neighbor_rank=neighbor_rank, boundary_type=boundary_type
    )

    assert feature.opp_info == (-1, -1, 0)
    assert feature.boundary_type == "periodic"
    assert feature.get_periodic_correction() == (-1, -1, 0)

    feature_id = (-1, 0, -1)
    neighbor_rank = [0, 2]
    boundary_type = "end"

    feature = pmmoto.core.subdomain_features.Edge(
        feature_id=feature_id, neighbor_rank=neighbor_rank, boundary_type=boundary_type
    )

    assert feature.opp_info == (1, 0, 1)
    assert feature.boundary_type == "end"
    assert feature.get_periodic_correction() == (0, 0, 0)


def test_corner():
    """
    Test the corner feature class
    """
    feature_id = (1, 1, -1)
    neighbor_rank = [0]
    boundary_type = "periodic"

    feature = pmmoto.core.subdomain_features.Corner(
        feature_id=feature_id, neighbor_rank=neighbor_rank, boundary_type=boundary_type
    )

    assert feature.opp_info == (-1, -1, 1)
    assert feature.boundary_type == "periodic"
    assert feature.get_periodic_correction() == (-1, -1, 1)

    feature_id = (-1, 1, -1)
    neighbor_rank = [0, 2]
    boundary_type = "end"

    feature = pmmoto.core.subdomain_features.Corner(
        feature_id=feature_id, neighbor_rank=neighbor_rank, boundary_type=boundary_type
    )

    assert feature.opp_info == (1, -1, 1)
    assert feature.boundary_type == "end"
    assert feature.get_periodic_correction() == (0, 0, 0)
