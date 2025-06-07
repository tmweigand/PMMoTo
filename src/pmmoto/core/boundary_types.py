"""boundary_types.py

Possible boundary types for PMMoTo
"""

from enum import Enum

__all__ = ["boundary_order"]


class BoundaryType(str, Enum):
    """Allowable boundary types."""

    END = "end"
    WALL = "wall"
    PERIODIC = "periodic"
    INTERNAL = "internal"


def boundary_order(boundaries: list[BoundaryType]) -> BoundaryType:
    """Prove the boundary type for joining boundaries.

    Fro edges and corners in 3D, this specifies the boundary type for the feature.
    #TODO  Get rid of this and allow edges and corners to carry all types.

    Args:
        boundaries (list[BoundaryType]): boundaries at given feature

    Returns:
        BoundaryType: The one boundary type for the feature

    """
    if BoundaryType.END in boundaries:
        return BoundaryType.END
    elif BoundaryType.WALL in boundaries:
        return BoundaryType.WALL
    else:
        return BoundaryType.PERIODIC
