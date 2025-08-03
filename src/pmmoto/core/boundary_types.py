"""boundary_types.py

Possible boundary types for PMMoTo
"""

from enum import Enum


class BoundaryType(str, Enum):
    """Allowable boundary types."""

    END = "end"
    WALL = "wall"
    PERIODIC = "periodic"
    INTERNAL = "internal"
