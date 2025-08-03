"""medial_axis.py

Tools for extracting a medial axis or skeleton.
"""

from ._skeletonize_3d_cy import _compute_thin_image
from ._medial_axis import _skeleton


__all__ = ["medial_axis", "skeleton"]


def medial_axis(img):
    ma = img.copy()
    ma = _compute_thin_image(ma)
    return ma


def skeleton(subdomain, img):
    ma = img.copy()
    _skeleton(subdomain, ma)

    return ma
