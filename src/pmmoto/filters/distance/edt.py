import numpy as np
from pmmoto.core import voxels
from pmmoto.core import _voxels
from . import _distance

__all__ = ["edt", "edt2d", "edt3d"]


def edt(img, subdomain=None):
    """
    Calculate the exact Eulicidean transform of an image

    Args:
        subdomain (_type_): _description_
        img (numpy array): _description_
    """

    if subdomain is not None:
        if subdomain.domain.periodic or subdomain.num_subdomains > 1:
            img_out = np.copy(img).astype(np.float32)
            boundary_voxels = voxels.get_boundary_voxels(subdomain=subdomain, img=img)
            img_out = _distance.get_initial_envelope(
                img, img_out, dimension=2, boundary_voxels=boundary_voxels
            )

    else:  # Simply perform the edt with no corrections
        if len(img.shape) == 3:
            img_out = edt3d(img)
        if len(img.shape) == 2:
            img_out = edt2d(img)
    return img_out


def edt2d(img, periodic=False):
    """
    Perform an exact Euclidean transform on a image
    For the first pass, collect all the solids (or transitions on all faces)
    The direction needs to be completed first. Then the off-direction need to be correct.
    C
    """
    img_out = np.copy(img).astype(np.float32)

    if periodic:
        correctors = _distance.get_initial_envelope_correctors_2d(img, dimension=0)
    else:
        correctors = None

    img_out = _distance.get_initial_envelope_2d(
        img, img_out, dimension=0, boundary_voxels=correctors
    )

    if periodic:
        boundary_vertices, boundary_f = _distance.get_boundary_hull_2d(
            img_out, dimension=1
        )
    else:
        boundary_vertices = None
        boundary_f = None

    _distance.get_parabolic_envelope_2d(
        img_out, dimension=1, boundary_vertices=boundary_vertices, boundary_f=boundary_f
    )

    return np.asarray(np.sqrt(img_out))


def edt3d(img, periodic=False):
    """
    Perform an exact Euclidean transform on a image
    For the first pass, collect all the solids (or transitions on all faces)
    The direction needs to be completed first. Then the off-direction need to be correct.
    C
    """
    img_out = np.copy(img).astype(np.float32)

    if periodic:
        correctors = _distance.get_initial_envelope_correctors(img=img, dimension=2)
    else:
        correctors = None

    img_out = _distance.get_initial_envelope(
        img, img_out, dimension=2, boundary_voxels=correctors
    )

    for dimension in [1, 0]:
        if periodic:
            boundary_vertices, boundary_f = _distance.get_boundary_hull(
                img_out, dimension=dimension
            )
        else:
            boundary_vertices = None
            boundary_f = None

        _distance.get_parabolic_envelope(
            img_out,
            dimension=dimension,
            boundary_vertices=boundary_vertices,
            boundary_f=boundary_f,
        )

    return np.asarray(np.sqrt(img_out))
