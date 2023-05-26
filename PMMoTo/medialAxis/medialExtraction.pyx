# distutils: language = c++

"""
This is an implementation of the 2D/3D thinning algorithm
of [Lee94]_ of binary images, based on [IAC15]_.

The original Java code [IAC15]_ carries the following message:

 * This work is an implementation by Ignacio Arganda-Carreras of the
 * 3D thinning algorithm from Lee et al. "Building skeleton models via 3-D
 * medial surface/axis thinning algorithms. Computer Vision, Graphics, and
 * Image Processing, 56(6):462-478, 1994." Based on the ITK version from
 * Hanno Homann <a href="http://hdl.handle.net/1926/1292"> http://hdl.handle.net/1926/1292</a>
 * <p>
 *  More information at Skeletonize3D homepage:
 *  https://imagej.net/Skeletonize3D
 *
 * @version 1.0 11/13/2015 (unique BSD licensed version for scikit-image)
 * @author Ignacio Arganda-Carreras (iargandacarreras at gmail.com)

References
----------
.. [Lee94] T.-C. Lee, R.L. Kashyap and C.-N. Chu, Building skeleton models
       via 3-D medial surface/axis thinning algorithms.
       Computer Vision, Graphics, and Image Processing, 56(6):462-478, 1994.

.. [IAC15] Ignacio Arganda-Carreras, 2015. Skeletonize3D plugin for ImageJ(C).
           https://imagej.net/Skeletonize3D

"""
from . cimport medialExtractionFunctions as mEFunc
from libc.string cimport memcpy
from libcpp.vector cimport vector
from libc.stdio cimport printf
from libcpp cimport bool

import numpy as np
from numpy cimport npy_intp, npy_uint8, ndarray
cimport cython

# ctypedef npy_uint8 pixel_type

# # struct to hold 3D coordinates
# cdef struct coordinate:
#     npy_intp p
#     npy_intp r
#     npy_intp c
#     npy_intp ID
#     npy_intp faceCount



@cython.boundscheck(False)
@cython.wraparound(False)
def _compute_thin_image(mEFunc.pixel_type[:, :, ::1] img not None):
    """Compute a thin image.

    Loop through the image multiple times, removing "simple" points, i.e.
    those point which can be removed without changing local connectivity in the
    3x3x3 neighborhood of a point.

    This routine implements the two-pass algorithm of [Lee94]_. Namely,
    for each of the six border types (positive and negative x-, y- and z-),
    the algorithm first collects all possibly deletable points, and then
    performs a sequential rechecking.

    The input, `img`, is assumed to be a 3D binary image in the
    (p, r, c) format [i.e., C ordered array], filled by zeros (background) and
    ones. 

    """
    cdef:
        int unchanged_borders = 0, curr_border, num_borders
        int borders[6]
        npy_intp p, r, c, ID
        bint no_change

        # list simple_border_points
        vector[mEFunc.coordinate] simple_border_points
        mEFunc.coordinate point

        Py_ssize_t num_border_points, i, j

        mEFunc.pixel_type neighb[27]

    # loop over the six directions in this order (for consistency with ImageJ)
    borders[:] = [4,3,2,1,5,6]

    #with nogil:
    # no need to worry about the z direction if the original image is 2D.
    if img.shape[0] == 3:
        num_borders = 4
    else:
        num_borders = 6

    # loop through the image several times until there is no change for all
    # the six border types
    while unchanged_borders < num_borders:
        unchanged_borders = 0
        for j in range(num_borders):

            curr_border = borders[j]
            simple_border_points.clear();
            mEFunc.find_simple_point_candidates_boundary(img, curr_border, simple_border_points)
            mEFunc.find_simple_point_candidates(img, curr_border, simple_border_points)
            # sequential re-checking to preserve connectivity when deleting
            # in a parallel way
            no_change = True
            num_border_points = simple_border_points.size()
            simple_border_points = sorted(simple_border_points, key=lambda d: d['faceCount'],reverse=True)
            for i in range(num_border_points):
                point = simple_border_points[i]
                p = point.p
                r = point.r
                c = point.c
                ID = point.ID
                if ID == 0:
                    mEFunc.get_neighborhood(img, p, r, c, neighb)
                elif ID > 0:
                    mEFunc.get_neighborhood_limited(img, p, r, c, ID, neighb)
                if mEFunc.is_simple_point(neighb):
                    img[p, r, c] = 0
                    no_change = False

            if no_change:
                unchanged_borders += 1


    return np.asarray(img)


@cython.boundscheck(False)
@cython.wraparound(False)
def _compute_thin_image_surface(mEFunc.pixel_type[:, :, ::1] img not None):
    """Compute a thin image.

    Loop through the image multiple times, removing "simple" points, i.e.
    those point which can be removed without changing local connectivity in the
    3x3x3 neighborhood of a point.

    This routine implements the two-pass algorithm of [Lee94]_. Namely,
    for each of the six border types (positive and negative x-, y- and z-),
    the algorithm first collects all possibly deletable points, and then
    performs a sequential rechecking.

    The input, `img`, is assumed to be a 3D binary image in the
    (p, r, c) format [i.e., C ordered array], filled by zeros (background) and
    ones. 

    """
    cdef:
        int unchanged_borders = 0, curr_border, num_borders
        int borders[6]
        npy_intp p, r, c, ID
        bint no_change

        # list simple_border_points
        vector[mEFunc.coordinate] simple_border_points
        mEFunc.coordinate point

        Py_ssize_t num_border_points, i, j

        mEFunc.pixel_type neighb[27]

    # loop over the six directions in this order (for consistency with ImageJ)
    borders[:] = [4,3,2,1,5,6]

    #with nogil:
    # no need to worry about the z direction if the original image is 2D.
    if img.shape[0] == 3:
        num_borders = 4
    else:
        num_borders = 6

    # loop through the image several times until there is no change for all
    # the six border types
    while unchanged_borders < num_borders:
        unchanged_borders = 0
        for j in range(num_borders):
            curr_border = borders[j]
            simple_border_points.clear();
            mEFunc.find_simple_point_candidates_boundary(img, curr_border, simple_border_points)
            mEFunc.find_simple_point_candidates(img, curr_border, simple_border_points)

            #find_simple_point_candidates_boundary(img, curr_border, simple_border_points)
            # sequential re-checking to preserve connectivity when deleting
            # in a parallel way
            no_change = True
            num_border_points = simple_border_points.size()
            #simple_border_points = sorted(simple_border_points, key=lambda d: d['faceCount'],reverse=True)
            for i in range(num_border_points):
                point = simple_border_points[i]
                p = point.p
                r = point.r
                c = point.c
                ID = point.ID
                if ID == 0:
                    mEFunc.get_neighborhood(img, p, r, c, neighb)
                elif ID > 0:
                    mEFunc.get_neighborhood_limited(img, p, r, c, ID, neighb)

                if mEFunc.is_simple_point(neighb) and (not mEFunc.is_surface_point(neighb) and mEFunc.is_endpoint_check(neighb) >= 3):
                    img[p, r, c] = 0
                    no_change = False

            if no_change:
                unchanged_borders += 1


    return np.asarray(img)
