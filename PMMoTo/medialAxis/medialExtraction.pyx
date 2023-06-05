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


@cython.boundscheck(False)
@cython.wraparound(False)
def getInternalBoundaries(mEFunc.pixel_type[:, :, ::1] img not None, int fIndex):

    cdef:
        npy_intp x, y, z, ID
        bint no_change

        # list simple_border_points
        vector[mEFunc.coordinate] simple_border_points
        mEFunc.coordinate point
        Py_ssize_t num_border_points, i, j
        mEFunc.pixel_type neighb[27]

    if fIndex == 0:
        mEFunc.find_simple_point_candidates_internalfaces_0(img,simple_border_points)
    if fIndex == 1:
        mEFunc.find_simple_point_candidates_internalfaces_1(img,simple_border_points)
    if fIndex == 2:
        mEFunc.find_simple_point_candidates_internalfaces_2(img,simple_border_points)
    if fIndex == 3:
        mEFunc.find_simple_point_candidates_internalfaces_3(img,simple_border_points)
    if fIndex == 4:
        mEFunc.find_simple_point_candidates_internalfaces_4(img,simple_border_points)
    if fIndex == 5:
        mEFunc.find_simple_point_candidates_internalfaces_5(img,simple_border_points)

    num_border_points = simple_border_points.size()
    simple_border_points = sorted(simple_border_points, key=lambda d: d['faceCount'],reverse=True)
    for i in range(num_border_points):
        point = simple_border_points[i]
        x = point.x
        y = point.y
        z = point.z
        ID = point.ID

        mEFunc.get_neighborhood(img, x, y, z, neighb)
        if mEFunc.is_simple_point(neighb):
            img[x, y, z] = 0
            no_change = False


    return np.ascontiguousarray(img)




@cython.boundscheck(False)
@cython.wraparound(False)
def _compute_thin_image_border(mEFunc.pixel_type[:, :, ::1] img not None, int curr_border, int unchanged_borders):

    cdef:
        int borders[6]
        npy_intp x, y, z, ID
        bint no_change

        # list simple_border_points
        vector[mEFunc.coordinate] simple_border_points
        vector[mEFunc.coordinateInfo] borderChanges
        mEFunc.coordinate point
        mEFunc.coordinateInfo pointB
        Py_ssize_t num_border_points, i, j
        mEFunc.pixel_type neighb[27]


    mEFunc.find_simple_point_candidates_boundary(img, curr_border, simple_border_points)
    mEFunc.find_simple_point_candidates(img, curr_border, simple_border_points)
    # sequential re-checking to preserve connectivity when deleting
    # in a parallel way
    no_change = True
    num_border_points = simple_border_points.size()
    simple_border_points = sorted(simple_border_points, key=lambda d: d['faceCount'],reverse=True)
    for i in range(num_border_points):
        point = simple_border_points[i]
        x = point.x
        y = point.y
        z = point.z
        ID = point.ID

        if ID == 0:
            mEFunc.get_neighborhood(img, x, y, z, neighb)
            if mEFunc.is_simple_point(neighb):
                img[x, y, z] = 0
                no_change = False

        elif ID > 0:
            mEFunc.get_neighborhood_limited(img, x, y, z, ID, neighb)

            x = point.x
            y = point.y
            z = point.z
            pointB.ID = ID
            pointB.faceCount = point.faceCount

            if mEFunc.is_simple_point(neighb):
                img[x, y, z] = 0
                no_change = False
                pointB.change = 0
                borderChanges.push_back(pointB)
            else:
                pointB.change = 1
                borderChanges.push_back(pointB)

    if no_change:
        unchanged_borders += 1

    return np.ascontiguousarray(img),unchanged_borders





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
        npy_intp x, y, z, ID
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
                x = point.x
                y = point.y
                z = point.z
                ID = point.ID
                if ID == 0:
                    mEFunc.get_neighborhood(img, x, y, z, neighb)
                elif ID > 0:
                    mEFunc.get_neighborhood_limited(img, x, y, z, ID, neighb)
                if mEFunc.is_simple_point(neighb):
                    img[x, y, z] = 0
                    no_change = False

            if no_change:
                unchanged_borders += 1


    return np.ascontiguousarray(img)


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
        npy_intp x, y, z, ID
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
                x = point.x
                y = point.y
                z = point.z
                ID = point.ID
                if ID == 0:
                    mEFunc.get_neighborhood(img, x, y, z, neighb)
                elif ID > 0:
                    mEFunc.get_neighborhood_limited(img, x, y, z, ID, neighb)

                if mEFunc.is_simple_point(neighb) and (not mEFunc.is_surface_point(neighb) and mEFunc.is_endpoint_check(neighb) >= 3):
                    img[x, y, z] = 0
                    no_change = False

            if no_change:
                unchanged_borders += 1


    return np.asarray(img)
