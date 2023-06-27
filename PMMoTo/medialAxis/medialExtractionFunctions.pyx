# distutils: language = c++
# cython: boundscheck=False, wraparound=False, initializedcheck=False, cdivision=True

from libc.string cimport memcpy
from libcpp.vector cimport vector
from libc.stdio cimport printf
from libcpp cimport bool

import numpy as np
from numpy cimport npy_intp, npy_int8, npy_uint8, ndarray, npy_float32
cimport cython


###### look-up tables
def fill_Euler_LUT():
    """ Look-up table for preserving Euler characteristic.

    This is column $\delta G_{26}$ of Table 2 of [Lee94]_.
    """
    cdef int arr[128]
    arr[:] = [1, -1, -1, 1, -3, -1, -1, 1, -1, 1, 1, -1, 3, 1, 1, -1, -3, -1,
                 3, 1, 1, -1, 3, 1, -1, 1, 1, -1, 3, 1, 1, -1, -3, 3, -1, 1, 1,
                 3, -1, 1, -1, 1, 1, -1, 3, 1, 1, -1, 1, 3, 3, 1, 5, 3, 3, 1,
                 -1, 1, 1, -1, 3, 1, 1, -1, -7, -1, -1, 1, -3, -1, -1, 1, -1,
                 1, 1, -1, 3, 1, 1, -1, -3, -1, 3, 1, 1, -1, 3, 1, -1, 1, 1,
                 -1, 3, 1, 1, -1, -3, 3, -1, 1, 1, 3, -1, 1, -1, 1, 1, -1, 3,
                 1, 1, -1, 1, 3, 3, 1, 5, 3, 3, 1, -1, 1, 1, -1, 3, 1, 1, -1]
    
    cdef ndarray LUT = np.zeros(256, dtype=np.intc)
    LUT[1::2] = arr
    return LUT

cdef int[::1] LUT = fill_Euler_LUT()








cdef void find_simple_point_candidates(pixel_type[:, :, ::1] img,
                                       int curr_border,
                                       vector[coordinate] & simple_border_points) nogil:
    """Inner loop of compute_thin_image.

    The algorithm of [Lee94]_ proceeds in two steps: (1) six directions are
    checked for simple border points to remove, and (2) these candidates are
    sequentially rechecked, see Sec 3 of [Lee94]_ for rationale and discussion.

    This routine implements the first step above: it loops over the image
    for a given direction and assembles candidates for removal.

    """
    cdef:
        cdef coordinate point

        pixel_type neighborhood[27]
        npy_intp x, y, z, ID
        bint is_border_pt
        int[::1] Euler_LUT = LUT

    # loop through the image
    for x in range(1, img.shape[0] - 1):
        for y in range(1, img.shape[1] - 1):
            for z in range(1, img.shape[2] - 1):

                # check if pixel is foreground
                if img[x, y, z] != 1:
                    continue

                is_border_pt = (curr_border == 0 and img[x-1,y  ,z  ] == 0 or  #W
                                curr_border == 1 and img[x+1,y  ,z  ] == 0 or  #E
                                curr_border == 2 and img[x  ,y-1,z  ] == 0 or  #S
                                curr_border == 3 and img[x  ,y+1,z  ] == 0 or  #N
                                curr_border == 4 and img[x  ,y  ,z-1] == 0 or  #B
                                curr_border == 5 and img[x  ,y  ,z+1] == 0     #U
                                )
                if not is_border_pt:
                    continue

                get_neighborhood(img, x, y, z, neighborhood)

                #   Check conditions 2 and 3
                if (is_endpoint(neighborhood) or
                    not is_Euler_invariant(neighborhood, Euler_LUT) or
                    not is_simple_point(neighborhood)):
                    continue

                point.x = x
                point.y = y
                point.z = z
                point.ID = 0
                point.faceCount = is_endpoint_check(neighborhood)
                simple_border_points.push_back(point)


cdef bint is_Euler_invariant(pixel_type neighbors[],
                             int[::1] lut) nogil:
    """Check if a point is Euler invariant.

    Calculate Euler characteristic for each octant and sum up.

    Parameters
    ----------
    neighbors
        neighbors of a point
    lut
        The look-up table for preserving the Euler characteristic.

    Returns
    -------
    bool (C bool, that is)

    """
    cdef int n, euler_char = 0

    # octant 0:
    n = 1
    if neighbors[0] == 1:
        n |= 128

    if neighbors[9] == 1:
        n |= 64

    if neighbors[3] == 1:
        n |= 32

    if neighbors[12] == 1:
        n |= 16

    if neighbors[1] == 1:
        n |= 8

    if neighbors[10] == 1:
        n |= 4

    if neighbors[4] == 1:
        n |= 2

    euler_char += lut[n]

    # octant 1:
    n = 1
    if neighbors[2] == 1:
        n |= 128

    if neighbors[1] == 1:
        n |= 64

    if neighbors[11] == 1:
        n |= 32

    if neighbors[10] == 1:
        n |= 16

    if neighbors[5] == 1:
        n |= 8

    if neighbors[4] == 1:
        n |= 4

    if neighbors[14] == 1:
        n |= 2

    euler_char += lut[n]

    # octant 2:
    n = 1
    if neighbors[6] == 1:
        n |= 128

    if neighbors[15] == 1:
        n |= 64

    if neighbors[7] == 1:
        n |= 32

    if neighbors[16] == 1:
        n |= 16

    if neighbors[3] == 1:
        n |= 8

    if neighbors[12] == 1:
        n |= 4

    if neighbors[4] == 1:
        n |= 2

    euler_char += lut[n]

    # octant 3:
    n = 1
    if neighbors[8] == 1:
        n |= 128

    if neighbors[7] == 1:
        n |= 64

    if neighbors[17] == 1:
        n |= 32

    if neighbors[16] == 1:
        n |= 16

    if neighbors[5] == 1:
        n |= 8

    if neighbors[4] == 1:
        n |= 4

    if neighbors[14] == 1:
        n |= 2

    euler_char += lut[n]

    # octant 4:
    n = 1
    if neighbors[18] == 1:
        n |= 128

    if neighbors[21] == 1:
        n |= 64

    if neighbors[9] == 1:
        n |= 32

    if neighbors[12] == 1:
        n |= 16

    if neighbors[19] == 1:
        n |= 8

    if neighbors[22] == 1:
        n |= 4

    if neighbors[10] == 1:
        n |= 2

    euler_char += lut[n]

    # octant 5:
    n = 1
    if neighbors[20] == 1:
        n |= 128

    if neighbors[23] == 1:
        n |= 64

    if neighbors[19] == 1:
        n |= 32

    if neighbors[22] == 1:
        n |= 16

    if neighbors[11] == 1:
        n |= 8

    if neighbors[14] == 1:
        n |= 4

    if neighbors[10] == 1:
        n |= 2

    euler_char += lut[n]

    # octant 6:
    n = 1
    if neighbors[24] == 1:
        n |= 128

    if neighbors[25] == 1:
        n |= 64

    if neighbors[15] == 1:
        n |= 32

    if neighbors[16] == 1:
        n |= 16

    if neighbors[21] == 1:
        n |= 8

    if neighbors[22] == 1:
        n |= 4

    if neighbors[12] == 1:
        n |= 2

    euler_char += lut[n]

    # octant 7:
    n = 1
    if neighbors[26] == 1:
        n |= 128

    if neighbors[23] == 1:
        n |= 64

    if neighbors[17] == 1:
        n |= 32

    if neighbors[14] == 1:
        n |= 16

    if neighbors[25] == 1:
        n |= 8

    if neighbors[22] == 1:
        n |= 4

    if neighbors[16] == 1:
        n |= 2

    euler_char += lut[n]
    return euler_char == 0


cdef inline bint is_endpoint(pixel_type neighbors[]) nogil:
    """An endpoint has exactly one neighbor in the 26-neighborhood.
    """
    # The center pixel is counted, thus r.h.s. is 2
    cdef int s = 0, j
    for j in range(27):
        s += neighbors[j]
    return s == 2

cdef inline int is_endpoint_check(pixel_type neighbors[]) nogil:
    """An endpoint has exactly one neighbor in the 26-neighborhood.
    """
    # The center pixel is counted, thus r.h.s. is 2
    cdef int s = 0, j
    for j in range(27):
        s += neighbors[j]
    return s


cdef bint is_surface_point(pixel_type neighbors[]) nogil:
    """Check is a point is a Surface Point.

    See Equations 8 and 9 in [Lee94]

    Parameters
    ----------
    neighbors : uint8 C array, shape(27,)
        neighbors of the point

    Returns
    -------
    bool
        Whether the point is surface point or not.

    """

    # Loop through all octants
    cdef int n,neighSum
    cdef bint surfacePoint = True, cond1 = False

    n = 1
    neighSum = 0
    surfaceOctant = False
    if neighbors[0] == 1:
        n |= 128
        neighSum = neighSum + 1
    if neighbors[9] == 1:
        n |= 64
        neighSum = neighSum + 1
    if neighbors[3] == 1:
        n |= 32
        neighSum = neighSum + 1
    if neighbors[12] == 1:
        n |= 16
        neighSum = neighSum + 1
    if neighbors[1] == 1:
        n |= 8
        neighSum = neighSum + 1
    if neighbors[10] == 1:
        n |= 4
        neighSum = neighSum + 1
    if neighbors[4] == 1:
        n |= 2
        neighSum = neighSum + 1

    if (n == 240 or n == 165 or n == 170 or n == 204):
        cond1 = True 
    
    if (not cond1 and neighSum > 3):
        surfacePoint = False


    n = 1
    neighSum = 0
    surfaceOctant = False
    if neighbors[2] == 1:
        n |= 128
        neighSum = neighSum + 1
    if neighbors[1] == 1:
        n |= 64
        neighSum = neighSum + 1
    if neighbors[11] == 1:
        n |= 32
        neighSum = neighSum + 1
    if neighbors[10] == 1:
        n |= 16
        neighSum = neighSum + 1
    if neighbors[5] == 1:
        n |= 8
        neighSum = neighSum + 1
    if neighbors[4] == 1:
        n |= 4
        neighSum = neighSum + 1
    if neighbors[14] == 1:
        n |= 2
        neighSum = neighSum + 1

    if (n == 240 or n == 165 or n == 170 or n == 204):
        cond1 = True 
    
    if (not cond1 and neighSum > 3):
        surfacePoint = False


    n = 1
    neighSum = 0
    surfaceOctant = False
    if neighbors[6] == 1:
        n |= 128
        neighSum = neighSum + 1
    if neighbors[15] == 1:
        n |= 64
        neighSum = neighSum + 1
    if neighbors[7] == 1:
        n |= 32
        neighSum = neighSum + 1
    if neighbors[16] == 1:
        n |= 16
        neighSum = neighSum + 1
    if neighbors[3] == 1:
        n |= 8
        neighSum = neighSum + 1
    if neighbors[12] == 1:
        n |= 4
        neighSum = neighSum + 1
    if neighbors[4] == 1:
        n |= 2
        neighSum = neighSum + 1

    if (n == 240 or n == 165 or n == 170 or n == 204):
        cond1 = True 
    
    if (not cond1 and neighSum > 3):
        surfacePoint = False


    n = 1
    neighSum = 0
    surfaceOctant = False
    if neighbors[8] == 1:
        n |= 128
        neighSum = neighSum + 1
    if neighbors[7] == 1:
        n |= 64
        neighSum = neighSum + 1
    if neighbors[17] == 1:
        n |= 32
        neighSum = neighSum + 1
    if neighbors[16] == 1:
        n |= 16
        neighSum = neighSum + 1
    if neighbors[5] == 1:
        n |= 8
        neighSum = neighSum + 1
    if neighbors[4] == 1:
        n |= 4
        neighSum = neighSum + 1
    if neighbors[14] == 1:
        n |= 2
        neighSum = neighSum + 1

    if (n == 240 or n == 165 or n == 170 or n == 204):
        cond1 = True 
    
    if (not cond1 and neighSum > 3):
        surfacePoint = False


    n = 1
    neighSum = 0
    surfaceOctant = False
    if neighbors[18] == 1:
        n |= 128
        neighSum = neighSum + 1
    if neighbors[21] == 1:
        n |= 64
        neighSum = neighSum + 1
    if neighbors[9] == 1:
        n |= 32
        neighSum = neighSum + 1
    if neighbors[12] == 1:
        n |= 16
        neighSum = neighSum + 1
    if neighbors[19] == 1:
        n |= 8
        neighSum = neighSum + 1
    if neighbors[22] == 1:
        n |= 4
        neighSum = neighSum + 1
    if neighbors[10] == 1:
        n |= 2
        neighSum = neighSum + 1

    if (n == 240 or n == 165 or n == 170 or n == 204):
        cond1 = True 
    
    if (not cond1 and neighSum > 3):
        surfacePoint = False


    n = 1
    neighSum = 0
    surfaceOctant = False
    if neighbors[20] == 1:
        n |= 128
        neighSum = neighSum + 1
    if neighbors[23] == 1:
        n |= 64
        neighSum = neighSum + 1
    if neighbors[19] == 1:
        n |= 32
        neighSum = neighSum + 1
    if neighbors[22] == 1:
        n |= 16
        neighSum = neighSum + 1
    if neighbors[11] == 1:
        n |= 8
        neighSum = neighSum + 1
    if neighbors[14] == 1:
        n |= 4
        neighSum = neighSum + 1
    if neighbors[10] == 1:
        n |= 2
        neighSum = neighSum + 1

    if (n == 240 or n == 165 or n == 170 or n == 204):
        cond1 = True 
    
    if (not cond1 and neighSum > 3):
        surfacePoint = False


    n = 1
    neighSum = 0
    surfaceOctant = False
    if neighbors[24] == 1:
        n |= 128
        neighSum = neighSum + 1
    if neighbors[25] == 1:
        n |= 64
        neighSum = neighSum + 1
    if neighbors[15] == 1:
        n |= 32
        neighSum = neighSum + 1
    if neighbors[16] == 1:
        n |= 16
        neighSum = neighSum + 1
    if neighbors[21] == 1:
        n |= 8
        neighSum = neighSum + 1
    if neighbors[22] == 1:
        n |= 4
        neighSum = neighSum + 1
    if neighbors[12] == 1:
        n |= 2
        neighSum = neighSum + 1

    if (n == 240 or n == 165 or n == 170 or n == 204):
        cond1 = True 
    
    if (not cond1 and neighSum > 3):
        surfacePoint = False


    n = 1
    neighSum = 0
    surfaceOctant = False
    if neighbors[26] == 1:
        n |= 128
        neighSum = neighSum + 1
    if neighbors[23] == 1:
        n |= 64
        neighSum = neighSum + 1
    if neighbors[17] == 1:
        n |= 32
        neighSum = neighSum + 1
    if neighbors[14] == 1:
        n |= 16
        neighSum = neighSum + 1
    if neighbors[25] == 1:
        n |= 8
        neighSum = neighSum + 1
    if neighbors[22] == 1:
        n |= 4
        neighSum = neighSum + 1
    if neighbors[16] == 1:
        n |= 2
        neighSum = neighSum + 1

    if (n == 240 or n == 165 or n == 170 or n == 204):
        cond1 = True 
    
    if (not cond1 and neighSum > 3):
        surfacePoint = False


    return surfacePoint

cdef bint is_simple_point(pixel_type neighbors[]) nogil:
    """Check is a point is a Simple Point.

    A point is simple iff its deletion does not change connectivity in
    the 3x3x3 neighborhood. (cf conditions 2 and 3 in [Lee94]_).

    This method is named "N(v)_labeling" in [Lee94]_.

    Parameters
    ----------
    neighbors : uint8 C array, shape(27,)
        neighbors of the point

    Returns
    -------
    bool
        Whether the point is simple or not.

    """
    # ignore center pixel (i=13) when counting (see [Lee94]_)
    cdef pixel_type cube[26]
    memcpy(cube, neighbors, 13*sizeof(pixel_type))
    memcpy(cube+13, neighbors+14, 13*sizeof(pixel_type))

    # set initial label
    cdef int label = 2, i

    # for all point in the neighborhood
    for i in range(26):
        if cube[i] == 1:
            # voxel has not been labeled yet
            # start recursion with any octant that contains the point i
            if i in (0, 1, 3, 4, 9, 10, 12):
                octree_labeling(1, label, cube)
            elif i in (2, 5, 11, 13):
                octree_labeling(2, label, cube)
            elif i in (6, 7, 14, 15):
                octree_labeling(3, label, cube)
            elif i in (8, 16):
                octree_labeling(4, label, cube)
            elif i in (17, 18, 20, 21):
                octree_labeling(5, label, cube)
            elif i in (19, 22):
                octree_labeling(6, label, cube)
            elif i in (23, 24):
                octree_labeling(7, label, cube)
            elif i == 25:
                octree_labeling(8, label, cube)
            label += 1
            if label - 2 >= 2:
                return False
    return True

cdef void octree_labeling(int octant, int label, pixel_type cube[]) nogil:
    """This is a recursive method that calculates the number of connected
    components in the 3D neighborhood after the center pixel would
    have been removed.

    See Figs. 6 and 7 of [Lee94]_ for the values of indices.

    Parameters
    ----------
    octant : int
        octant index
    label : int
        the current label of the center point
    cube : uint8 C array, shape(26,)
        local neighborhood of the point

    """
    # This routine checks if there are points in the octant with value 1
    # Then sets points in this octant to current label
    # and recursive labeling of adjacent octants.
    #
    # Below, leading underscore means build-time variables.

    if octant == 1:
        if cube[0] == 1:
            cube[0] = label
        if cube[1] == 1:
            cube[1] = label
            octree_labeling(2, label, cube)
        if cube[3] == 1:
            cube[3] = label
            octree_labeling(3, label, cube)
        if cube[4] == 1:
            cube[4] = label
            octree_labeling(2, label, cube)
            octree_labeling(3, label, cube)
            octree_labeling(4, label, cube)
        if cube[9] == 1:
            cube[9] = label
            octree_labeling(5, label, cube)
        if cube[10] == 1:
            cube[10] = label
            octree_labeling(2, label, cube)
            octree_labeling(5, label, cube)
            octree_labeling(6, label, cube)
        if cube[12] == 1:
            cube[12] = label
            octree_labeling(3, label, cube)
            octree_labeling(5, label, cube)
            octree_labeling(7, label, cube)

    if octant == 2:
        if cube[1] == 1:
            cube[1] = label
            octree_labeling(1, label, cube)
        if cube[4] == 1:
            cube[4] = label
            octree_labeling(1, label, cube)
            octree_labeling(3, label, cube)
            octree_labeling(4, label, cube)
        if cube[10] == 1:
            cube[10] = label
            octree_labeling(1, label, cube)
            octree_labeling(5, label, cube)
            octree_labeling(6, label, cube)
        if cube[2] == 1:
            cube[2] = label
        if cube[5] == 1:
            cube[5] = label
            octree_labeling(4, label, cube)
        if cube[11] == 1:
            cube[11] = label
            octree_labeling(6, label, cube)
        if cube[13] == 1:
            cube[13] = label
            octree_labeling(4, label, cube)
            octree_labeling(6, label, cube)
            octree_labeling(8, label, cube)

    if octant == 3:
        if cube[3] == 1:
            cube[3] = label
            octree_labeling(1, label, cube)
        if cube[4] == 1:
            cube[4] = label
            octree_labeling(1, label, cube)
            octree_labeling(2, label, cube)
            octree_labeling(4, label, cube)
        if cube[12] == 1:
            cube[12] = label
            octree_labeling(1, label, cube)
            octree_labeling(5, label, cube)
            octree_labeling(7, label, cube)
        if cube[6] == 1:
            cube[6] = label
        if cube[7] == 1:
            cube[7] = label
            octree_labeling(4, label, cube)
        if cube[14] == 1:
            cube[14] = label
            octree_labeling(7, label, cube)
        if cube[15] == 1:
            cube[15] = label
            octree_labeling(4, label, cube)
            octree_labeling(7, label, cube)
            octree_labeling(8, label, cube)

    if octant == 4:
        if cube[4] == 1:
            cube[4] = label
            octree_labeling(1, label, cube)
            octree_labeling(2, label, cube)
            octree_labeling(3, label, cube)
        if cube[5] == 1:
            cube[5] = label
            octree_labeling(2, label, cube)
        if cube[13] == 1:
            cube[13] = label
            octree_labeling(2, label, cube)
            octree_labeling(6, label, cube)
            octree_labeling(8, label, cube)
        if cube[7] == 1:
            cube[7] = label
            octree_labeling(3, label, cube)
        if cube[15] == 1:
            cube[15] = label
            octree_labeling(3, label, cube)
            octree_labeling(7, label, cube)
            octree_labeling(8, label, cube)
        if cube[8] == 1:
            cube[8] = label
        if cube[16] == 1:
            cube[16] = label
            octree_labeling(8, label, cube)

    if octant == 5:
        if cube[9] == 1:
            cube[9] = label
            octree_labeling(1, label, cube)
        if cube[10] == 1:
            cube[10] = label
            octree_labeling(1, label, cube)
            octree_labeling(2, label, cube)
            octree_labeling(6, label, cube)
        if cube[12] == 1:
            cube[12] = label
            octree_labeling(1, label, cube)
            octree_labeling(3, label, cube)
            octree_labeling(7, label, cube)
        if cube[17] == 1:
            cube[17] = label
        if cube[18] == 1:
            cube[18] = label
            octree_labeling(6, label, cube)
        if cube[20] == 1:
            cube[20] = label
            octree_labeling(7, label, cube)
        if cube[21] == 1:
            cube[21] = label
            octree_labeling(6, label, cube)
            octree_labeling(7, label, cube)
            octree_labeling(8, label, cube)

    if octant == 6:
        if cube[10] == 1:
            cube[10] = label
            octree_labeling(1, label, cube)
            octree_labeling(2, label, cube)
            octree_labeling(5, label, cube)
        if cube[11] == 1:
            cube[11] = label
            octree_labeling(2, label, cube)
        if cube[13] == 1:
            cube[13] = label
            octree_labeling(2, label, cube)
            octree_labeling(4, label, cube)
            octree_labeling(8, label, cube)
        if cube[18] == 1:
            cube[18] = label
            octree_labeling(5, label, cube)
        if cube[21] == 1:
            cube[21] = label
            octree_labeling(5, label, cube)
            octree_labeling(7, label, cube)
            octree_labeling(8, label, cube)
        if cube[19] == 1:
            cube[19] = label
        if cube[22] == 1:
            cube[22] = label
            octree_labeling(8, label, cube)

    if octant == 7:
        if cube[12] == 1:
            cube[12] = label
            octree_labeling(1, label, cube)
            octree_labeling(3, label, cube)
            octree_labeling(5, label, cube)
        if cube[14] == 1:
            cube[14] = label
            octree_labeling(3, label, cube)
        if cube[15] == 1:
            cube[15] = label
            octree_labeling(3, label, cube)
            octree_labeling(4, label, cube)
            octree_labeling(8, label, cube)
        if cube[20] == 1:
            cube[20] = label
            octree_labeling(5, label, cube)
        if cube[21] == 1:
            cube[21] = label
            octree_labeling(5, label, cube)
            octree_labeling(6, label, cube)
            octree_labeling(8, label, cube)
        if cube[23] == 1:
            cube[23] = label
        if cube[24] == 1:
            cube[24] = label
            octree_labeling(8, label, cube)

    if octant == 8:
        if cube[13] == 1:
            cube[13] = label
            octree_labeling(2, label, cube)
            octree_labeling(4, label, cube)
            octree_labeling(6, label, cube)
        if cube[15] == 1:
            cube[15] = label
            octree_labeling(3, label, cube)
            octree_labeling(4, label, cube)
            octree_labeling(7, label, cube)
        if cube[16] == 1:
            cube[16] = label
            octree_labeling(4, label, cube)
        if cube[21] == 1:
            cube[21] = label
            octree_labeling(5, label, cube)
            octree_labeling(6, label, cube)
            octree_labeling(7, label, cube)
        if cube[22] == 1:
            cube[22] = label
            octree_labeling(6, label, cube)
        if cube[24] == 1:
            cube[24] = label
            octree_labeling(7, label, cube)
        if cube[25] == 1:
            cube[25] = label

cdef void get_neighborhood(pixel_type[:, :, ::1] img,
                           npy_intp x, npy_intp y, npy_intp z,
                           pixel_type neighborhood[]) nogil:
    """Get the neighborhood of a pixel.
    Consistent with ImageJ and Skimage Ordering 
    """
    neighborhood[ 0] = img[x-1, y+1, z-1]
    neighborhood[ 1] = img[x  , y+1, z-1]
    neighborhood[ 2] = img[x+1, y+1, z-1]

    neighborhood[ 3] = img[x-1, y  , z-1]
    neighborhood[ 4] = img[x  , y  , z-1]
    neighborhood[ 5] = img[x+1, y  , z-1]

    neighborhood[ 6] = img[x-1, y-1, z-1]
    neighborhood[ 7] = img[x  , y-1, z-1]
    neighborhood[ 8] = img[x+1, y-1, z-1]

    neighborhood[ 9] = img[x-1, y+1, z]
    neighborhood[10] = img[x  , y+1, z]
    neighborhood[11] = img[x+1, y+1, z]

    neighborhood[12] = img[x-1, y  , z]
    neighborhood[13] = img[x  , y  , z]
    neighborhood[14] = img[x+1, y  , z]

    neighborhood[15] = img[x-1, y-1, z]
    neighborhood[16] = img[x  , y-1, z]
    neighborhood[17] = img[x+1, y-1, z]

    neighborhood[18] = img[x-1, y+1, z+1]
    neighborhood[19] = img[x  , y+1, z+1]
    neighborhood[20] = img[x+1, y+1, z+1]

    neighborhood[21] = img[x-1, y  , z+1]
    neighborhood[22] = img[x ,  y  , z+1]
    neighborhood[23] = img[x+1, y  , z+1]

    neighborhood[24] = img[x-1, y-1, z+1]
    neighborhood[25] = img[x  , y-1, z+1]
    neighborhood[26] = img[x+1, y-1, z+1]

cdef void get_neighborhood_limited(pixel_type[:, :, ::1] img,
                               npy_intp x, npy_intp y, npy_intp z, npy_intp ID,
                               pixel_type neighborhood[]) nogil:

    cdef int lID
    if ID < 20:
        lID = ID - 10
        if lID == 0:
            get_neighborhood_boundary_faces_0(img,x,y,z,neighborhood)
        if lID == 1:
            get_neighborhood_boundary_faces_1(img,x,y,z,neighborhood)
        if lID == 2:
            get_neighborhood_boundary_faces_2(img,x,y,z,neighborhood)
        if lID == 3:
            get_neighborhood_boundary_faces_3(img,x,y,z,neighborhood)
        if lID == 4:
            get_neighborhood_boundary_faces_4(img,x,y,z,neighborhood)
        if lID == 5:
            get_neighborhood_boundary_faces_5(img,x,y,z,neighborhood)
    elif ID < 40:
        lID = ID - 20
        if lID == 0:
            get_neighborhood_boundary_edges_0(img,x,y,z,neighborhood)
        if lID == 1:
            get_neighborhood_boundary_edges_1(img,x,y,z,neighborhood)
        if lID == 2:
            get_neighborhood_boundary_edges_2(img,x,y,z,neighborhood)
        if lID == 3:
            get_neighborhood_boundary_edges_3(img,x,y,z,neighborhood)
        if lID == 4:
            get_neighborhood_boundary_edges_4(img,x,y,z,neighborhood)
        if lID == 5:
            get_neighborhood_boundary_edges_5(img,x,y,z,neighborhood)
        if lID == 6:
            get_neighborhood_boundary_edges_6(img,x,y,z,neighborhood)
        if lID == 7:
            get_neighborhood_boundary_edges_7(img,x,y,z,neighborhood)
        if lID == 8:
            get_neighborhood_boundary_edges_8(img,x,y,z,neighborhood)
        if lID == 9:
            get_neighborhood_boundary_edges_9(img,x,y,z,neighborhood)
        if lID == 10:
            get_neighborhood_boundary_edges_10(img,x,y,z,neighborhood)
        if lID == 11:
            get_neighborhood_boundary_edges_11(img,x,y,z,neighborhood)
    else:
        lID = ID - 40
        if lID == 0:
            get_neighborhood_boundary_corners_0(img,x,y,z,neighborhood)
        if lID == 1:
            get_neighborhood_boundary_corners_1(img,x,y,z,neighborhood)
        if lID == 2:
            get_neighborhood_boundary_corners_2(img,x,y,z,neighborhood)
        if lID == 3:
            get_neighborhood_boundary_corners_3(img,x,y,z,neighborhood)
        if lID == 4:
            get_neighborhood_boundary_corners_4(img,x,y,z,neighborhood)
        if lID == 5:
            get_neighborhood_boundary_corners_5(img,x,y,z,neighborhood)
        if lID == 6:
            get_neighborhood_boundary_corners_6(img,x,y,z,neighborhood)
        if lID == 7:
            get_neighborhood_boundary_corners_7(img,x,y,z,neighborhood)


cdef void get_neighborhood_boundary_faces_0(pixel_type[:, :, ::1] img,
                               npy_intp x, npy_intp y, npy_intp z,
                               pixel_type neighborhood[]) nogil:
    neighborhood[0] = 0
    neighborhood[1] = img[x, y+1, z-1]
    neighborhood[2] = img[x+1, y+1, z-1]
    neighborhood[3] = 0
    neighborhood[4] = img[x, y, z-1]
    neighborhood[5] = img[x+1, y, z-1]
    neighborhood[6] = 0
    neighborhood[7] = img[x, y-1, z-1]
    neighborhood[8] = img[x+1, y-1, z-1]
    neighborhood[9] = 0
    neighborhood[10] = img[x, y+1, z]
    neighborhood[11] = img[x+1, y+1, z]
    neighborhood[12] = 0
    neighborhood[13] = img[x, y, z]
    neighborhood[14] = img[x+1, y, z]
    neighborhood[15] = 0
    neighborhood[16] = img[x, y-1, z]
    neighborhood[17] = img[x+1, y-1, z]
    neighborhood[18] = 0
    neighborhood[19] = img[x, y+1, z+1]
    neighborhood[20] = img[x+1, y+1, z+1]
    neighborhood[21] = 0
    neighborhood[22] = img[x, y, z+1]
    neighborhood[23] = img[x+1, y, z+1]
    neighborhood[24] = 0
    neighborhood[25] = img[x, y-1, z+1]
    neighborhood[26] = img[x+1, y-1, z+1]


cdef void get_neighborhood_boundary_faces_1(pixel_type[:, :, ::1] img,
                               npy_intp x, npy_intp y, npy_intp z,
                               pixel_type neighborhood[]) nogil:
    neighborhood[0] = img[x-1, y+1, z-1]
    neighborhood[1] = img[x, y+1, z-1]
    neighborhood[2] = 0
    neighborhood[3] = img[x-1, y, z-1]
    neighborhood[4] = img[x, y, z-1]
    neighborhood[5] = 0
    neighborhood[6] = img[x-1, y-1, z-1]
    neighborhood[7] = img[x, y-1, z-1]
    neighborhood[8] = 0
    neighborhood[9] = img[x-1, y+1, z]
    neighborhood[10] = img[x, y+1, z]
    neighborhood[11] = 0
    neighborhood[12] = img[x-1, y, z]
    neighborhood[13] = img[x, y, z]
    neighborhood[14] = 0
    neighborhood[15] = img[x-1, y-1, z]
    neighborhood[16] = img[x, y-1, z]
    neighborhood[17] = 0
    neighborhood[18] = img[x-1, y+1, z+1]
    neighborhood[19] = img[x, y+1, z+1]
    neighborhood[20] = 0
    neighborhood[21] = img[x-1, y, z+1]
    neighborhood[22] = img[x, y, z+1]
    neighborhood[23] = 0
    neighborhood[24] = img[x-1, y-1, z+1]
    neighborhood[25] = img[x, y-1, z+1]
    neighborhood[26] = 0


cdef void get_neighborhood_boundary_faces_2(pixel_type[:, :, ::1] img,
                               npy_intp x, npy_intp y, npy_intp z,
                               pixel_type neighborhood[]) nogil:
    neighborhood[0] = img[x-1, y+1, z-1]
    neighborhood[1] = img[x, y+1, z-1]
    neighborhood[2] = img[x+1, y+1, z-1]
    neighborhood[3] = img[x-1, y, z-1]
    neighborhood[4] = img[x, y, z-1]
    neighborhood[5] = img[x+1, y, z-1]
    neighborhood[6] = 0
    neighborhood[7] = 0
    neighborhood[8] = 0
    neighborhood[9] = img[x-1, y+1, z]
    neighborhood[10] = img[x, y+1, z]
    neighborhood[11] = img[x+1, y+1, z]
    neighborhood[12] = img[x-1, y, z]
    neighborhood[13] = img[x, y, z]
    neighborhood[14] = img[x+1, y, z]
    neighborhood[15] = 0
    neighborhood[16] = 0
    neighborhood[17] = 0
    neighborhood[18] = img[x-1, y+1, z+1]
    neighborhood[19] = img[x, y+1, z+1]
    neighborhood[20] = img[x+1, y+1, z+1]
    neighborhood[21] = img[x-1, y, z+1]
    neighborhood[22] = img[x, y, z+1]
    neighborhood[23] = img[x+1, y, z+1]
    neighborhood[24] = 0
    neighborhood[25] = 0
    neighborhood[26] = 0


cdef void get_neighborhood_boundary_faces_3(pixel_type[:, :, ::1] img,
                               npy_intp x, npy_intp y, npy_intp z,
                               pixel_type neighborhood[]) nogil:
    neighborhood[0] = 0
    neighborhood[1] = 0
    neighborhood[2] = 0
    neighborhood[3] = img[x-1, y, z-1]
    neighborhood[4] = img[x, y, z-1]
    neighborhood[5] = img[x+1, y, z-1]
    neighborhood[6] = img[x-1, y-1, z-1]
    neighborhood[7] = img[x, y-1, z-1]
    neighborhood[8] = img[x+1, y-1, z-1]
    neighborhood[9] = 0
    neighborhood[10] = 0
    neighborhood[11] = 0
    neighborhood[12] = img[x-1, y, z]
    neighborhood[13] = img[x, y, z]
    neighborhood[14] = img[x+1, y, z]
    neighborhood[15] = img[x-1, y-1, z]
    neighborhood[16] = img[x, y-1, z]
    neighborhood[17] = img[x+1, y-1, z]
    neighborhood[18] = 0
    neighborhood[19] = 0
    neighborhood[20] = 0
    neighborhood[21] = img[x-1, y, z+1]
    neighborhood[22] = img[x, y, z+1]
    neighborhood[23] = img[x+1, y, z+1]
    neighborhood[24] = img[x-1, y-1, z+1]
    neighborhood[25] = img[x, y-1, z+1]
    neighborhood[26] = img[x+1, y-1, z+1]


cdef void get_neighborhood_boundary_faces_4(pixel_type[:, :, ::1] img,
                               npy_intp x, npy_intp y, npy_intp z,
                               pixel_type neighborhood[]) nogil:
    neighborhood[0] = 0
    neighborhood[1] = 0
    neighborhood[2] = 0
    neighborhood[3] = 0
    neighborhood[4] = 0
    neighborhood[5] = 0
    neighborhood[6] = 0
    neighborhood[7] = 0
    neighborhood[8] = 0
    neighborhood[9] = img[x-1, y+1, z]
    neighborhood[10] = img[x, y+1, z]
    neighborhood[11] = img[x+1, y+1, z]
    neighborhood[12] = img[x-1, y, z]
    neighborhood[13] = img[x, y, z]
    neighborhood[14] = img[x+1, y, z]
    neighborhood[15] = img[x-1, y-1, z]
    neighborhood[16] = img[x, y-1, z]
    neighborhood[17] = img[x+1, y-1, z]
    neighborhood[18] = img[x-1, y+1, z+1]
    neighborhood[19] = img[x, y+1, z+1]
    neighborhood[20] = img[x+1, y+1, z+1]
    neighborhood[21] = img[x-1, y, z+1]
    neighborhood[22] = img[x, y, z+1]
    neighborhood[23] = img[x+1, y, z+1]
    neighborhood[24] = img[x-1, y-1, z+1]
    neighborhood[25] = img[x, y-1, z+1]
    neighborhood[26] = img[x+1, y-1, z+1]


cdef void get_neighborhood_boundary_faces_5(pixel_type[:, :, ::1] img,
                               npy_intp x, npy_intp y, npy_intp z,
                               pixel_type neighborhood[]) nogil:
    neighborhood[0] = img[x-1, y+1, z-1]
    neighborhood[1] = img[x, y+1, z-1]
    neighborhood[2] = img[x+1, y+1, z-1]
    neighborhood[3] = img[x-1, y, z-1]
    neighborhood[4] = img[x, y, z-1]
    neighborhood[5] = img[x+1, y, z-1]
    neighborhood[6] = img[x-1, y-1, z-1]
    neighborhood[7] = img[x, y-1, z-1]
    neighborhood[8] = img[x+1, y-1, z-1]
    neighborhood[9] = img[x-1, y+1, z]
    neighborhood[10] = img[x, y+1, z]
    neighborhood[11] = img[x+1, y+1, z]
    neighborhood[12] = img[x-1, y, z]
    neighborhood[13] = img[x, y, z]
    neighborhood[14] = img[x+1, y, z]
    neighborhood[15] = img[x-1, y-1, z]
    neighborhood[16] = img[x, y-1, z]
    neighborhood[17] = img[x+1, y-1, z]
    neighborhood[18] = 0
    neighborhood[19] = 0
    neighborhood[20] = 0
    neighborhood[21] = 0
    neighborhood[22] = 0
    neighborhood[23] = 0
    neighborhood[24] = 0
    neighborhood[25] = 0
    neighborhood[26] = 0


cdef void get_neighborhood_boundary_edges_0(pixel_type[:, :, ::1] img,
                               npy_intp x, npy_intp y, npy_intp z,
                               pixel_type neighborhood[]) nogil:
    neighborhood[0] = 0
    neighborhood[1] = 0
    neighborhood[2] = 0
    neighborhood[3] = 0
    neighborhood[4] = 0
    neighborhood[5] = 0
    neighborhood[6] = 0
    neighborhood[7] = 0
    neighborhood[8] = 0
    neighborhood[9] = 0
    neighborhood[10] = img[x, y+1, z]
    neighborhood[11] = img[x+1, y+1, z]
    neighborhood[12] = 0
    neighborhood[13] = img[x, y, z]
    neighborhood[14] = img[x+1, y, z]
    neighborhood[15] = 0
    neighborhood[16] = img[x, y-1, z]
    neighborhood[17] = img[x+1, y-1, z]
    neighborhood[18] = 0
    neighborhood[19] = img[x, y+1, z+1]
    neighborhood[20] = img[x+1, y+1, z+1]
    neighborhood[21] = 0
    neighborhood[22] = img[x, y, z+1]
    neighborhood[23] = img[x+1, y, z+1]
    neighborhood[24] = 0
    neighborhood[25] = img[x, y-1, z+1]
    neighborhood[26] = img[x+1, y-1, z+1]


cdef void get_neighborhood_boundary_edges_1(pixel_type[:, :, ::1] img,
                               npy_intp x, npy_intp y, npy_intp z,
                               pixel_type neighborhood[]) nogil:
    neighborhood[0] = 0
    neighborhood[1] = img[x, y+1, z-1]
    neighborhood[2] = img[x+1, y+1, z-1]
    neighborhood[3] = 0
    neighborhood[4] = img[x, y, z-1]
    neighborhood[5] = img[x+1, y, z-1]
    neighborhood[6] = 0
    neighborhood[7] = img[x, y-1, z-1]
    neighborhood[8] = img[x+1, y-1, z-1]
    neighborhood[9] = 0
    neighborhood[10] = img[x, y+1, z]
    neighborhood[11] = img[x+1, y+1, z]
    neighborhood[12] = 0
    neighborhood[13] = img[x, y, z]
    neighborhood[14] = img[x+1, y, z]
    neighborhood[15] = 0
    neighborhood[16] = img[x, y-1, z]
    neighborhood[17] = img[x+1, y-1, z]
    neighborhood[18] = 0
    neighborhood[19] = 0
    neighborhood[20] = 0
    neighborhood[21] = 0
    neighborhood[22] = 0
    neighborhood[23] = 0
    neighborhood[24] = 0
    neighborhood[25] = 0
    neighborhood[26] = 0


cdef void get_neighborhood_boundary_edges_2(pixel_type[:, :, ::1] img,
                               npy_intp x, npy_intp y, npy_intp z,
                               pixel_type neighborhood[]) nogil:
    neighborhood[0] = 0
    neighborhood[1] = img[x, y+1, z-1]
    neighborhood[2] = img[x+1, y+1, z-1]
    neighborhood[3] = 0
    neighborhood[4] = img[x, y, z-1]
    neighborhood[5] = img[x+1, y, z-1]
    neighborhood[6] = 0
    neighborhood[7] = 0
    neighborhood[8] = 0
    neighborhood[9] = 0
    neighborhood[10] = img[x, y+1, z]
    neighborhood[11] = img[x+1, y+1, z]
    neighborhood[12] = 0
    neighborhood[13] = img[x, y, z]
    neighborhood[14] = img[x+1, y, z]
    neighborhood[15] = 0
    neighborhood[16] = 0
    neighborhood[17] = 0
    neighborhood[18] = 0
    neighborhood[19] = img[x, y+1, z+1]
    neighborhood[20] = img[x+1, y+1, z+1]
    neighborhood[21] = 0
    neighborhood[22] = img[x, y, z+1]
    neighborhood[23] = img[x+1, y, z+1]
    neighborhood[24] = 0
    neighborhood[25] = 0
    neighborhood[26] = 0


cdef void get_neighborhood_boundary_edges_3(pixel_type[:, :, ::1] img,
                               npy_intp x, npy_intp y, npy_intp z,
                               pixel_type neighborhood[]) nogil:
    neighborhood[0] = 0
    neighborhood[1] = 0
    neighborhood[2] = 0
    neighborhood[3] = 0
    neighborhood[4] = img[x, y, z-1]
    neighborhood[5] = img[x+1, y, z-1]
    neighborhood[6] = 0
    neighborhood[7] = img[x, y-1, z-1]
    neighborhood[8] = img[x+1, y-1, z-1]
    neighborhood[9] = 0
    neighborhood[10] = 0
    neighborhood[11] = 0
    neighborhood[12] = 0
    neighborhood[13] = img[x, y, z]
    neighborhood[14] = img[x+1, y, z]
    neighborhood[15] = 0
    neighborhood[16] = img[x, y-1, z]
    neighborhood[17] = img[x+1, y-1, z]
    neighborhood[18] = 0
    neighborhood[19] = 0
    neighborhood[20] = 0
    neighborhood[21] = 0
    neighborhood[22] = img[x, y, z+1]
    neighborhood[23] = img[x+1, y, z+1]
    neighborhood[24] = 0
    neighborhood[25] = img[x, y-1, z+1]
    neighborhood[26] = img[x+1, y-1, z+1]


cdef void get_neighborhood_boundary_edges_4(pixel_type[:, :, ::1] img,
                               npy_intp x, npy_intp y, npy_intp z,
                               pixel_type neighborhood[]) nogil:
    neighborhood[0] = 0
    neighborhood[1] = 0
    neighborhood[2] = 0
    neighborhood[3] = 0
    neighborhood[4] = 0
    neighborhood[5] = 0
    neighborhood[6] = 0
    neighborhood[7] = 0
    neighborhood[8] = 0
    neighborhood[9] = img[x-1, y+1, z]
    neighborhood[10] = img[x, y+1, z]
    neighborhood[11] = 0
    neighborhood[12] = img[x-1, y, z]
    neighborhood[13] = img[x, y, z]
    neighborhood[14] = 0
    neighborhood[15] = img[x-1, y-1, z]
    neighborhood[16] = img[x, y-1, z]
    neighborhood[17] = 0
    neighborhood[18] = img[x-1, y+1, z+1]
    neighborhood[19] = img[x, y+1, z+1]
    neighborhood[20] = 0
    neighborhood[21] = img[x-1, y, z+1]
    neighborhood[22] = img[x, y, z+1]
    neighborhood[23] = 0
    neighborhood[24] = img[x-1, y-1, z+1]
    neighborhood[25] = img[x, y-1, z+1]
    neighborhood[26] = 0


cdef void get_neighborhood_boundary_edges_5(pixel_type[:, :, ::1] img,
                               npy_intp x, npy_intp y, npy_intp z,
                               pixel_type neighborhood[]) nogil:
    neighborhood[0] = img[x-1, y+1, z-1]
    neighborhood[1] = img[x, y+1, z-1]
    neighborhood[2] = 0
    neighborhood[3] = img[x-1, y, z-1]
    neighborhood[4] = img[x, y, z-1]
    neighborhood[5] = 0
    neighborhood[6] = img[x-1, y-1, z-1]
    neighborhood[7] = img[x, y-1, z-1]
    neighborhood[8] = 0
    neighborhood[9] = img[x-1, y+1, z]
    neighborhood[10] = img[x, y+1, z]
    neighborhood[11] = 0
    neighborhood[12] = img[x-1, y, z]
    neighborhood[13] = img[x, y, z]
    neighborhood[14] = 0
    neighborhood[15] = img[x-1, y-1, z]
    neighborhood[16] = img[x, y-1, z]
    neighborhood[17] = 0
    neighborhood[18] = 0
    neighborhood[19] = 0
    neighborhood[20] = 0
    neighborhood[21] = 0
    neighborhood[22] = 0
    neighborhood[23] = 0
    neighborhood[24] = 0
    neighborhood[25] = 0
    neighborhood[26] = 0


cdef void get_neighborhood_boundary_edges_6(pixel_type[:, :, ::1] img,
                               npy_intp x, npy_intp y, npy_intp z,
                               pixel_type neighborhood[]) nogil:
    neighborhood[0] = img[x-1, y+1, z-1]
    neighborhood[1] = img[x, y+1, z-1]
    neighborhood[2] = 0
    neighborhood[3] = img[x-1, y, z-1]
    neighborhood[4] = img[x, y, z-1]
    neighborhood[5] = 0
    neighborhood[6] = 0
    neighborhood[7] = 0
    neighborhood[8] = 0
    neighborhood[9] = img[x-1, y+1, z]
    neighborhood[10] = img[x, y+1, z]
    neighborhood[11] = 0
    neighborhood[12] = img[x-1, y, z]
    neighborhood[13] = img[x, y, z]
    neighborhood[14] = 0
    neighborhood[15] = 0
    neighborhood[16] = 0
    neighborhood[17] = 0
    neighborhood[18] = img[x-1, y+1, z+1]
    neighborhood[19] = img[x, y+1, z+1]
    neighborhood[20] = 0
    neighborhood[21] = img[x-1, y, z+1]
    neighborhood[22] = img[x, y, z+1]
    neighborhood[23] = 0
    neighborhood[24] = 0
    neighborhood[25] = 0
    neighborhood[26] = 0


cdef void get_neighborhood_boundary_edges_7(pixel_type[:, :, ::1] img,
                               npy_intp x, npy_intp y, npy_intp z,
                               pixel_type neighborhood[]) nogil:
    neighborhood[0] = 0
    neighborhood[1] = 0
    neighborhood[2] = 0
    neighborhood[3] = img[x-1, y, z-1]
    neighborhood[4] = img[x, y, z-1]
    neighborhood[5] = 0
    neighborhood[6] = img[x-1, y-1, z-1]
    neighborhood[7] = img[x, y-1, z-1]
    neighborhood[8] = 0
    neighborhood[9] = 0
    neighborhood[10] = 0
    neighborhood[11] = 0
    neighborhood[12] = img[x-1, y, z]
    neighborhood[13] = img[x, y, z]
    neighborhood[14] = 0
    neighborhood[15] = img[x-1, y-1, z]
    neighborhood[16] = img[x, y-1, z]
    neighborhood[17] = 0
    neighborhood[18] = 0
    neighborhood[19] = 0
    neighborhood[20] = 0
    neighborhood[21] = img[x-1, y, z+1]
    neighborhood[22] = img[x, y, z+1]
    neighborhood[23] = 0
    neighborhood[24] = img[x-1, y-1, z+1]
    neighborhood[25] = img[x, y-1, z+1]
    neighborhood[26] = 0


cdef void get_neighborhood_boundary_edges_8(pixel_type[:, :, ::1] img,
                               npy_intp x, npy_intp y, npy_intp z,
                               pixel_type neighborhood[]) nogil:
    neighborhood[0] = 0
    neighborhood[1] = 0
    neighborhood[2] = 0
    neighborhood[3] = 0
    neighborhood[4] = 0
    neighborhood[5] = 0
    neighborhood[6] = 0
    neighborhood[7] = 0
    neighborhood[8] = 0
    neighborhood[9] = img[x-1, y+1, z]
    neighborhood[10] = img[x, y+1, z]
    neighborhood[11] = img[x+1, y+1, z]
    neighborhood[12] = img[x-1, y, z]
    neighborhood[13] = img[x, y, z]
    neighborhood[14] = img[x+1, y, z]
    neighborhood[15] = 0
    neighborhood[16] = 0
    neighborhood[17] = 0
    neighborhood[18] = img[x-1, y+1, z+1]
    neighborhood[19] = img[x, y+1, z+1]
    neighborhood[20] = img[x+1, y+1, z+1]
    neighborhood[21] = img[x-1, y, z+1]
    neighborhood[22] = img[x, y, z+1]
    neighborhood[23] = img[x+1, y, z+1]
    neighborhood[24] = 0
    neighborhood[25] = 0
    neighborhood[26] = 0


cdef void get_neighborhood_boundary_edges_9(pixel_type[:, :, ::1] img,
                               npy_intp x, npy_intp y, npy_intp z,
                               pixel_type neighborhood[]) nogil:
    neighborhood[0] = img[x-1, y+1, z-1]
    neighborhood[1] = img[x, y+1, z-1]
    neighborhood[2] = img[x+1, y+1, z-1]
    neighborhood[3] = img[x-1, y, z-1]
    neighborhood[4] = img[x, y, z-1]
    neighborhood[5] = img[x+1, y, z-1]
    neighborhood[6] = 0
    neighborhood[7] = 0
    neighborhood[8] = 0
    neighborhood[9] = img[x-1, y+1, z]
    neighborhood[10] = img[x, y+1, z]
    neighborhood[11] = img[x+1, y+1, z]
    neighborhood[12] = img[x-1, y, z]
    neighborhood[13] = img[x, y, z]
    neighborhood[14] = img[x+1, y, z]
    neighborhood[15] = 0
    neighborhood[16] = 0
    neighborhood[17] = 0
    neighborhood[18] = 0
    neighborhood[19] = 0
    neighborhood[20] = 0
    neighborhood[21] = 0
    neighborhood[22] = 0
    neighborhood[23] = 0
    neighborhood[24] = 0
    neighborhood[25] = 0
    neighborhood[26] = 0


cdef void get_neighborhood_boundary_edges_10(pixel_type[:, :, ::1] img,
                               npy_intp x, npy_intp y, npy_intp z,
                               pixel_type neighborhood[]) nogil:
    neighborhood[0] = 0
    neighborhood[1] = 0
    neighborhood[2] = 0
    neighborhood[3] = 0
    neighborhood[4] = 0
    neighborhood[5] = 0
    neighborhood[6] = 0
    neighborhood[7] = 0
    neighborhood[8] = 0
    neighborhood[9] = 0
    neighborhood[10] = 0
    neighborhood[11] = 0
    neighborhood[12] = img[x-1, y, z]
    neighborhood[13] = img[x, y, z]
    neighborhood[14] = img[x+1, y, z]
    neighborhood[15] = img[x-1, y-1, z]
    neighborhood[16] = img[x, y-1, z]
    neighborhood[17] = img[x+1, y-1, z]
    neighborhood[18] = 0
    neighborhood[19] = 0
    neighborhood[20] = 0
    neighborhood[21] = img[x-1, y, z+1]
    neighborhood[22] = img[x, y, z+1]
    neighborhood[23] = img[x+1, y, z+1]
    neighborhood[24] = img[x-1, y-1, z+1]
    neighborhood[25] = img[x, y-1, z+1]
    neighborhood[26] = img[x+1, y-1, z+1]


cdef void get_neighborhood_boundary_edges_11(pixel_type[:, :, ::1] img,
                               npy_intp x, npy_intp y, npy_intp z,
                               pixel_type neighborhood[]) nogil:
    neighborhood[0] = 0
    neighborhood[1] = 0
    neighborhood[2] = 0
    neighborhood[3] = img[x-1, y, z-1]
    neighborhood[4] = img[x, y, z-1]
    neighborhood[5] = img[x+1, y, z-1]
    neighborhood[6] = img[x-1, y-1, z-1]
    neighborhood[7] = img[x, y-1, z-1]
    neighborhood[8] = img[x+1, y-1, z-1]
    neighborhood[9] = 0
    neighborhood[10] = 0
    neighborhood[11] = 0
    neighborhood[12] = img[x-1, y, z]
    neighborhood[13] = img[x, y, z]
    neighborhood[14] = img[x+1, y, z]
    neighborhood[15] = img[x-1, y-1, z]
    neighborhood[16] = img[x, y-1, z]
    neighborhood[17] = img[x+1, y-1, z]
    neighborhood[18] = 0
    neighborhood[19] = 0
    neighborhood[20] = 0
    neighborhood[21] = 0
    neighborhood[22] = 0
    neighborhood[23] = 0
    neighborhood[24] = 0
    neighborhood[25] = 0
    neighborhood[26] = 0


cdef void get_neighborhood_boundary_corners_0(pixel_type[:, :, ::1] img,
                               npy_intp x, npy_intp y, npy_intp z,
                               pixel_type neighborhood[]) nogil:
    neighborhood[0] = 0
    neighborhood[1] = 0
    neighborhood[2] = 0
    neighborhood[3] = 0
    neighborhood[4] = 0
    neighborhood[5] = 0
    neighborhood[6] = 0
    neighborhood[7] = 0
    neighborhood[8] = 0
    neighborhood[9] = 0
    neighborhood[10] = img[x, y+1, z]
    neighborhood[11] = img[x+1, y+1, z]
    neighborhood[12] = 0
    neighborhood[13] = img[x, y, z]
    neighborhood[14] = img[x+1, y, z]
    neighborhood[15] = 0
    neighborhood[16] = 0
    neighborhood[17] = 0
    neighborhood[18] = 0
    neighborhood[19] = img[x, y+1, z+1]
    neighborhood[20] = img[x+1, y+1, z+1]
    neighborhood[21] = 0
    neighborhood[22] = img[x, y, z+1]
    neighborhood[23] = img[x+1, y, z+1]
    neighborhood[24] = 0
    neighborhood[25] = 0
    neighborhood[26] = 0


cdef void get_neighborhood_boundary_corners_1(pixel_type[:, :, ::1] img,
                               npy_intp x, npy_intp y, npy_intp z,
                               pixel_type neighborhood[]) nogil:
    neighborhood[0] = 0
    neighborhood[1] = img[x, y+1, z-1]
    neighborhood[2] = img[x+1, y+1, z-1]
    neighborhood[3] = 0
    neighborhood[4] = img[x, y, z-1]
    neighborhood[5] = img[x+1, y, z-1]
    neighborhood[6] = 0
    neighborhood[7] = 0
    neighborhood[8] = 0
    neighborhood[9] = 0
    neighborhood[10] = img[x, y+1, z]
    neighborhood[11] = img[x+1, y+1, z]
    neighborhood[12] = 0
    neighborhood[13] = img[x, y, z]
    neighborhood[14] = img[x+1, y, z]
    neighborhood[15] = 0
    neighborhood[16] = 0
    neighborhood[17] = 0
    neighborhood[18] = 0
    neighborhood[19] = 0
    neighborhood[20] = 0
    neighborhood[21] = 0
    neighborhood[22] = 0
    neighborhood[23] = 0
    neighborhood[24] = 0
    neighborhood[25] = 0
    neighborhood[26] = 0


cdef void get_neighborhood_boundary_corners_2(pixel_type[:, :, ::1] img,
                               npy_intp x, npy_intp y, npy_intp z,
                               pixel_type neighborhood[]) nogil:
    neighborhood[0] = 0
    neighborhood[1] = 0
    neighborhood[2] = 0
    neighborhood[3] = 0
    neighborhood[4] = 0
    neighborhood[5] = 0
    neighborhood[6] = 0
    neighborhood[7] = 0
    neighborhood[8] = 0
    neighborhood[9] = 0
    neighborhood[10] = 0
    neighborhood[11] = 0
    neighborhood[12] = 0
    neighborhood[13] = img[x, y, z]
    neighborhood[14] = img[x+1, y, z]
    neighborhood[15] = 0
    neighborhood[16] = img[x, y-1, z]
    neighborhood[17] = img[x+1, y-1, z]
    neighborhood[18] = 0
    neighborhood[19] = 0
    neighborhood[20] = 0
    neighborhood[21] = 0
    neighborhood[22] = img[x, y, z+1]
    neighborhood[23] = img[x+1, y, z+1]
    neighborhood[24] = 0
    neighborhood[25] = img[x, y-1, z+1]
    neighborhood[26] = img[x+1, y-1, z+1]


cdef void get_neighborhood_boundary_corners_3(pixel_type[:, :, ::1] img,
                               npy_intp x, npy_intp y, npy_intp z,
                               pixel_type neighborhood[]) nogil:
    neighborhood[0] = 0
    neighborhood[1] = 0
    neighborhood[2] = 0
    neighborhood[3] = 0
    neighborhood[4] = img[x, y, z-1]
    neighborhood[5] = img[x+1, y, z-1]
    neighborhood[6] = 0
    neighborhood[7] = img[x, y-1, z-1]
    neighborhood[8] = img[x+1, y-1, z-1]
    neighborhood[9] = 0
    neighborhood[10] = 0
    neighborhood[11] = 0
    neighborhood[12] = 0
    neighborhood[13] = img[x, y, z]
    neighborhood[14] = img[x+1, y, z]
    neighborhood[15] = 0
    neighborhood[16] = img[x, y-1, z]
    neighborhood[17] = img[x+1, y-1, z]
    neighborhood[18] = 0
    neighborhood[19] = 0
    neighborhood[20] = 0
    neighborhood[21] = 0
    neighborhood[22] = 0
    neighborhood[23] = 0
    neighborhood[24] = 0
    neighborhood[25] = 0
    neighborhood[26] = 0


cdef void get_neighborhood_boundary_corners_4(pixel_type[:, :, ::1] img,
                               npy_intp x, npy_intp y, npy_intp z,
                               pixel_type neighborhood[]) nogil:
    neighborhood[0] = 0
    neighborhood[1] = 0
    neighborhood[2] = 0
    neighborhood[3] = 0
    neighborhood[4] = 0
    neighborhood[5] = 0
    neighborhood[6] = 0
    neighborhood[7] = 0
    neighborhood[8] = 0
    neighborhood[9] = img[x-1, y+1, z]
    neighborhood[10] = img[x, y+1, z]
    neighborhood[11] = 0
    neighborhood[12] = img[x-1, y, z]
    neighborhood[13] = img[x, y, z]
    neighborhood[14] = 0
    neighborhood[15] = 0
    neighborhood[16] = 0
    neighborhood[17] = 0
    neighborhood[18] = img[x-1, y+1, z+1]
    neighborhood[19] = img[x, y+1, z+1]
    neighborhood[20] = 0
    neighborhood[21] = img[x-1, y, z+1]
    neighborhood[22] = img[x, y, z+1]
    neighborhood[23] = 0
    neighborhood[24] = 0
    neighborhood[25] = 0
    neighborhood[26] = 0


cdef void get_neighborhood_boundary_corners_5(pixel_type[:, :, ::1] img,
                               npy_intp x, npy_intp y, npy_intp z,
                               pixel_type neighborhood[]) nogil:
    neighborhood[0] = img[x-1, y+1, z-1]
    neighborhood[1] = img[x, y+1, z-1]
    neighborhood[2] = 0
    neighborhood[3] = img[x-1, y, z-1]
    neighborhood[4] = img[x, y, z-1]
    neighborhood[5] = 0
    neighborhood[6] = 0
    neighborhood[7] = 0
    neighborhood[8] = 0
    neighborhood[9] = img[x-1, y+1, z]
    neighborhood[10] = img[x, y+1, z]
    neighborhood[11] = 0
    neighborhood[12] = img[x-1, y, z]
    neighborhood[13] = img[x, y, z]
    neighborhood[14] = 0
    neighborhood[15] = 0
    neighborhood[16] = 0
    neighborhood[17] = 0
    neighborhood[18] = 0
    neighborhood[19] = 0
    neighborhood[20] = 0
    neighborhood[21] = 0
    neighborhood[22] = 0
    neighborhood[23] = 0
    neighborhood[24] = 0
    neighborhood[25] = 0
    neighborhood[26] = 0


cdef void get_neighborhood_boundary_corners_6(pixel_type[:, :, ::1] img,
                               npy_intp x, npy_intp y, npy_intp z,
                               pixel_type neighborhood[]) nogil:
    neighborhood[0] = 0
    neighborhood[1] = 0
    neighborhood[2] = 0
    neighborhood[3] = 0
    neighborhood[4] = 0
    neighborhood[5] = 0
    neighborhood[6] = 0
    neighborhood[7] = 0
    neighborhood[8] = 0
    neighborhood[9] = 0
    neighborhood[10] = 0
    neighborhood[11] = 0
    neighborhood[12] = img[x-1, y, z]
    neighborhood[13] = img[x, y, z]
    neighborhood[14] = 0
    neighborhood[15] = img[x-1, y-1, z]
    neighborhood[16] = img[x, y-1, z]
    neighborhood[17] = 0
    neighborhood[18] = 0
    neighborhood[19] = 0
    neighborhood[20] = 0
    neighborhood[21] = img[x-1, y, z+1]
    neighborhood[22] = img[x, y, z+1]
    neighborhood[23] = 0
    neighborhood[24] = img[x-1, y-1, z+1]
    neighborhood[25] = img[x, y-1, z+1]
    neighborhood[26] = 0


cdef void get_neighborhood_boundary_corners_7(pixel_type[:, :, ::1] img,
                               npy_intp x, npy_intp y, npy_intp z,
                               pixel_type neighborhood[]) nogil:
    neighborhood[0] = 0
    neighborhood[1] = 0
    neighborhood[2] = 0
    neighborhood[3] = img[x-1, y, z-1]
    neighborhood[4] = img[x, y, z-1]
    neighborhood[5] = 0
    neighborhood[6] = img[x-1, y-1, z-1]
    neighborhood[7] = img[x, y-1, z-1]
    neighborhood[8] = 0
    neighborhood[9] = 0
    neighborhood[10] = 0
    neighborhood[11] = 0
    neighborhood[12] = img[x-1, y, z]
    neighborhood[13] = img[x, y, z]
    neighborhood[14] = 0
    neighborhood[15] = img[x-1, y-1, z]
    neighborhood[16] = img[x, y-1, z]
    neighborhood[17] = 0
    neighborhood[18] = 0
    neighborhood[19] = 0
    neighborhood[20] = 0
    neighborhood[21] = 0
    neighborhood[22] = 0
    neighborhood[23] = 0
    neighborhood[24] = 0
    neighborhood[25] = 0
    neighborhood[26] = 0


cdef int is_Euler_invariant_Octant_0(pixel_type neighbors[],
                             int[::1] lut) nogil:

    cdef int n, euler_char = 0
    # octant 0:
    n = 1
    if neighbors[0] == 1:
        n |= 128

    if neighbors[9] == 1:
        n |= 64

    if neighbors[3] == 1:
        n |= 32

    if neighbors[12] == 1:
        n |= 16

    if neighbors[1] == 1:
        n |= 8

    if neighbors[10] == 1:
        n |= 4

    if neighbors[4] == 1:
        n |= 2

    euler_char += lut[n]
    return euler_char


cdef int is_Euler_invariant_Octant_1(pixel_type neighbors[],
                             int[::1] lut) nogil:

    cdef int n, euler_char = 0
    # octant 1:
    n = 1
    if neighbors[2] == 1:
        n |= 128

    if neighbors[1] == 1:
        n |= 64

    if neighbors[11] == 1:
        n |= 32

    if neighbors[10] == 1:
        n |= 16

    if neighbors[5] == 1:
        n |= 8

    if neighbors[4] == 1:
        n |= 4

    if neighbors[14] == 1:
        n |= 2

    euler_char += lut[n]
    return euler_char


cdef int is_Euler_invariant_Octant_2(pixel_type neighbors[],
                             int[::1] lut) nogil:

    cdef int n, euler_char = 0
    # octant 2:
    n = 1
    if neighbors[6] == 1:
        n |= 128

    if neighbors[15] == 1:
        n |= 64

    if neighbors[7] == 1:
        n |= 32

    if neighbors[16] == 1:
        n |= 16

    if neighbors[3] == 1:
        n |= 8

    if neighbors[12] == 1:
        n |= 4

    if neighbors[4] == 1:
        n |= 2

    euler_char += lut[n]
    return euler_char


cdef int is_Euler_invariant_Octant_3(pixel_type neighbors[],
                             int[::1] lut) nogil:

    cdef int n, euler_char = 0
    # octant 3:
    n = 1
    if neighbors[8] == 1:
        n |= 128

    if neighbors[7] == 1:
        n |= 64

    if neighbors[17] == 1:
        n |= 32

    if neighbors[16] == 1:
        n |= 16

    if neighbors[5] == 1:
        n |= 8

    if neighbors[4] == 1:
        n |= 4

    if neighbors[14] == 1:
        n |= 2

    euler_char += lut[n]
    return euler_char


cdef int is_Euler_invariant_Octant_4(pixel_type neighbors[],
                             int[::1] lut) nogil:

    cdef int n, euler_char = 0
    # octant 4:
    n = 1
    if neighbors[18] == 1:
        n |= 128

    if neighbors[21] == 1:
        n |= 64

    if neighbors[9] == 1:
        n |= 32

    if neighbors[12] == 1:
        n |= 16

    if neighbors[19] == 1:
        n |= 8

    if neighbors[22] == 1:
        n |= 4

    if neighbors[10] == 1:
        n |= 2

    euler_char += lut[n]
    return euler_char


cdef int is_Euler_invariant_Octant_5(pixel_type neighbors[],
                             int[::1] lut) nogil:

    cdef int n, euler_char = 0
    # octant 5:
    n = 1
    if neighbors[20] == 1:
        n |= 128

    if neighbors[23] == 1:
        n |= 64

    if neighbors[19] == 1:
        n |= 32

    if neighbors[22] == 1:
        n |= 16

    if neighbors[11] == 1:
        n |= 8

    if neighbors[14] == 1:
        n |= 4

    if neighbors[10] == 1:
        n |= 2

    euler_char += lut[n]
    return euler_char


cdef int is_Euler_invariant_Octant_6(pixel_type neighbors[],
                             int[::1] lut) nogil:

    cdef int n, euler_char = 0
    # octant 6:
    n = 1
    if neighbors[24] == 1:
        n |= 128

    if neighbors[25] == 1:
        n |= 64

    if neighbors[15] == 1:
        n |= 32

    if neighbors[16] == 1:
        n |= 16

    if neighbors[21] == 1:
        n |= 8

    if neighbors[22] == 1:
        n |= 4

    if neighbors[12] == 1:
        n |= 2

    euler_char += lut[n]
    return euler_char


cdef int is_Euler_invariant_Octant_7(pixel_type neighbors[],
                             int[::1] lut) nogil:

    cdef int n, euler_char = 0
    # octant 7:
    n = 1
    if neighbors[26] == 1:
        n |= 128

    if neighbors[23] == 1:
        n |= 64

    if neighbors[17] == 1:
        n |= 32

    if neighbors[14] == 1:
        n |= 16

    if neighbors[25] == 1:
        n |= 8

    if neighbors[22] == 1:
        n |= 4

    if neighbors[16] == 1:
        n |= 2

    euler_char += lut[n]
    return euler_char


cdef void find_simple_point_candidates_faces_0(pixel_type[:, :, ::1] img,
                                        npy_float32[:, :, ::1] edt,
                                        int curr_border,
                                        npy_intp [:,:] fLoop,
                                        vector[coordinate] & simple_border_points) nogil:

    cdef:
        cdef coordinate point

        pixel_type neighborhood[27]
        npy_intp x, y, z
        bint is_border_pt
        int euler_char = 0
        int[::1] Euler_LUT = LUT


    for x in range(fLoop[0,0], fLoop[0,1]):
        for y in range(fLoop[1,0], fLoop[1,1]):
            for z in range(fLoop[2,0], fLoop[2,1]):

                #img[x,y,z] = 155   
                if img[x, y, z] != 1:
                    continue

                is_border_pt = (
                                curr_border == 1 and img[x+1,y  ,z  ] == 0 or  #E
                                curr_border == 2 and img[x  ,y-1,z  ] == 0 or  #S
                                curr_border == 3 and img[x  ,y+1,z  ] == 0 or  #N
                                curr_border == 4 and img[x  ,y  ,z-1] == 0 or  #B
                                curr_border == 5 and img[x  ,y  ,z+1] == 0     #U
                            )

                if not is_border_pt:
                    continue

                get_neighborhood_boundary_faces_0(img, x, y, z, neighborhood)
                if (not is_endpoint(neighborhood)):
                    euler_char = 0
                    euler_char = euler_char + is_Euler_invariant_Octant_1(neighborhood, Euler_LUT)
                    euler_char = euler_char + is_Euler_invariant_Octant_3(neighborhood, Euler_LUT)
                    euler_char = euler_char + is_Euler_invariant_Octant_5(neighborhood, Euler_LUT)
                    euler_char = euler_char + is_Euler_invariant_Octant_7(neighborhood, Euler_LUT)
                    if not euler_char==0 or not is_simple_point(neighborhood):
                        continue

                    point.x = x
                    point.y = y
                    point.z = z
                    point.ID = 10
                    point.faceCount = is_endpoint_check(neighborhood)
                    point.edt = edt[x,y,z]
                    simple_border_points.push_back(point)


cdef void find_simple_point_candidates_faces_1(pixel_type[:, :, ::1] img,
                                        npy_float32[:, :, ::1] edt,
                                        int curr_border,
                                        npy_intp [:,:] fLoop,
                                        vector[coordinate] & simple_border_points) nogil:

    cdef:
        cdef coordinate point

        pixel_type neighborhood[27]
        npy_intp x, y, z
        bint is_border_pt
        int euler_char = 0
        int[::1] Euler_LUT = LUT


    for x in range(fLoop[0,0], fLoop[0,1]):
        for y in range(fLoop[1,0], fLoop[1,1]):
            for z in range(fLoop[2,0], fLoop[2,1]):

                #img[x,y,z] = 155   
                if img[x, y, z] != 1:
                    continue

                is_border_pt = (
                                curr_border == 0 and img[x-1,y  ,z  ] == 0 or  #W
                                curr_border == 2 and img[x  ,y-1,z  ] == 0 or  #S
                                curr_border == 3 and img[x  ,y+1,z  ] == 0 or  #N
                                curr_border == 4 and img[x  ,y  ,z-1] == 0 or  #B
                                curr_border == 5 and img[x  ,y  ,z+1] == 0     #U
                            )

                if not is_border_pt:
                    continue

                get_neighborhood_boundary_faces_1(img, x, y, z, neighborhood)
                if (not is_endpoint(neighborhood)):
                    euler_char = 0
                    euler_char = euler_char + is_Euler_invariant_Octant_0(neighborhood, Euler_LUT)
                    euler_char = euler_char + is_Euler_invariant_Octant_2(neighborhood, Euler_LUT)
                    euler_char = euler_char + is_Euler_invariant_Octant_4(neighborhood, Euler_LUT)
                    euler_char = euler_char + is_Euler_invariant_Octant_6(neighborhood, Euler_LUT)
                    if not euler_char==0 or not is_simple_point(neighborhood):
                        continue

                    point.x = x
                    point.y = y
                    point.z = z
                    point.ID = 11
                    point.faceCount = is_endpoint_check(neighborhood)
                    point.edt = edt[x,y,z]
                    simple_border_points.push_back(point)


cdef void find_simple_point_candidates_faces_2(pixel_type[:, :, ::1] img,
                                        npy_float32[:, :, ::1] edt,
                                        int curr_border,
                                        npy_intp [:,:] fLoop,
                                        vector[coordinate] & simple_border_points) nogil:

    cdef:
        cdef coordinate point

        pixel_type neighborhood[27]
        npy_intp x, y, z
        bint is_border_pt
        int euler_char = 0
        int[::1] Euler_LUT = LUT


    for x in range(fLoop[0,0], fLoop[0,1]):
        for y in range(fLoop[1,0], fLoop[1,1]):
            for z in range(fLoop[2,0], fLoop[2,1]):

                #img[x,y,z] = 155   
                if img[x, y, z] != 1:
                    continue

                is_border_pt = (
                                curr_border == 0 and img[x-1,y  ,z  ] == 0 or  #W
                                curr_border == 1 and img[x+1,y  ,z  ] == 0 or  #E
                                curr_border == 3 and img[x  ,y+1,z  ] == 0 or  #N
                                curr_border == 4 and img[x  ,y  ,z-1] == 0 or  #B
                                curr_border == 5 and img[x  ,y  ,z+1] == 0     #U
                            )

                if not is_border_pt:
                    continue

                get_neighborhood_boundary_faces_2(img, x, y, z, neighborhood)
                if (not is_endpoint(neighborhood)):
                    euler_char = 0
                    euler_char = euler_char + is_Euler_invariant_Octant_0(neighborhood, Euler_LUT)
                    euler_char = euler_char + is_Euler_invariant_Octant_1(neighborhood, Euler_LUT)
                    euler_char = euler_char + is_Euler_invariant_Octant_4(neighborhood, Euler_LUT)
                    euler_char = euler_char + is_Euler_invariant_Octant_5(neighborhood, Euler_LUT)
                    if not euler_char==0 or not is_simple_point(neighborhood):
                        continue

                    point.x = x
                    point.y = y
                    point.z = z
                    point.ID = 12
                    point.faceCount = is_endpoint_check(neighborhood)
                    point.edt = edt[x,y,z]
                    simple_border_points.push_back(point)


cdef void find_simple_point_candidates_faces_3(pixel_type[:, :, ::1] img,
                                        npy_float32[:, :, ::1] edt,
                                        int curr_border,
                                        npy_intp [:,:] fLoop,
                                        vector[coordinate] & simple_border_points) nogil:

    cdef:
        cdef coordinate point

        pixel_type neighborhood[27]
        npy_intp x, y, z
        bint is_border_pt
        int euler_char = 0
        int[::1] Euler_LUT = LUT


    for x in range(fLoop[0,0], fLoop[0,1]):
        for y in range(fLoop[1,0], fLoop[1,1]):
            for z in range(fLoop[2,0], fLoop[2,1]):

                #img[x,y,z] = 155   
                if img[x, y, z] != 1:
                    continue

                is_border_pt = (
                                curr_border == 0 and img[x-1,y  ,z  ] == 0 or  #W
                                curr_border == 1 and img[x+1,y  ,z  ] == 0 or  #E
                                curr_border == 2 and img[x  ,y-1,z  ] == 0 or  #S
                                curr_border == 4 and img[x  ,y  ,z-1] == 0 or  #B
                                curr_border == 5 and img[x  ,y  ,z+1] == 0     #U
                            )

                if not is_border_pt:
                    continue

                get_neighborhood_boundary_faces_3(img, x, y, z, neighborhood)
                if (not is_endpoint(neighborhood)):
                    euler_char = 0
                    euler_char = euler_char + is_Euler_invariant_Octant_2(neighborhood, Euler_LUT)
                    euler_char = euler_char + is_Euler_invariant_Octant_3(neighborhood, Euler_LUT)
                    euler_char = euler_char + is_Euler_invariant_Octant_6(neighborhood, Euler_LUT)
                    euler_char = euler_char + is_Euler_invariant_Octant_7(neighborhood, Euler_LUT)
                    if not euler_char==0 or not is_simple_point(neighborhood):
                        continue

                    point.x = x
                    point.y = y
                    point.z = z
                    point.ID = 13
                    point.faceCount = is_endpoint_check(neighborhood)
                    point.edt = edt[x,y,z]
                    simple_border_points.push_back(point)


cdef void find_simple_point_candidates_faces_4(pixel_type[:, :, ::1] img,
                                        npy_float32[:, :, ::1] edt,
                                        int curr_border,
                                        npy_intp [:,:] fLoop,
                                        vector[coordinate] & simple_border_points) nogil:

    cdef:
        cdef coordinate point

        pixel_type neighborhood[27]
        npy_intp x, y, z
        bint is_border_pt
        int euler_char = 0
        int[::1] Euler_LUT = LUT


    for x in range(fLoop[0,0], fLoop[0,1]):
        for y in range(fLoop[1,0], fLoop[1,1]):
            for z in range(fLoop[2,0], fLoop[2,1]):

                #img[x,y,z] = 155   
                if img[x, y, z] != 1:
                    continue

                is_border_pt = (
                                curr_border == 0 and img[x-1,y  ,z  ] == 0 or  #W
                                curr_border == 1 and img[x+1,y  ,z  ] == 0 or  #E
                                curr_border == 2 and img[x  ,y-1,z  ] == 0 or  #S
                                curr_border == 3 and img[x  ,y+1,z  ] == 0 or  #N
                                curr_border == 5 and img[x  ,y  ,z+1] == 0     #U
                            )

                if not is_border_pt:
                    continue

                get_neighborhood_boundary_faces_4(img, x, y, z, neighborhood)
                if (not is_endpoint(neighborhood)):
                    euler_char = 0
                    euler_char = euler_char + is_Euler_invariant_Octant_4(neighborhood, Euler_LUT)
                    euler_char = euler_char + is_Euler_invariant_Octant_5(neighborhood, Euler_LUT)
                    euler_char = euler_char + is_Euler_invariant_Octant_6(neighborhood, Euler_LUT)
                    euler_char = euler_char + is_Euler_invariant_Octant_7(neighborhood, Euler_LUT)
                    if not euler_char==0 or not is_simple_point(neighborhood):
                        continue

                    point.x = x
                    point.y = y
                    point.z = z
                    point.ID = 14
                    point.faceCount = is_endpoint_check(neighborhood)
                    point.edt = edt[x,y,z]
                    simple_border_points.push_back(point)


cdef void find_simple_point_candidates_faces_5(pixel_type[:, :, ::1] img,
                                        npy_float32[:, :, ::1] edt,
                                        int curr_border,
                                        npy_intp [:,:] fLoop,
                                        vector[coordinate] & simple_border_points) nogil:

    cdef:
        cdef coordinate point

        pixel_type neighborhood[27]
        npy_intp x, y, z
        bint is_border_pt
        int euler_char = 0
        int[::1] Euler_LUT = LUT


    for x in range(fLoop[0,0], fLoop[0,1]):
        for y in range(fLoop[1,0], fLoop[1,1]):
            for z in range(fLoop[2,0], fLoop[2,1]):

                #img[x,y,z] = 155   
                if img[x, y, z] != 1:
                    continue

                is_border_pt = (
                                curr_border == 0 and img[x-1,y  ,z  ] == 0 or  #W
                                curr_border == 1 and img[x+1,y  ,z  ] == 0 or  #E
                                curr_border == 2 and img[x  ,y-1,z  ] == 0 or  #S
                                curr_border == 3 and img[x  ,y+1,z  ] == 0 or  #N
                                curr_border == 4 and img[x  ,y  ,z-1] == 0     #B
                            )

                if not is_border_pt:
                    continue

                get_neighborhood_boundary_faces_5(img, x, y, z, neighborhood)
                if (not is_endpoint(neighborhood)):
                    euler_char = 0
                    euler_char = euler_char + is_Euler_invariant_Octant_0(neighborhood, Euler_LUT)
                    euler_char = euler_char + is_Euler_invariant_Octant_1(neighborhood, Euler_LUT)
                    euler_char = euler_char + is_Euler_invariant_Octant_2(neighborhood, Euler_LUT)
                    euler_char = euler_char + is_Euler_invariant_Octant_3(neighborhood, Euler_LUT)
                    if not euler_char==0 or not is_simple_point(neighborhood):
                        continue

                    point.x = x
                    point.y = y
                    point.z = z
                    point.ID = 15
                    point.faceCount = is_endpoint_check(neighborhood)
                    point.edt = edt[x,y,z]
                    simple_border_points.push_back(point)


cdef void find_simple_point_candidates_edges_0(pixel_type[:, :, ::1] img,
                                        npy_float32[:, :, ::1] edt,
                                        int curr_border,
                                        npy_intp [:,:] fLoop,
                                        vector[coordinate] & simple_border_points) nogil:

    cdef:
        cdef coordinate point

        pixel_type neighborhood[27]
        npy_intp x, y, z
        bint is_border_pt
        int euler_char = 0
        int[::1] Euler_LUT = LUT


    for x in range(fLoop[0,0], fLoop[0,1]):
        for y in range(fLoop[1,0], fLoop[1,1]):
            for z in range(fLoop[2,0], fLoop[2,1]):

                #img[x,y,z] = 155
                if img[x, y, z] != 1:
                    continue

                is_border_pt = (
                                curr_border == 1 and img[x+1,y  ,z  ] == 0 or  #E
                                curr_border == 2 and img[x  ,y-1,z  ] == 0 or  #S
                                curr_border == 3 and img[x  ,y+1,z  ] == 0 or  #N
                                curr_border == 5 and img[x  ,y  ,z+1] == 0     #U
                                )

                if not is_border_pt:
                    continue

                get_neighborhood_boundary_edges_0(img, x, y, z, neighborhood)
                if (not is_endpoint(neighborhood)):
                    euler_char = 0
                    euler_char = euler_char + is_Euler_invariant_Octant_5(neighborhood, Euler_LUT)
                    euler_char = euler_char + is_Euler_invariant_Octant_7(neighborhood, Euler_LUT)
                    if not euler_char == 0 or not is_simple_point(neighborhood):
                        continue

                    point.x = x
                    point.y = y
                    point.z = z
                    point.ID = 20
                    point.faceCount = is_endpoint_check(neighborhood)
                    point.edt = edt[x,y,z]
                    simple_border_points.push_back(point)


cdef void find_simple_point_candidates_edges_1(pixel_type[:, :, ::1] img,
                                        npy_float32[:, :, ::1] edt,
                                        int curr_border,
                                        npy_intp [:,:] fLoop,
                                        vector[coordinate] & simple_border_points) nogil:

    cdef:
        cdef coordinate point

        pixel_type neighborhood[27]
        npy_intp x, y, z
        bint is_border_pt
        int euler_char = 0
        int[::1] Euler_LUT = LUT


    for x in range(fLoop[0,0], fLoop[0,1]):
        for y in range(fLoop[1,0], fLoop[1,1]):
            for z in range(fLoop[2,0], fLoop[2,1]):

                #img[x,y,z] = 155
                if img[x, y, z] != 1:
                    continue

                is_border_pt = (
                                curr_border == 1 and img[x+1,y  ,z  ] == 0 or  #E
                                curr_border == 2 and img[x  ,y-1,z  ] == 0 or  #S
                                curr_border == 3 and img[x  ,y+1,z  ] == 0 or  #N
                                curr_border == 4 and img[x  ,y  ,z-1] == 0     #B
                                )

                if not is_border_pt:
                    continue

                get_neighborhood_boundary_edges_1(img, x, y, z, neighborhood)
                if (not is_endpoint(neighborhood)):
                    euler_char = 0
                    euler_char = euler_char + is_Euler_invariant_Octant_1(neighborhood, Euler_LUT)
                    euler_char = euler_char + is_Euler_invariant_Octant_3(neighborhood, Euler_LUT)
                    if not euler_char == 0 or not is_simple_point(neighborhood):
                        continue

                    point.x = x
                    point.y = y
                    point.z = z
                    point.ID = 21
                    point.faceCount = is_endpoint_check(neighborhood)
                    point.edt = edt[x,y,z]
                    simple_border_points.push_back(point)


cdef void find_simple_point_candidates_edges_2(pixel_type[:, :, ::1] img,
                                        npy_float32[:, :, ::1] edt,
                                        int curr_border,
                                        npy_intp [:,:] fLoop,
                                        vector[coordinate] & simple_border_points) nogil:

    cdef:
        cdef coordinate point

        pixel_type neighborhood[27]
        npy_intp x, y, z
        bint is_border_pt
        int euler_char = 0
        int[::1] Euler_LUT = LUT


    for x in range(fLoop[0,0], fLoop[0,1]):
        for y in range(fLoop[1,0], fLoop[1,1]):
            for z in range(fLoop[2,0], fLoop[2,1]):

                #img[x,y,z] = 155
                if img[x, y, z] != 1:
                    continue

                is_border_pt = (
                                curr_border == 1 and img[x+1,y  ,z  ] == 0 or  #E
                                curr_border == 3 and img[x  ,y+1,z  ] == 0 or  #N
                                curr_border == 4 and img[x  ,y  ,z-1] == 0 or  #B
                                curr_border == 5 and img[x  ,y  ,z+1] == 0     #U
                                )

                if not is_border_pt:
                    continue

                get_neighborhood_boundary_edges_2(img, x, y, z, neighborhood)
                if (not is_endpoint(neighborhood)):
                    euler_char = 0
                    euler_char = euler_char + is_Euler_invariant_Octant_1(neighborhood, Euler_LUT)
                    euler_char = euler_char + is_Euler_invariant_Octant_5(neighborhood, Euler_LUT)
                    if not euler_char == 0 or not is_simple_point(neighborhood):
                        continue

                    point.x = x
                    point.y = y
                    point.z = z
                    point.ID = 22
                    point.faceCount = is_endpoint_check(neighborhood)
                    point.edt = edt[x,y,z]
                    simple_border_points.push_back(point)


cdef void find_simple_point_candidates_edges_3(pixel_type[:, :, ::1] img,
                                        npy_float32[:, :, ::1] edt,
                                        int curr_border,
                                        npy_intp [:,:] fLoop,
                                        vector[coordinate] & simple_border_points) nogil:

    cdef:
        cdef coordinate point

        pixel_type neighborhood[27]
        npy_intp x, y, z
        bint is_border_pt
        int euler_char = 0
        int[::1] Euler_LUT = LUT


    for x in range(fLoop[0,0], fLoop[0,1]):
        for y in range(fLoop[1,0], fLoop[1,1]):
            for z in range(fLoop[2,0], fLoop[2,1]):

                #img[x,y,z] = 155
                if img[x, y, z] != 1:
                    continue

                is_border_pt = (
                                curr_border == 1 and img[x+1,y  ,z  ] == 0 or  #E
                                curr_border == 2 and img[x  ,y-1,z  ] == 0 or  #S
                                curr_border == 4 and img[x  ,y  ,z-1] == 0 or  #B
                                curr_border == 5 and img[x  ,y  ,z+1] == 0     #U
                                )

                if not is_border_pt:
                    continue

                get_neighborhood_boundary_edges_3(img, x, y, z, neighborhood)
                if (not is_endpoint(neighborhood)):
                    euler_char = 0
                    euler_char = euler_char + is_Euler_invariant_Octant_3(neighborhood, Euler_LUT)
                    euler_char = euler_char + is_Euler_invariant_Octant_7(neighborhood, Euler_LUT)
                    if not euler_char == 0 or not is_simple_point(neighborhood):
                        continue

                    point.x = x
                    point.y = y
                    point.z = z
                    point.ID = 23
                    point.faceCount = is_endpoint_check(neighborhood)
                    point.edt = edt[x,y,z]
                    simple_border_points.push_back(point)


cdef void find_simple_point_candidates_edges_4(pixel_type[:, :, ::1] img,
                                        npy_float32[:, :, ::1] edt,
                                        int curr_border,
                                        npy_intp [:,:] fLoop,
                                        vector[coordinate] & simple_border_points) nogil:

    cdef:
        cdef coordinate point

        pixel_type neighborhood[27]
        npy_intp x, y, z
        bint is_border_pt
        int euler_char = 0
        int[::1] Euler_LUT = LUT


    for x in range(fLoop[0,0], fLoop[0,1]):
        for y in range(fLoop[1,0], fLoop[1,1]):
            for z in range(fLoop[2,0], fLoop[2,1]):

                #img[x,y,z] = 155
                if img[x, y, z] != 1:
                    continue

                is_border_pt = (
                                curr_border == 0 and img[x-1,y  ,z  ] == 0 or  #W
                                curr_border == 2 and img[x  ,y-1,z  ] == 0 or  #S
                                curr_border == 3 and img[x  ,y+1,z  ] == 0 or  #N
                                curr_border == 5 and img[x  ,y  ,z+1] == 0     #U
                                )

                if not is_border_pt:
                    continue

                get_neighborhood_boundary_edges_4(img, x, y, z, neighborhood)
                if (not is_endpoint(neighborhood)):
                    euler_char = 0
                    euler_char = euler_char + is_Euler_invariant_Octant_4(neighborhood, Euler_LUT)
                    euler_char = euler_char + is_Euler_invariant_Octant_6(neighborhood, Euler_LUT)
                    if not euler_char == 0 or not is_simple_point(neighborhood):
                        continue

                    point.x = x
                    point.y = y
                    point.z = z
                    point.ID = 24
                    point.faceCount = is_endpoint_check(neighborhood)
                    point.edt = edt[x,y,z]
                    simple_border_points.push_back(point)


cdef void find_simple_point_candidates_edges_5(pixel_type[:, :, ::1] img,
                                        npy_float32[:, :, ::1] edt,
                                        int curr_border,
                                        npy_intp [:,:] fLoop,
                                        vector[coordinate] & simple_border_points) nogil:

    cdef:
        cdef coordinate point

        pixel_type neighborhood[27]
        npy_intp x, y, z
        bint is_border_pt
        int euler_char = 0
        int[::1] Euler_LUT = LUT


    for x in range(fLoop[0,0], fLoop[0,1]):
        for y in range(fLoop[1,0], fLoop[1,1]):
            for z in range(fLoop[2,0], fLoop[2,1]):

                #img[x,y,z] = 155
                if img[x, y, z] != 1:
                    continue

                is_border_pt = (
                                curr_border == 0 and img[x-1,y  ,z  ] == 0 or  #W
                                curr_border == 2 and img[x  ,y-1,z  ] == 0 or  #S
                                curr_border == 3 and img[x  ,y+1,z  ] == 0 or  #N
                                curr_border == 4 and img[x  ,y  ,z-1] == 0     #B
                                )

                if not is_border_pt:
                    continue

                get_neighborhood_boundary_edges_5(img, x, y, z, neighborhood)
                if (not is_endpoint(neighborhood)):
                    euler_char = 0
                    euler_char = euler_char + is_Euler_invariant_Octant_0(neighborhood, Euler_LUT)
                    euler_char = euler_char + is_Euler_invariant_Octant_2(neighborhood, Euler_LUT)
                    if not euler_char == 0 or not is_simple_point(neighborhood):
                        continue

                    point.x = x
                    point.y = y
                    point.z = z
                    point.ID = 25
                    point.faceCount = is_endpoint_check(neighborhood)
                    point.edt = edt[x,y,z]
                    simple_border_points.push_back(point)


cdef void find_simple_point_candidates_edges_6(pixel_type[:, :, ::1] img,
                                        npy_float32[:, :, ::1] edt,
                                        int curr_border,
                                        npy_intp [:,:] fLoop,
                                        vector[coordinate] & simple_border_points) nogil:

    cdef:
        cdef coordinate point

        pixel_type neighborhood[27]
        npy_intp x, y, z
        bint is_border_pt
        int euler_char = 0
        int[::1] Euler_LUT = LUT


    for x in range(fLoop[0,0], fLoop[0,1]):
        for y in range(fLoop[1,0], fLoop[1,1]):
            for z in range(fLoop[2,0], fLoop[2,1]):

                #img[x,y,z] = 155
                if img[x, y, z] != 1:
                    continue

                is_border_pt = (
                                curr_border == 0 and img[x-1,y  ,z  ] == 0 or  #W
                                curr_border == 3 and img[x  ,y+1,z  ] == 0 or  #N
                                curr_border == 4 and img[x  ,y  ,z-1] == 0 or  #B
                                curr_border == 5 and img[x  ,y  ,z+1] == 0     #U
                                )

                if not is_border_pt:
                    continue

                get_neighborhood_boundary_edges_6(img, x, y, z, neighborhood)
                if (not is_endpoint(neighborhood)):
                    euler_char = 0
                    euler_char = euler_char + is_Euler_invariant_Octant_0(neighborhood, Euler_LUT)
                    euler_char = euler_char + is_Euler_invariant_Octant_4(neighborhood, Euler_LUT)
                    if not euler_char == 0 or not is_simple_point(neighborhood):
                        continue

                    point.x = x
                    point.y = y
                    point.z = z
                    point.ID = 26
                    point.faceCount = is_endpoint_check(neighborhood)
                    point.edt = edt[x,y,z]
                    simple_border_points.push_back(point)


cdef void find_simple_point_candidates_edges_7(pixel_type[:, :, ::1] img,
                                        npy_float32[:, :, ::1] edt,
                                        int curr_border,
                                        npy_intp [:,:] fLoop,
                                        vector[coordinate] & simple_border_points) nogil:

    cdef:
        cdef coordinate point

        pixel_type neighborhood[27]
        npy_intp x, y, z
        bint is_border_pt
        int euler_char = 0
        int[::1] Euler_LUT = LUT


    for x in range(fLoop[0,0], fLoop[0,1]):
        for y in range(fLoop[1,0], fLoop[1,1]):
            for z in range(fLoop[2,0], fLoop[2,1]):

                #img[x,y,z] = 155
                if img[x, y, z] != 1:
                    continue

                is_border_pt = (
                                curr_border == 0 and img[x-1,y  ,z  ] == 0 or  #W
                                curr_border == 2 and img[x  ,y-1,z  ] == 0 or  #S
                                curr_border == 4 and img[x  ,y  ,z-1] == 0 or  #B
                                curr_border == 5 and img[x  ,y  ,z+1] == 0     #U
                                )

                if not is_border_pt:
                    continue

                get_neighborhood_boundary_edges_7(img, x, y, z, neighborhood)
                if (not is_endpoint(neighborhood)):
                    euler_char = 0
                    euler_char = euler_char + is_Euler_invariant_Octant_2(neighborhood, Euler_LUT)
                    euler_char = euler_char + is_Euler_invariant_Octant_6(neighborhood, Euler_LUT)
                    if not euler_char == 0 or not is_simple_point(neighborhood):
                        continue

                    point.x = x
                    point.y = y
                    point.z = z
                    point.ID = 27
                    point.faceCount = is_endpoint_check(neighborhood)
                    point.edt = edt[x,y,z]
                    simple_border_points.push_back(point)


cdef void find_simple_point_candidates_edges_8(pixel_type[:, :, ::1] img,
                                        npy_float32[:, :, ::1] edt,
                                        int curr_border,
                                        npy_intp [:,:] fLoop,
                                        vector[coordinate] & simple_border_points) nogil:

    cdef:
        cdef coordinate point

        pixel_type neighborhood[27]
        npy_intp x, y, z
        bint is_border_pt
        int euler_char = 0
        int[::1] Euler_LUT = LUT


    for x in range(fLoop[0,0], fLoop[0,1]):
        for y in range(fLoop[1,0], fLoop[1,1]):
            for z in range(fLoop[2,0], fLoop[2,1]):

                #img[x,y,z] = 155
                if img[x, y, z] != 1:
                    continue

                is_border_pt = (
                                curr_border == 0 and img[x-1,y  ,z  ] == 0 or  #W
                                curr_border == 1 and img[x+1,y  ,z  ] == 0 or  #E
                                curr_border == 3 and img[x  ,y+1,z  ] == 0 or  #N
                                curr_border == 5 and img[x  ,y  ,z+1] == 0     #U
                                )

                if not is_border_pt:
                    continue

                get_neighborhood_boundary_edges_8(img, x, y, z, neighborhood)
                if (not is_endpoint(neighborhood)):
                    euler_char = 0
                    euler_char = euler_char + is_Euler_invariant_Octant_4(neighborhood, Euler_LUT)
                    euler_char = euler_char + is_Euler_invariant_Octant_5(neighborhood, Euler_LUT)
                    if not euler_char == 0 or not is_simple_point(neighborhood):
                        continue

                    point.x = x
                    point.y = y
                    point.z = z
                    point.ID = 28
                    point.faceCount = is_endpoint_check(neighborhood)
                    point.edt = edt[x,y,z]
                    simple_border_points.push_back(point)


cdef void find_simple_point_candidates_edges_9(pixel_type[:, :, ::1] img,
                                        npy_float32[:, :, ::1] edt,
                                        int curr_border,
                                        npy_intp [:,:] fLoop,
                                        vector[coordinate] & simple_border_points) nogil:

    cdef:
        cdef coordinate point

        pixel_type neighborhood[27]
        npy_intp x, y, z
        bint is_border_pt
        int euler_char = 0
        int[::1] Euler_LUT = LUT


    for x in range(fLoop[0,0], fLoop[0,1]):
        for y in range(fLoop[1,0], fLoop[1,1]):
            for z in range(fLoop[2,0], fLoop[2,1]):

                #img[x,y,z] = 155
                if img[x, y, z] != 1:
                    continue

                is_border_pt = (
                                curr_border == 0 and img[x-1,y  ,z  ] == 0 or  #W
                                curr_border == 1 and img[x+1,y  ,z  ] == 0 or  #E
                                curr_border == 3 and img[x  ,y+1,z  ] == 0 or  #N
                                curr_border == 4 and img[x  ,y  ,z-1] == 0     #B
                                )

                if not is_border_pt:
                    continue

                get_neighborhood_boundary_edges_9(img, x, y, z, neighborhood)
                if (not is_endpoint(neighborhood)):
                    euler_char = 0
                    euler_char = euler_char + is_Euler_invariant_Octant_0(neighborhood, Euler_LUT)
                    euler_char = euler_char + is_Euler_invariant_Octant_1(neighborhood, Euler_LUT)
                    if not euler_char == 0 or not is_simple_point(neighborhood):
                        continue

                    point.x = x
                    point.y = y
                    point.z = z
                    point.ID = 29
                    point.faceCount = is_endpoint_check(neighborhood)
                    point.edt = edt[x,y,z]
                    simple_border_points.push_back(point)


cdef void find_simple_point_candidates_edges_10(pixel_type[:, :, ::1] img,
                                        npy_float32[:, :, ::1] edt,
                                        int curr_border,
                                        npy_intp [:,:] fLoop,
                                        vector[coordinate] & simple_border_points) nogil:

    cdef:
        cdef coordinate point

        pixel_type neighborhood[27]
        npy_intp x, y, z
        bint is_border_pt
        int euler_char = 0
        int[::1] Euler_LUT = LUT


    for x in range(fLoop[0,0], fLoop[0,1]):
        for y in range(fLoop[1,0], fLoop[1,1]):
            for z in range(fLoop[2,0], fLoop[2,1]):

                #img[x,y,z] = 155
                if img[x, y, z] != 1:
                    continue

                is_border_pt = (
                                curr_border == 0 and img[x-1,y  ,z  ] == 0 or  #W
                                curr_border == 1 and img[x+1,y  ,z  ] == 0 or  #E
                                curr_border == 2 and img[x  ,y-1,z  ] == 0 or  #S
                                curr_border == 5 and img[x  ,y  ,z+1] == 0     #U
                                )

                if not is_border_pt:
                    continue

                get_neighborhood_boundary_edges_10(img, x, y, z, neighborhood)
                if (not is_endpoint(neighborhood)):
                    euler_char = 0
                    euler_char = euler_char + is_Euler_invariant_Octant_6(neighborhood, Euler_LUT)
                    euler_char = euler_char + is_Euler_invariant_Octant_7(neighborhood, Euler_LUT)
                    if not euler_char == 0 or not is_simple_point(neighborhood):
                        continue

                    point.x = x
                    point.y = y
                    point.z = z
                    point.ID = 30
                    point.faceCount = is_endpoint_check(neighborhood)
                    point.edt = edt[x,y,z]
                    simple_border_points.push_back(point)


cdef void find_simple_point_candidates_edges_11(pixel_type[:, :, ::1] img,
                                        npy_float32[:, :, ::1] edt,
                                        int curr_border,
                                        npy_intp [:,:] fLoop,
                                        vector[coordinate] & simple_border_points) nogil:

    cdef:
        cdef coordinate point

        pixel_type neighborhood[27]
        npy_intp x, y, z
        bint is_border_pt
        int euler_char = 0
        int[::1] Euler_LUT = LUT


    for x in range(fLoop[0,0], fLoop[0,1]):
        for y in range(fLoop[1,0], fLoop[1,1]):
            for z in range(fLoop[2,0], fLoop[2,1]):

                #img[x,y,z] = 155
                if img[x, y, z] != 1:
                    continue

                is_border_pt = (
                                curr_border == 0 and img[x-1,y  ,z  ] == 0 or  #W
                                curr_border == 1 and img[x+1,y  ,z  ] == 0 or  #E
                                curr_border == 2 and img[x  ,y-1,z  ] == 0 or  #S
                                curr_border == 4 and img[x  ,y  ,z-1] == 0     #B
                                )

                if not is_border_pt:
                    continue

                get_neighborhood_boundary_edges_11(img, x, y, z, neighborhood)
                if (not is_endpoint(neighborhood)):
                    euler_char = 0
                    euler_char = euler_char + is_Euler_invariant_Octant_2(neighborhood, Euler_LUT)
                    euler_char = euler_char + is_Euler_invariant_Octant_3(neighborhood, Euler_LUT)
                    if not euler_char == 0 or not is_simple_point(neighborhood):
                        continue

                    point.x = x
                    point.y = y
                    point.z = z
                    point.ID = 31
                    point.faceCount = is_endpoint_check(neighborhood)
                    point.edt = edt[x,y,z]
                    simple_border_points.push_back(point)


cdef void find_simple_point_candidates_corners_0(pixel_type[:, :, ::1] img,
                                        npy_float32[:, :, ::1] edt,
                                        int curr_border,
                                        npy_intp [:,:] fLoop,
                                        vector[coordinate] & simple_border_points) nogil:

    cdef:
        cdef coordinate point

        pixel_type neighborhood[27]
        npy_intp x, y, z
        bint is_border_pt
        int euler_char = 0
        int[::1] Euler_LUT = LUT


    for x in range(fLoop[0,0], fLoop[0,1]):
        for y in range(fLoop[1,0], fLoop[1,1]):
            for z in range(fLoop[2,0], fLoop[2,1]):

                #img[x,y,z] = 155
                if img[x, y, z] == 1:

                    is_border_pt = (
                                    curr_border == 1 and img[x+1,y  ,z  ] == 0 or  #E
                                    curr_border == 3 and img[x  ,y+1,z  ] == 0 or  #N
                                    curr_border == 5 and img[x  ,y  ,z+1] == 0     #U
                                    )

                    if is_border_pt:

                        get_neighborhood_boundary_corners_0(img, x, y, z, neighborhood)
                        if (not is_endpoint(neighborhood)):
                            euler_char = 0
                            euler_char = euler_char + is_Euler_invariant_Octant_5(neighborhood, Euler_LUT)
                            if euler_char==0 or is_simple_point(neighborhood):
                                point.x = x
                                point.y = y
                                point.z = z
                                point.ID = 40
                                point.faceCount = is_endpoint_check(neighborhood)
                                point.edt = edt[x,y,z]
                                simple_border_points.push_back(point)



cdef void find_simple_point_candidates_corners_1(pixel_type[:, :, ::1] img,
                                        npy_float32[:, :, ::1] edt,
                                        int curr_border,
                                        npy_intp [:,:] fLoop,
                                        vector[coordinate] & simple_border_points) nogil:

    cdef:
        cdef coordinate point

        pixel_type neighborhood[27]
        npy_intp x, y, z
        bint is_border_pt
        int euler_char = 0
        int[::1] Euler_LUT = LUT


    for x in range(fLoop[0,0], fLoop[0,1]):
        for y in range(fLoop[1,0], fLoop[1,1]):
            for z in range(fLoop[2,0], fLoop[2,1]):

                #img[x,y,z] = 155
                if img[x, y, z] == 1:

                    is_border_pt = (
                                    curr_border == 1 and img[x+1,y  ,z  ] == 0 or  #E
                                    curr_border == 3 and img[x  ,y+1,z  ] == 0 or  #N
                                    curr_border == 4 and img[x  ,y  ,z-1] == 0     #B
                                    )

                    if is_border_pt:

                        get_neighborhood_boundary_corners_1(img, x, y, z, neighborhood)
                        if (not is_endpoint(neighborhood)):
                            euler_char = 0
                            euler_char = euler_char + is_Euler_invariant_Octant_1(neighborhood, Euler_LUT)
                            if euler_char==0 or is_simple_point(neighborhood):
                                point.x = x
                                point.y = y
                                point.z = z
                                point.ID = 41
                                point.faceCount = is_endpoint_check(neighborhood)
                                point.edt = edt[x,y,z]
                                simple_border_points.push_back(point)



cdef void find_simple_point_candidates_corners_2(pixel_type[:, :, ::1] img,
                                        npy_float32[:, :, ::1] edt,
                                        int curr_border,
                                        npy_intp [:,:] fLoop,
                                        vector[coordinate] & simple_border_points) nogil:

    cdef:
        cdef coordinate point

        pixel_type neighborhood[27]
        npy_intp x, y, z
        bint is_border_pt
        int euler_char = 0
        int[::1] Euler_LUT = LUT


    for x in range(fLoop[0,0], fLoop[0,1]):
        for y in range(fLoop[1,0], fLoop[1,1]):
            for z in range(fLoop[2,0], fLoop[2,1]):

                #img[x,y,z] = 155
                if img[x, y, z] == 1:

                    is_border_pt = (
                                    curr_border == 1 and img[x+1,y  ,z  ] == 0 or  #E
                                    curr_border == 2 and img[x  ,y-1,z  ] == 0 or  #S
                                    curr_border == 5 and img[x  ,y  ,z+1] == 0     #U
                                    )

                    if is_border_pt:

                        get_neighborhood_boundary_corners_2(img, x, y, z, neighborhood)
                        if (not is_endpoint(neighborhood)):
                            euler_char = 0
                            euler_char = euler_char + is_Euler_invariant_Octant_7(neighborhood, Euler_LUT)
                            if euler_char==0 or is_simple_point(neighborhood):
                                point.x = x
                                point.y = y
                                point.z = z
                                point.ID = 42
                                point.faceCount = is_endpoint_check(neighborhood)
                                point.edt = edt[x,y,z]
                                simple_border_points.push_back(point)



cdef void find_simple_point_candidates_corners_3(pixel_type[:, :, ::1] img,
                                        npy_float32[:, :, ::1] edt,
                                        int curr_border,
                                        npy_intp [:,:] fLoop,
                                        vector[coordinate] & simple_border_points) nogil:

    cdef:
        cdef coordinate point

        pixel_type neighborhood[27]
        npy_intp x, y, z
        bint is_border_pt
        int euler_char = 0
        int[::1] Euler_LUT = LUT


    for x in range(fLoop[0,0], fLoop[0,1]):
        for y in range(fLoop[1,0], fLoop[1,1]):
            for z in range(fLoop[2,0], fLoop[2,1]):

                #img[x,y,z] = 155
                if img[x, y, z] == 1:

                    is_border_pt = (
                                    curr_border == 1 and img[x+1,y  ,z  ] == 0 or  #E
                                    curr_border == 2 and img[x  ,y-1,z  ] == 0 or  #S
                                    curr_border == 4 and img[x  ,y  ,z-1] == 0     #B
                                    )

                    if is_border_pt:

                        get_neighborhood_boundary_corners_3(img, x, y, z, neighborhood)
                        if (not is_endpoint(neighborhood)):
                            euler_char = 0
                            euler_char = euler_char + is_Euler_invariant_Octant_3(neighborhood, Euler_LUT)
                            if euler_char==0 or is_simple_point(neighborhood):
                                point.x = x
                                point.y = y
                                point.z = z
                                point.ID = 43
                                point.faceCount = is_endpoint_check(neighborhood)
                                point.edt = edt[x,y,z]
                                simple_border_points.push_back(point)



cdef void find_simple_point_candidates_corners_4(pixel_type[:, :, ::1] img,
                                        npy_float32[:, :, ::1] edt,
                                        int curr_border,
                                        npy_intp [:,:] fLoop,
                                        vector[coordinate] & simple_border_points) nogil:

    cdef:
        cdef coordinate point

        pixel_type neighborhood[27]
        npy_intp x, y, z
        bint is_border_pt
        int euler_char = 0
        int[::1] Euler_LUT = LUT


    for x in range(fLoop[0,0], fLoop[0,1]):
        for y in range(fLoop[1,0], fLoop[1,1]):
            for z in range(fLoop[2,0], fLoop[2,1]):

                #img[x,y,z] = 155
                if img[x, y, z] == 1:

                    is_border_pt = (
                                    curr_border == 0 and img[x-1,y  ,z  ] == 0 or  #W
                                    curr_border == 3 and img[x  ,y+1,z  ] == 0 or  #N
                                    curr_border == 5 and img[x  ,y  ,z+1] == 0     #U
                                    )

                    if is_border_pt:

                        get_neighborhood_boundary_corners_4(img, x, y, z, neighborhood)
                        if (not is_endpoint(neighborhood)):
                            euler_char = 0
                            euler_char = euler_char + is_Euler_invariant_Octant_4(neighborhood, Euler_LUT)
                            if euler_char==0 or is_simple_point(neighborhood):
                                point.x = x
                                point.y = y
                                point.z = z
                                point.ID = 44
                                point.faceCount = is_endpoint_check(neighborhood)
                                point.edt = edt[x,y,z]
                                simple_border_points.push_back(point)



cdef void find_simple_point_candidates_corners_5(pixel_type[:, :, ::1] img,
                                        npy_float32[:, :, ::1] edt,
                                        int curr_border,
                                        npy_intp [:,:] fLoop,
                                        vector[coordinate] & simple_border_points) nogil:

    cdef:
        cdef coordinate point

        pixel_type neighborhood[27]
        npy_intp x, y, z
        bint is_border_pt
        int euler_char = 0
        int[::1] Euler_LUT = LUT


    for x in range(fLoop[0,0], fLoop[0,1]):
        for y in range(fLoop[1,0], fLoop[1,1]):
            for z in range(fLoop[2,0], fLoop[2,1]):

                #img[x,y,z] = 155
                if img[x, y, z] == 1:

                    is_border_pt = (
                                    curr_border == 0 and img[x-1,y  ,z  ] == 0 or  #W
                                    curr_border == 3 and img[x  ,y+1,z  ] == 0 or  #N
                                    curr_border == 4 and img[x  ,y  ,z-1] == 0     #B
                                    )

                    if is_border_pt:

                        get_neighborhood_boundary_corners_5(img, x, y, z, neighborhood)
                        if (not is_endpoint(neighborhood)):
                            euler_char = 0
                            euler_char = euler_char + is_Euler_invariant_Octant_0(neighborhood, Euler_LUT)
                            if euler_char==0 or is_simple_point(neighborhood):
                                point.x = x
                                point.y = y
                                point.z = z
                                point.ID = 45
                                point.faceCount = is_endpoint_check(neighborhood)
                                point.edt = edt[x,y,z]
                                simple_border_points.push_back(point)



cdef void find_simple_point_candidates_corners_6(pixel_type[:, :, ::1] img,
                                        npy_float32[:, :, ::1] edt,
                                        int curr_border,
                                        npy_intp [:,:] fLoop,
                                        vector[coordinate] & simple_border_points) nogil:

    cdef:
        cdef coordinate point

        pixel_type neighborhood[27]
        npy_intp x, y, z
        bint is_border_pt
        int euler_char = 0
        int[::1] Euler_LUT = LUT


    for x in range(fLoop[0,0], fLoop[0,1]):
        for y in range(fLoop[1,0], fLoop[1,1]):
            for z in range(fLoop[2,0], fLoop[2,1]):

                #img[x,y,z] = 155
                if img[x, y, z] == 1:

                    is_border_pt = (
                                    curr_border == 0 and img[x-1,y  ,z  ] == 0 or  #W
                                    curr_border == 2 and img[x  ,y-1,z  ] == 0 or  #S
                                    curr_border == 5 and img[x  ,y  ,z+1] == 0     #U
                                    )

                    if is_border_pt:

                        get_neighborhood_boundary_corners_6(img, x, y, z, neighborhood)
                        if (not is_endpoint(neighborhood)):
                            euler_char = 0
                            euler_char = euler_char + is_Euler_invariant_Octant_6(neighborhood, Euler_LUT)
                            if euler_char==0 or is_simple_point(neighborhood):
                                point.x = x
                                point.y = y
                                point.z = z
                                point.ID = 46
                                point.faceCount = is_endpoint_check(neighborhood)
                                point.edt = edt[x,y,z]
                                simple_border_points.push_back(point)



cdef void find_simple_point_candidates_corners_7(pixel_type[:, :, ::1] img,
                                        npy_float32[:, :, ::1] edt,
                                        int curr_border,
                                        npy_intp [:,:] fLoop,
                                        vector[coordinate] & simple_border_points) nogil:

    cdef:
        cdef coordinate point

        pixel_type neighborhood[27]
        npy_intp x, y, z
        bint is_border_pt
        int euler_char = 0
        int[::1] Euler_LUT = LUT


    for x in range(fLoop[0,0], fLoop[0,1]):
        for y in range(fLoop[1,0], fLoop[1,1]):
            for z in range(fLoop[2,0], fLoop[2,1]):

                #img[x,y,z] = 155
                if img[x, y, z] == 1:

                    is_border_pt = (
                                    curr_border == 0 and img[x-1,y  ,z  ] == 0 or  #W
                                    curr_border == 2 and img[x  ,y-1,z  ] == 0 or  #S
                                    curr_border == 4 and img[x  ,y  ,z-1] == 0     #B
                                    )

                    if is_border_pt:

                        get_neighborhood_boundary_corners_7(img, x, y, z, neighborhood)
                        if (not is_endpoint(neighborhood)):
                            euler_char = 0
                            euler_char = euler_char + is_Euler_invariant_Octant_2(neighborhood, Euler_LUT)
                            if euler_char==0 or is_simple_point(neighborhood):
                                point.x = x
                                point.y = y
                                point.z = z
                                point.ID = 47
                                point.faceCount = is_endpoint_check(neighborhood)
                                point.edt = edt[x,y,z]
                                simple_border_points.push_back(point)


cdef void find_simple_point_candidates_TEST(pixel_type[:, :, ::1] img,
                                        npy_float32[:, :, ::1] edt,
                                        int fErode,
                                        npy_intp [:,:] fLoop,
                                        vector[coordinate] & simple_border_points) nogil:

    cdef:
        cdef coordinate point

        pixel_type neighborhood[27]
        npy_intp x, y, z
        bint is_border_pt
        int euler_char = 0
        int[::1] Euler_LUT = LUT


    for x in range(fLoop[0,0], fLoop[0,1]):
        for y in range(fLoop[1,0], fLoop[1,1]):
            for z in range(fLoop[2,0], fLoop[2,1]):

                #img[x,y,z] = 255
                if img[x, y, z] != 1:
                    continue

                is_border_pt = (
                                fErode == 0 and img[x+1,y  ,z  ] == 0 or
                                fErode == 1 and img[x-1,y  ,z  ] == 0 or
                                fErode == 2 and img[x  ,y+1,z  ] == 0 or
                                fErode == 3 and img[x  ,y-1,z  ] == 0 or
                                fErode == 4 and img[x  ,y  ,z+1] == 0 or
                                fErode == 5 and img[x  ,y  ,z-1] == 0    
                            )

                if not is_border_pt:
                    continue

                get_neighborhood(img, x, y, z, neighborhood)
                #   Check conditions 2 and 3
                if (is_endpoint(neighborhood) or
                    not is_Euler_invariant(neighborhood, Euler_LUT) or
                    not is_simple_point(neighborhood)):
                    continue

                point.x = x
                point.y = y
                point.z = z
                point.ID = 0
                point.faceCount = is_endpoint_check(neighborhood)
                point.edt = edt[x,y,z]
                simple_border_points.push_back(point)

cdef void find_simple_point_candidates_TEST2(pixel_type[:, :, ::1] img,
                                        npy_float32[:, :, ::1] edt,
                                        int fErode,
                                        npy_intp [:,:] fLoop,
                                        vector[coordinate] & simple_border_points) nogil:

    cdef:
        cdef coordinate point

        pixel_type neighborhood[27]
        npy_intp x, y, z
        bint is_border_pt
        int euler_char = 0
        int[::1] Euler_LUT = LUT


    for x in range(fLoop[0,0], fLoop[0,1]):
        for y in range(fLoop[1,0], fLoop[1,1]):
            for z in range(fLoop[2,0], fLoop[2,1]):

                #img[x,y,z] = 255
                if img[x, y, z] != 1:
                    continue

                is_border_pt = (
                                fErode == 0 and img[x+1,y  ,z  ] == 0 or
                                fErode == 1 and img[x-1,y  ,z  ] == 0 or
                                fErode == 2 and img[x  ,y+1,z  ] == 0 or
                                fErode == 3 and img[x  ,y-1,z  ] == 0 or
                                fErode == 4 and img[x  ,y  ,z+1] == 0 or
                                fErode == 5 and img[x  ,y  ,z-1] == 0    
                            )

                if not is_border_pt:
                    continue

                get_neighborhood(img, x, y, z, neighborhood)
                #   Check conditions 2 and 3
                if (is_endpoint(neighborhood) or
                    not is_Euler_invariant(neighborhood, Euler_LUT) or
                    not is_simple_point(neighborhood)):
                    continue

                point.x = x
                point.y = y
                point.z = z
                point.ID = 0
                point.faceCount = is_endpoint_check(neighborhood)
                point.edt = edt[x,y,z]
                simple_border_points.push_back(point)