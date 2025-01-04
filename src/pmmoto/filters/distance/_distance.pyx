# cython: profile=True
# cython: linetrace=True
# cython: boundscheck=True
# cython: wraparound=False
# cython: cdivision=True

from libcpp cimport bool
cimport numpy as np
np.import_array()

import numpy as np
from numpy cimport int8_t, int16_t, int32_t, int64_t
from numpy cimport uint8_t, uint16_t, uint32_t, uint64_t
from libcpp.vector cimport vector

from pmmoto.core import _voxels
ctypedef fused UINT:
    uint8_t
    uint16_t
    uint32_t
    uint64_t

ctypedef fused INT:
    int8_t
    int16_t
    int32_t
    int64_t

ctypedef fused NUMBER:
    UINT
    INT
    float
    double

def _tofinite(
    float[:, :, :] img_out,
    size_t size
):
    """
    Wrapper Function to to_finite
    """
    to_finite(
        <float*>&img_out[0, 0, 0],
        size
    )


def _tofinite_2d(
    float[:, :] img_out,
    size_t size
):
    """
    Wrapper Function to to_finite
    """
    to_finite(
        <float*>&img_out[0, 0],
        size
    )



def get_parabolic_envelope(img, dimension, lower_hull = None, upper_hull=None):
    """
    Determine the parabolic envelop along the specified dimension
    """
    # Identify the two dimensions to iterate over
    dims = {0, 1, 2}
    dims.remove(dimension)
    dim1, dim2 = dims

    # Dimensions of the output index array
    cdef size_t s1 = img.shape[dim1]
    cdef size_t s2 = img.shape[dim2]

    cdef int x
    cdef int y

    if lower_hull is None:
        lower_hull = [[] for _ in range(s1*s2)]

    if upper_hull is None:
        upper_hull = [[] for _ in range(s1*s2)]

    location = {}
    for x in range(s1):
        location[dim1] = x
        for y in range(s2):
            location[dim2] = y
            _get_parabolic_envelope(
                img,
                location,
                dimension,
                lower_hull[x*s1 + y],
                upper_hull[x*s1 + y],
            )

def get_parabolic_envelope_2d(
    img,
    dimension,
    lower_hull,
    upper_hull,
):
    """
    Determine the parabolic envelop along the specified dimension
    """
    # Identify the dimension to iterate over
    other_dim = [0, 1]
    other_dim.remove(dimension)
    other_dim = other_dim[0]

    # Dimensions of the output index array
    cdef size_t s = img.shape[other_dim]

    if lower_hull is None:
        lower_hull = [[] for _ in range(s)]
    if upper_hull is None:
        upper_hull = [[] for _ in range(s)]

    location = {}
    for x in range(s):
        location[other_dim] = x
        _get_parabolic_envelope_2d(
            img,
            location,
            dimension,
            lower_hull[x],
            upper_hull[x],
        )


def _get_parabolic_envelope(
    float[:, :, :] img,
    dict location,
    int dimension,
    vector[Hull] lower_hull,
    vector[Hull] upper_hull
):
    """
    """

    cdef uint64_t end = img.shape[dimension]
    cdef int[3] start = _voxels.get_start_indices(dimension, location)
    cdef int stride = _voxels.normalized_strides(img)[dimension]

    # Call the low-level function to get the parabolic envelope
    _determine_boundary_parabolic_envelope(
        <float*>&img[start[0], start[1], start[2]],
        end,
        stride,
        lower_hull,
        upper_hull
    )


def _get_parabolic_envelope_2d(
    float[:, :] img,
    dict location,
    int dimension,
    vector[Hull] lower_hull,
    vector[Hull] upper_hull,
):
    """
    """

    cdef uint64_t end = img.shape[dimension]
    cdef int[2] start = _voxels.get_start_indices_2d(dimension, location)
    cdef int stride = _voxels.normalized_strides(img)[dimension]

    # Call the low-level function to get the parabolic envelope
    _determine_boundary_parabolic_envelope(
        <float*>&img[start[0], start[1]],
        end,
        stride,
        lower_hull,
        upper_hull
    )


def determine_parabolic_envelope_1d(
    float[:] img,
    uint64_t start,
    uint64_t end,
    vector[Hull] lower_hull,
    vector[Hull] upper_hull,
):
    """
    """
    # Call the low-level function to get the nearest boundary index
    _determine_boundary_parabolic_envelope(
        <float*>&img[start],
        end,
        1,  # stride
        lower_hull,
        upper_hull,
    )


def get_initial_envelope_correctors(img, dimension):
    """
    Determine the nearest solid (or phase change of multilabel) index
    for 3d array
    """
    dims = {0, 1, 2}
    dims.remove(dimension)
    dim1, dim2 = dims

    # Collect nearest solid (or phase change of multi-label) index
    nearest_index = np.zeros([img.shape[dim1], img.shape[dim2], 2])
    nearest_index[:, :, 0] = _voxels.get_nearest_boundary_index_face(
        img=img,
        dimension=dimension,
        label=0,
        forward=True,
    ).astype(np.float32)
    nearest_index[:, :, 1] = _voxels.get_nearest_boundary_index_face(
        img=img,
        dimension=dimension,
        label=0,
        forward=False,
    ).astype(np.float32)

    # initialize correctors
    lower_correctors = np.zeros([img.shape[dim1], img.shape[dim2]])
    upper_correctors = np.zeros([img.shape[dim1], img.shape[dim2]])
    # correct indexes
    lower_correctors = np.where(
        nearest_index[:, :, 1] != -1,
        img.shape[dimension] - nearest_index[:, :, 1],
        np.inf)
    upper_correctors = np.where(
        nearest_index[:, :, 0] != -1,
        nearest_index[:, :, 0] + 1,
        np.inf)

    return lower_correctors, upper_correctors


def get_initial_envelope_correctors_2d(img, dimension):
    """
    Determine the nearest solid (or phase change of multilabel) index
    for 2d array
    """
    other_dim = [0, 1]
    other_dim.remove(dimension)
    other_dim = other_dim[0]

    # Collect nearest solid (or phase change of multi-label) index
    nearest_index = np.zeros([img.shape[other_dim], 2])
    nearest_index[:, 0] = _voxels.get_nearest_boundary_index_face_2d(
        img=img,
        dimension=dimension,
        label=0,
        forward=True,
    ).astype(np.float32)
    nearest_index[:, 1] = _voxels.get_nearest_boundary_index_face_2d(
        img=img,
        dimension=dimension,
        label=0,
        forward=False,
    ).astype(np.float32)

    # initialize correctors
    lower_correctors = np.zeros(img.shape[other_dim])
    upper_correctors = np.zeros(img.shape[other_dim])

    # correct indexes
    lower_correctors = np.where(
        nearest_index[:, 1] != -1,
        img.shape[dimension] - nearest_index[:, 1],
        np.inf)
    upper_correctors = np.where(
        nearest_index[:, 0] != -1,
        nearest_index[:, 0] + 1,
        np.inf)

    return lower_correctors, upper_correctors


def get_initial_envelope(
    img,
    img_out,
    dimension,
    lower_boundary = None,
    upper_boundary = None
):
    """
    """
    # Identify the two dimensions to iterate over
    dims = {0, 1, 2}
    dims.remove(dimension)
    dim1, dim2 = dims

    # Dimensions of the output index array
    cdef size_t s1 = img.shape[dim1]
    cdef size_t s2 = img.shape[dim2]
    cdef size_t sdim = img.shape[dimension]

    if lower_boundary is None:
        lower_boundary = np.full((s1, s2), np.inf)
    if upper_boundary is None:
        upper_boundary = np.full((s1, s2), np.inf)

    location = {}
    for x in range(s1):
        location[dim1] = x
        for y in range(s2):
            location[dim2] = y
            _get_initial_envelope(
                img,
                img_out,
                location,
                dimension,
                lower_boundary[x, y],
                upper_boundary[x, y]
            )

    _tofinite(img_out, s1*s2*sdim)

    return img_out


def get_initial_envelope_2d(
    img,
    img_out,
    dimension,
    lower_boundary = None,
    upper_boundary = None
):
    """
    """
    # Identify the dimension to iterate over
    other_dim = [0, 1]
    other_dim.remove(dimension)
    other_dim = other_dim[0]

    # Dimensions of the output index array
    cdef size_t s = img.shape[other_dim]
    cdef size_t sdim = img.shape[dimension]

    if lower_boundary is None:
        lower_boundary = np.full(s, np.inf)
    if upper_boundary is None:
        upper_boundary = np.full(s, np.inf)

    location = {}
    for x in range(s):
        location[other_dim] = x
        _get_initial_envelope_2d(
            img,
            img_out,
            location,
            dimension,
            lower_boundary[x],
            upper_boundary[x]
        )

    _tofinite_2d(img_out, s*sdim)

    return img_out


def determine_initial_envelope_1d(
    uint8_t[:] img,
    uint64_t start,
    int size,
    float lower_corrector,
    float upper_corrector
):
    """
    Perform a distance transform to compute an initial envelope in 1D.

    This function computes a distance transform on a 1D image array using
    the `squared_edt_1d_multi_seg_new` method. It applies corrections to
    adjust the lower and upper bounds during the transformation.

    Parameters:
        img (uint8_t[:]): A 1D array representing the input image data.
            Each element is an unsigned 8-bit integer.
        start (uint64_t): The starting index in the image where the
            transform will begin.
        size (int): The size of the segment to process, starting from the
            `start` index.
        lower_corrector (float): A correction factor applied to adjust the
            lower bound of the distance transform.
        upper_corrector (float): A correction factor applied to adjust the
            upper bound of the distance transform.

    Returns:
        np.ndarray[float, ndim=1]: A 1D array of type `float32` containing
            the computed distances for the specified segment. The output
            array has the same size as the input image, but only the
            specified range is updated with the transform results.

    Notes:
        - The function uses Cython for performance, leveraging pointers
          to directly manipulate memory.
        - The `squared_edt_1d_multi_seg_new` function is assumed to handle
          the core computation, efficiently updating the `output` array
          in place.
    """
    cdef size_t voxels = img.size
    cdef np.ndarray[float, ndim=1] output = np.zeros((voxels, ), dtype=np.float32)
    cdef float[:] outputview = output

    squared_edt_1d_multi_seg_new(
        <uint8_t*>&img[start],
        <float*>&outputview[start],
        size,
        1,
        1,
        lower_corrector,
        upper_corrector
    )

    return output


def _get_initial_envelope(
    uint8_t[:, :, :] img,
    float[:, :, :] img_out,
    dict location,
    int dimension,
    float lower_corrector,
    float upper_corrector
):
    """
    Perform a distance transform to compute an initial envelope in 3D.

    This function computes a distance transform on a 3D image array,
    applying corrections to adjust the lower and upper bounds during the
    transformation process.

    Parameters:
        img (UINT[:,:,:]): A 3D array representing the input image data, where each
            element is an unsigned integer.
        start (uint64_t): The starting index in the flattened 1D representation of
            the image where the transform will begin.
        size (int): The size of the segment to process in the flattened array, starting
            from the `start` index.
        lower_corrector (float): A correction factor applied to adjust the lower
            bound of the distance transform.
        upper_corrector (float): A correction factor applied to adjust the upper
            bound of the distance transform.

    Returns:
        np.ndarray[float, ndim=3]: A 3D array of type `float32` containing the computed
            distances for the specified segment.

    Notes:
        - The input 3D image is treated as a flattened 1D array for the purposes of
          this function, with indices mapped accordingly.
        - The function initializes an output array of the same total size as the input
          image, ensuring that all elements are properly handled.
        - This function serves as a wrapper for the actual distance transform logic,
          which is expected to be implemented elsewhere.
    """

    # Determine the size of the output slice
    cdef size_t _size = img.shape[dimension]

    cdef int[3] start = _voxels.get_start_indices(dimension, location)
    cdef int stride = _voxels.normalized_strides(img)[dimension]

    squared_edt_1d_multi_seg_new(
        <uint8_t*>&img[start[0], start[1], start[2]],
        <float*>&img_out[start[0], start[1], start[2]],
        _size,
        stride,
        1,
        lower_corrector,
        upper_corrector
    )

    return img_out


def _get_initial_envelope_2d(
    uint8_t[:, :] img,
    float[:, :] img_out,
    dict location,
    int dimension,
    float lower_corrector,
    float upper_corrector
):
    """
    """

    # Determine the size of the output slice
    cdef size_t _size = img.shape[dimension]

    cdef int[2] start = _voxels.get_start_indices_2d(dimension, location)
    cdef int stride = _voxels.normalized_strides(img)[dimension]

    squared_edt_1d_multi_seg_new(
        <uint8_t*>&img[start[0], start[1]],
        <float*>&img_out[start[0], start[1]],
        _size,
        stride,
        1,
        lower_corrector,
        upper_corrector
    )

    return img_out


def get_boundary_hull(
    float[:, :, :] img,
    int dimension,
    uint8_t num_hull,
    bool left = True
):
    """
    Determine the initial and last parabola vertex and value for a given 3d array
    """
    # Identify the two dimensions to iterate over
    dims = {0, 1, 2}
    dims.remove(dimension)
    dim1, dim2 = dims


    # Determine the size of the output slice
    cdef size_t _size = img.shape[dimension]
    cdef size_t s1 = img.shape[dim1]
    cdef size_t s2 = img.shape[dim2]

    cdef vector[vector[Hull]] hull
    hull = vector[vector[Hull]]()
    
    # Resize the outer vector and initialize each inner vector
    for _ in range(s1 * s2):
        hull.push_back(vector[Hull]())

    cdef int[3] start
    cdef int stride = _voxels.normalized_strides(img)[dimension]
    location = {}
    for x in range(s1):
        location[dim1] = x
        for y in range(s2):
            location[dim2] = y
            start = _voxels.get_start_indices(dimension, location)
            hull[x*s1 + y] = return_boundary_hull(
                <float*>&img[start[0], start[1], start[2]],
                _size,
                stride,
                num_hull,
                left
                )

    return hull


def get_boundary_hull_2d(
    float[:, :] img,
    int dimension,
    uint8_t num_hull
):
    """
    Determine the initial and last parabola vertex and value for a given 2d array
    """
    other_dim = [0, 1]
    other_dim.remove(dimension)
    other_dim = other_dim[0]

    # Determine the size of the output slice
    cdef size_t _size = img.shape[dimension]
    cdef size_t s = img.shape[other_dim]

    cdef vector[vector[Hull]] l_hull = vector[vector[Hull]](s)
    cdef vector[vector[Hull]] r_hull = vector[vector[Hull]](s)

    cdef int[2] start
    cdef int stride = _voxels.normalized_strides(img)[dimension]
    location = {}
    for n in range(s):
        location[other_dim] = n
        start = _voxels.get_start_indices_2d(dimension, location)
        l_hull[n] = return_boundary_hull(
            <float*>&img[start[0], start[1]],
            _size,
            stride,
            num_hull,
            True
        )

        r_hull[n] = return_boundary_hull(
            <float*>&img[start[0], start[1]],
            _size,
            stride,
            num_hull,
            False
        )

    return l_hull, r_hull


def get_boundary_hull_1d(
    float[:] img,
    uint64_t start,
    uint64_t end,
    uint8_t num_hull,
    bool left = True
):
    """
    Determine the initial and last parabola vertex and value for a given 1d array
    """
    cdef vector[Hull] hull

    hull = return_boundary_hull(
        <float*>&img[start],
        end,
        1,
        num_hull,
        left
    )

    return hull
