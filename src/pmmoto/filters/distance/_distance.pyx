# cython: profile=True
# cython: linetrace=False
# cython: boundscheck=False
# cython: wraparound=False
# cython: cdivision=True


from pmmoto.core import _voxels

import cython
from cython cimport floating
from cpython cimport array 
from libcpp cimport bool
cimport numpy as np
np.import_array()

import numpy as np
from libcpp.vector cimport vector

from numpy cimport int8_t, int16_t, int32_t, int64_t
from numpy cimport uint8_t, uint16_t, uint32_t, uint64_t

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

cdef extern from "_distance.hpp":
    cdef inline void to_finite(
        float *f, 
        const size_t voxels
    ) nogil

    cdef void squared_edt_1d_multi_seg_new[T](
        T *labels,
        float *dest,
        int n,
        int stride,
        float anisotropy,
        const float lower_corrector,
        const float upper_corrector
    ) nogil

    cdef void _determine_boundary_parabolic_envelope(
            float *img,
            const int n,
            const long int stride,
            const int lower_vertex,
            const float lower_f, 
            const int upper_vertex, 
            const float upper_f
        ) nogil

    cdef void return_boundary_hull(
        float *img, 
        const int n,
        const long int stride,
        int &hull_vertex,
        float &hull_f,
        bool left
        ) nogil

def _tofinite(
    float[:,:,:] img_out,
    size_t size
):
    """
    Wrapper Function to to_finite
    """
    to_finite(
        <float*>&img_out[0,0,0],
        size
    )

def _tofinite_2d(
    float[:,:] img_out,
    size_t size
):
    """
    Wrapper Function to to_finite 
    """
    to_finite(
        <float*>&img_out[0,0],
        size
    )


def get_parabolic_envelope(img,dimension,boundary_vertices = None,boundary_f=None):
    """
    Determine the parabolic envelop along the specified dimension
    """
    # Identify the two dimensions to iterate over
    dims = {0, 1, 2}
    dims.remove(dimension)
    dim1, dim2 = dims

    # Dimensions of the output index array
    cdef size_t sx = img.shape[dim1]
    cdef size_t sy = img.shape[dim2] 

    if boundary_vertices is None or boundary_f is None:
        boundary_vertices = np.full((sx, sy, 2), np.inf,dtype = np.int32)
        boundary_f = np.full((sx, sy, 2), np.inf)

    location = {}
    for x in range(sx):
        location[dim1] = x
        for y in range(sy):
            location[dim2] = y
            _get_parabolic_envelope(
                img,
                location,
                dimension,
                boundary_vertices[x,y,1],
                boundary_f[x,y,1],
                boundary_vertices[x,y,0],
                boundary_f[x,y,0]
                )

def get_parabolic_envelope_2d(img,dimension,boundary_vertices = None,boundary_f=None):
    """
    Determine the parabolic envelop along the specified dimension
    """
    # Identify the dimension to iterate over
    other_dim = [0, 1]
    other_dim.remove(dimension)
    other_dim = other_dim[0]

    # Dimensions of the output index array
    cdef size_t s = img.shape[other_dim]
    
    if boundary_vertices is None or boundary_f is None:
        boundary_vertices = np.full((s, 2), np.inf,dtype = np.int32)
        boundary_f = np.full((s, 2), np.inf,dtype = np.float32)

    location = {}
    for x in range(s):
        location[other_dim] = x
        _get_parabolic_envelope_2d(
            img,
            location,
            dimension,
            boundary_vertices[x,1],
            boundary_f[x,1],
            boundary_vertices[x,0],
            boundary_f[x,0]
            )


def _get_parabolic_envelope(
    float[:, :, :] img,
    dict location,
    int dimension,
    const int lower_vertex,
    const float lower_f, 
    const int upper_vertex, 
    const float upper_f
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
        lower_vertex,
        lower_f,
        upper_vertex,
        upper_f
         )

def _get_parabolic_envelope_2d(
    float[:, :] img,
    dict location,
    int dimension,
    const int lower_vertex,
    const float lower_f, 
    const int upper_vertex, 
    const float upper_f
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
        lower_vertex,
        lower_f,
        upper_vertex,
        upper_f
         )

def determine_parabolic_envelope_1d(
    float[:] img,
    uint64_t start,
    uint64_t end,
    const int lower_vertex,
    const float lower_f, 
    const int upper_vertex, 
    const float upper_f
    ):
    """
    """
    # Call the low-level function to get the nearest boundary index
    _determine_boundary_parabolic_envelope(
        <float*>&img[start],
        end,  
        1, #stride                     
        lower_vertex,
        lower_f,
        upper_vertex,
        upper_f
         )


def get_initial_envelope_correctors(img, dimension):
    """
    Determine the nearest solid (or phase change of multilabel) index
    for 3d array
    """
    dims = {0, 1, 2}
    dims.remove(dimension)
    dim1, dim2 = dims


    ## Collect nearset solid (or phase change of multilabel) index
    nearest_index = np.zeros([img.shape[dim1],img.shape[dim2],2])
    nearest_index[:,:,0] = _voxels.get_nearest_boundary_index_face(
        img=img,
        dimension=dimension,
        label=0,
        forward=True,
    ).astype(np.float32)
    nearest_index[:,:, 1] = _voxels.get_nearest_boundary_index_face(
        img=img,
        dimension=dimension,
        label=0,
        forward=False,
    ).astype(np.float32)

    ### initialize correctors
    correctors = np.zeros([img.shape[dim1],img.shape[dim2], 2])

    ### correct indexes
    correctors[:,:, 0] = np.where(
        nearest_index[:,:,1] != -1,
        img.shape[dimension] - nearest_index[:,:, 1],
        np.inf)
    correctors[:,:, 1] = np.where(
        nearest_index[:,:,0] != -1,
        nearest_index[:,:, 0] + 1,
        np.inf)

    return correctors


def get_initial_envelope_correctors_2d(img, dimension):
    """
    Determine the nearest solid (or phase change of multilabel) index
    for 2d array
    """
    other_dim = [0, 1]
    other_dim.remove(dimension)
    other_dim = other_dim[0]

    ## Collect nearset solid (or phase change of multilabel) index
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

    ### initialize correctors
    correctors = np.zeros([img.shape[other_dim], 2])

    ### correct indexes
    correctors[:, 0] = np.where(
        nearest_index[:,1] != -1,
        img.shape[dimension] - nearest_index[:, 1],
        np.inf)
    correctors[:, 1] = np.where(
        nearest_index[:,0] != -1,
        nearest_index[:, 0] + 1,
        np.inf)

    return correctors


def get_initial_envelope(img,img_out,dimension,boundary_voxels = None):
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

    if boundary_voxels is None:
        boundary_voxels = np.full((s1, s2, 2), np.inf) 
    
    
    location = {}
    for x in range(s1):
        location[dim1] = x
        for y in range(s2):
            location[dim2] = y
            
            _get_initial_envelope(img,img_out,location,dimension,boundary_voxels[x,y,0],boundary_voxels[x,y,1])
    
    _tofinite(img_out,s1*s2*sdim)

    return img_out

def get_initial_envelope_2d(img,img_out,dimension,boundary_voxels = None):
    """
    """
    # Identify the dimension to iterate over
    other_dim = [0, 1]
    other_dim.remove(dimension)
    other_dim = other_dim[0]

    # Dimensions of the output index array
    cdef size_t s = img.shape[other_dim]
    cdef size_t sdim = img.shape[dimension]

    if boundary_voxels is None:
        boundary_voxels = np.full((s, 2), np.inf) 
    
    location = {}
    for x in range(s):
        location[other_dim] = x
        _get_initial_envelope_2d(img,img_out,location,dimension,boundary_voxels[x,0],boundary_voxels[x,1])
    
    _tofinite_2d(img_out,s*sdim)

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
    cdef np.ndarray[float, ndim=1] output = np.zeros( (voxels,), dtype=np.float32 )
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
    uint8_t[:,:,:] img,
    float[:,:,:] img_out,
    dict location,
    int dimension,
    float lower_corrector,
    float upper_corrector
):
    """
    Perform a distance transform to compute an initial envelope in 3D.

    This function computes a distance transform on a 3D image array, applying corrections 
    to adjust the lower and upper bounds during the transformation process. 

    Parameters:
        img (UINT[:,:,:]): A 3D array representing the input image data, where each 
            element is an unsigned integer.
        start (uint64_t): The starting index in the flattened 1D representation of the image 
            where the transform will begin.
        size (int): The size of the segment to process in the flattened array, starting 
            from the `start` index.
        lower_corrector (float): A correction factor applied to adjust the lower bound of the 
            distance transform.
        upper_corrector (float): A correction factor applied to adjust the upper bound of the 
            distance transform.

    Returns:
        np.ndarray[float, ndim=3]: A 3D array of type `float32` containing the computed 
            distances for the specified segment. 
    
    Notes:
        - The input 3D image is treated as a flattened 1D array for the purposes of this function, 
          with indices mapped accordingly.
        - The function initializes an output array of the same total size as the input image, 
          ensuring that all elements are properly handled.
        - This function serves as a wrapper for the actual distance transform logic, which is 
          expected to be implemented elsewhere.
    """

    # Determine the size of the output slice
    cdef size_t _size = img.shape[dimension]

    cdef int[3] start = _voxels.get_start_indices(dimension, location)
    cdef int stride = _voxels.normalized_strides(img)[dimension]
    
    squared_edt_1d_multi_seg_new(
        <uint8_t*>&img[start[0],start[1],start[2]],
        <float*>&img_out[start[0],start[1],start[2]],
        _size,                                          
        stride,
        1,
        lower_corrector,
        upper_corrector                     
    )

    return img_out


def _get_initial_envelope_2d(
    uint8_t[:,:] img,
    float[:,:] img_out,
    dict location,
    int dimension,
    float lower_corrector,
    float upper_corrector
):
    """
    Perform a distance transform to compute an initial envelope in 2D.

    This function computes a distance transform on a 2D image array, applying corrections 
    to adjust the lower and upper bounds during the transformation process. 

    Parameters:
        img (UINT[:,:]): A 2D array representing the input image data, where each 
            element is an unsigned integer.
        start (uint64_t): The starting index in the flattened 1D representation of the image 
            where the transform will begin.
        size (int): The size of the segment to process in the flattened array, starting 
            from the `start` index.
        lower_corrector (float): A correction factor applied to adjust the lower bound of the 
            distance transform.
        upper_corrector (float): A correction factor applied to adjust the upper bound of the 
            distance transform.

    Returns:
        np.ndarray[float, ndim=2]: A 2D array of type `float32` containing the computed 
            distances for the specified segment. 
    
    Notes:
        - The input 2D image is treated as a flattened 1D array for the purposes of this function, 
          with indices mapped accordingly.
        - The function initializes an output array of the same total size as the input image, 
          ensuring that all elements are properly handled.
        - This function serves as a wrapper for the actual distance transform logic, which is 
          expected to be implemented elsewhere.
    """

    # Determine the size of the output slice
    cdef size_t _size = img.shape[dimension]

    cdef int[2] start = _voxels.get_start_indices_2d(dimension, location)
    cdef int stride = _voxels.normalized_strides(img)[dimension]
    
    squared_edt_1d_multi_seg_new(
        <uint8_t*>&img[start[0],start[1]],
        <float*>&img_out[start[0],start[1]],
        _size,                                          
        stride,
        1,
        lower_corrector,
        upper_corrector                     
    )

    return img_out




def get_boundary_hull(
    float[:,:,:] img,
    int dimension):
    """
    Determine the initial and last parabola vertex and value for a given 3d array
    """
    # Identify the two dimensions to iterate over
    dims = {0, 1, 2}
    dims.remove(dimension)
    dim1, dim2 = dims

    hull_vertices = np.zeros([img.shape[dim1],img.shape[dim2],2],dtype=np.int32)
    hull_f = np.zeros([img.shape[dim1],img.shape[dim2], 2],dtype=np.float32)

    cdef int[:,:,:] _hull_vertices = hull_vertices
    cdef float[:,:,:] _hull_f = hull_f
        
    # Determine the size of the output slice
    cdef size_t _size = img.shape[dimension]
    cdef size_t s1 = img.shape[dim1]
    cdef size_t s2 = img.shape[dim2]

    cdef int[3] start
    cdef int stride = _voxels.normalized_strides(img)[dimension]
    location = {}
    for x in range(s1):
        location[dim1] = x
        for y in range(s2):
            location[dim2] = y
            start = _voxels.get_start_indices(dimension, location)
            for nn in [0,1]:
                return_boundary_hull(
                    <float*>&img[start[0],start[1],start[2]],
                    _size,    
                    stride,
                    _hull_vertices[x,y,nn],
                    _hull_f[x,y,nn],
                    not nn
                    )

    return hull_vertices,hull_f


def get_boundary_hull_2d(
    float[:,:] img,
    int dimension):
    """
    Determine the initial and last parabola vertex and value for a given 2d array
    """
    other_dim = [0, 1]
    other_dim.remove(dimension)
    other_dim = other_dim[0]

    hull_vertices = np.zeros([img.shape[other_dim], 2],dtype=np.int32)
    hull_f = np.zeros([img.shape[other_dim], 2],dtype=np.float32)

    cdef int[:,:] _hull_vertices = hull_vertices
    cdef float[:,:] _hull_f = hull_f
        
    # Determine the size of the output slice
    cdef size_t _size = img.shape[dimension]
    cdef size_t s = img.shape[other_dim]

    cdef int[2] start
    cdef int stride = _voxels.normalized_strides(img)[dimension]
    location = {}
    for n in range(s):
        location[other_dim] = n
        start = _voxels.get_start_indices_2d(dimension, location)
        for nn in [0,1]:
            return_boundary_hull(
                <float*>&img[start[0],start[1]],
                _size,    
                stride,
                _hull_vertices[n,nn],
                _hull_f[n,nn],
                not nn
                )

    return hull_vertices,hull_f



def get_boundary_hull_1d(float[:] img,
                     uint64_t start,
                     uint64_t end,
                     bool left = True):
    """
    Determine the initial and last parabola vertex and value for a given 1d array
    """
    # Call the low-level function to get the nearest boundary index
    
    cdef int b_vertex
    cdef float b_f

    return_boundary_hull(
        <float*>&img[start],
        end,  
        1,
        b_vertex,
        b_f,
        left
        )

    return b_vertex,b_f