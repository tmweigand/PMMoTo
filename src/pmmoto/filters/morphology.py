"""morphology.py"""
import math
import edt
import numpy as np
from scipy.signal import fftconvolve
from ..core import communication
from ..core import utils

__all__ = [
    "gen_struct_ratio",
    "gen_struct_element",
    "morph_add",
    "dilate",
    "morph_subtract",
    "erode",
    "opening",
    "closing",
    "multiphase_dilation"
]

def gen_struct_ratio(subdomain,radius):
    """
    Generate the structuring element dimensions for halo communication
    """
    voxel = subdomain.domain.voxel
    struct_ratio = np.array([math.ceil(radius/voxel[0]),math.ceil(radius/voxel[0]),
                       math.ceil(radius/voxel[1]),math.ceil(radius/voxel[1]),
                       math.ceil(radius/voxel[2]),math.ceil(radius/voxel[2])],
                       dtype=np.int64)

    return struct_ratio

def gen_struct_element(subdomain,radius):
    """
    Generate the structuring element for FFT morpology approach
    """
    voxel = subdomain.domain.voxel
    struct_ratio = gen_struct_ratio(subdomain,radius)

    _x = np.linspace(-struct_ratio[0]*voxel[0],struct_ratio[0]*voxel[0],struct_ratio[0]*2+1)
    _y = np.linspace(-struct_ratio[1]*voxel[1],struct_ratio[1]*voxel[1],struct_ratio[1]*2+1)
    _z = np.linspace(-struct_ratio[2]*voxel[2],struct_ratio[2]*voxel[2],struct_ratio[2]*2+1)
    _xg,_yg,_zg = np.meshgrid(_x,_y,_z,indexing='ij')
    _s = _xg**2 + _yg**2 + _zg**2

    struct_element = np.array(_s <= radius*radius,dtype=np.uint8)

    return struct_ratio,struct_element

def morph_add(subdomain,grid,radius,fft = False):
    """
    Perform a morpological dilation on a multiphase domain
    """
    struct_ratio,struct_element = gen_struct_element(subdomain,radius)
    halo_grid,halo = communication.generate_halo(subdomain,grid,struct_ratio)

    if fft:
        _grid = fftconvolve(halo_grid, struct_element, mode='same') > 0.1
        _grid_out = _grid.astype(np.uint8)
    else:
        _grid_distance = edt.edt3dsq(np.logical_not(halo_grid), anisotropy=subdomain.domain.voxel)
        _grid_out = np.where( (_grid_distance <= radius*radius),1,0).astype(np.uint8)

    grid_out = utils.unpad(_grid_out,halo)

    return grid_out

def dilate(subdomain,grid,radius,fft = False):
    """
    Wrapper to morph_add
    """
    grid_out = morph_add(subdomain,grid,radius,fft)

    return grid_out

def morph_subtract(subdomain,grid,radius,fft = False):
    """
    Perform a morpological subtraction
    """
    struct_ratio,struct_element = gen_struct_element(subdomain,radius)
    halo_grid,halo = communication.generate_halo(subdomain,grid,struct_ratio)

    if fft:
        ### Boundary condtion fix for erosion
        _pad = 1
        if any(subdomain.boundary_type == 0):
           _pad = np.max(struct_ratio)
        _grid = np.pad(array=halo_grid, pad_width = _pad, mode = 'constant', constant_values = 1)
        _grid = fftconvolve(_grid, struct_element, mode='same') > (struct_element.sum() - 0.1)
        _grid_out = utils.unpad(_grid,_pad*np.ones_like(halo)).astype(np.uint8)
    else:
        _grid_distance = edt.edt3dsq(halo_grid, anisotropy=subdomain.domain.voxel)
        _grid_out = np.where( (_grid_distance <= radius*radius),0,1).astype(np.uint8)

    grid_out = utils.unpad(_grid_out,halo)

    return grid_out

def erode(subdomain,grid,radius,fft = False):
    """
    Wrapper to morph_subtract
    """
    grid_out = morph_subtract(subdomain,grid,radius,fft)

    return grid_out


def opening(subdomain,grid,radius,fft = False):
    """
    Morphological opening
    """
    _erode = morph_subtract(subdomain,grid,radius,fft)
    open_map = morph_add(subdomain,_erode,radius,fft)
    return open_map

def closing(subdomain,grid,radius,fft = False):
    """
    Morphological opening
    """
    _dilate = morph_add(subdomain,grid,radius,fft)
    closing_map = morph_subtract(subdomain,_dilate,radius,fft)
    return closing_map

def multiphase_dilation(subdomain,grid,phase,radius,fft = False):
    """
    Perform a morpological dilation on a multiphase domain
    """
    struct_ratio,struct_element = gen_struct_element(subdomain,radius)
    halo_grid,halo = communication.generate_halo(subdomain,grid,struct_ratio)

    if fft:
        _grid = np.where(halo_grid == phase,1,0)
        _grid = fftconvolve(_grid, struct_element, mode='same') > 0.1
        _grid_out = np.where(_grid,phase,halo_grid).astype(np.uint8)
    else:
        _grid = np.where(halo_grid == phase,0,1)
        _grid_distance = edt.edt3dsq(_grid, anisotropy=subdomain.domain.voxel)
        _grid_out = np.where( (_grid_distance <= radius*radius),phase,halo_grid).astype(np.uint8)

    grid_out = utils.unpad(_grid_out,halo)

    return grid_out
