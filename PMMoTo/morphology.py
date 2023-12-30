import numpy as np
from mpi4py import MPI
import edt
import math
from .core import communication
comm = MPI.COMM_WORLD


#### ToDO: Add erosion,dilation,subtraction
#### Add fft in case it is faster. Believe it was not compared to edt approach

def gen_struct_element(subdomain,radius):
    """
    Generate the
    """
    voxel = subdomain.domain.voxel
    struct = np.array([math.ceil(radius/voxel[0]),math.ceil(radius/voxel[0]),
                       math.ceil(radius/voxel[1]),math.ceil(radius/voxel[1]),
                       math.ceil(radius/voxel[2]),math.ceil(radius/voxel[2])],
                       dtype=np.int64)

    #### Dont quite remmember why I wrote this. 
    x = np.linspace(-struct[0]*voxel[0],struct[0]*voxel[0],struct[0]*2+1)
    y = np.linspace(-struct[1]*voxel[1],struct[1]*voxel[1],struct[1]*2+1)
    z = np.linspace(-struct[2]*voxel[2],struct[2]*voxel[2],struct[2]*2+1)    
    xg,yg,zg = np.meshgrid(x,y,z,indexing='ij')
    s = xg**2 + yg**2 + zg**2

    struct_element = np.array(s <= radius * radius)

    return struct

def morph_add(subdomain,grid,phase,radius):
    """
    Perform a morpological addition on a given phase
    """

    struct = gen_struct_element(subdomain,radius)
    halo_grid,halo = communication.generate_halo(subdomain,grid,struct)

    ### Convert input grid or multiphase grid to binary for EDT
    _grid = np.where(halo_grid == phase,0,1)

    ### Perform EDT on haloed grid so no errors on boundaries
    _grid_distance = edt.edt3d(_grid, anisotropy=subdomain.domain.voxel)

    ### Morph Add based on EDT
    _grid_out = np.where( (_grid_distance <= radius),phase,halo_grid).astype(np.uint8)

    ### Trim Halo
    dim = _grid_out.shape
    _grid_out = _grid_out[halo[0]:dim[0]-halo[1],
                          halo[2]:dim[1]-halo[3],
                          halo[4]:dim[2]-halo[5]]
    grid_out = np.ascontiguousarray(_grid_out)

    return grid_out
