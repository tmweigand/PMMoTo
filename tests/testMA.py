import numpy as np
from mpi4py import MPI
from scipy import ndimage
import scipy.ndimage as ndi
from scipy.ndimage import distance_transform_edt
from scipy.spatial import KDTree
import edt
import PMMoTo
from skimage.morphology import skeletonize


def my_function():

    nodes = [50,50,50]
    inlet  = [1,0,0]
    outlet = [-1,0,0]
    boundaries = [0,0,0]
    res = 1
    domainFile = './testDomains/10pack.out'

    rank = 0; subDomains = [1,1,1]; size = 1
    domain,sDL = PMMoTo.genDomainSubDomain(rank,size,subDomains,nodes,boundaries,inlet,outlet,res,"Sphere",domainFile,PMMoTo.readPorousMediaXYZR)
    sDEDTL = PMMoTo.calcEDT(rank,size,domain,sDL,sDL.grid,stats = False)

    sDMAL = PMMoTo.medialAxis.medialAxisEval(rank,size,domain,sDL,sDL.grid,sDEDTL.EDT,connect = True,cutoff = 0)

    ### Save Grid Data where kwargs are used for saving other grid data (i.e. EDT, Medial Axis)
    setGridDict = {'MA': sDMAL.MA}
    PMMoTo.saveGridData("dataOut/gridComplete",rank,domain,sDL,**setGridDict)

    overlap = 2
    subNodes = [0,0,0]
    nRem = [0,0,0]
    subNodes[0],nRem[0] = divmod(nodes[0],2)
    subNodes[1],nRem[1] = divmod(nodes[0],2)
    subNodes[2],nRem[2] = divmod(nodes[0],2)

    grid1 = sDL.grid[:,0:subNodes[1]+overlap,:]
    grid1 = np.ascontiguousarray(grid1)
    grid2 = sDL.grid[:,subNodes[1]-overlap:,:]
    grid2 = np.ascontiguousarray(grid1)

    ma1 = PMMoTo.medialAxis._skeletonize_3d_cy._compute_thin_image(grid1)
    ma2 = PMMoTo.medialAxis._skeletonize_3d_cy._compute_thin_image(grid2)


    sDLSave = np.copy(sDL.grid[:,subNodes[1]+overlap:,:])
    sDL.grid[:,subNodes[1]+overlap:,:] = 10
    PMMoTo.saveGridData("dataOut/gridLeft",rank,domain,sDL,**setGridDict)
    sDL.grid[:,subNodes[1]+overlap:,:] = sDLSave



    sDLSave = np.copy(sDL.grid[:,:subNodes[1]-overlap,:])
    sDL.grid[:,:subNodes[1]-overlap,:] = 10
    PMMoTo.saveGridData("dataOut/gridRight",rank,domain,sDL,**setGridDict)


if __name__ == "__main__":
    my_function()
