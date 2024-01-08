import numpy as np
cimport numpy as cnp
cnp.import_array()
from mpi4py import MPI
from pykdtree.kdtree import KDTree
### if using WSL, uncomment line below. 
#from scipy.spatial import KDTree

from edt import edt3d
from ..core import communication
from ..core import nodes
from ..core import utils

comm = MPI.COMM_WORLD

__all__ = [
    "calc_edt"
]

def fixInterface(edt,subdomain,solids,external_solids):
    """
    Loop through faces and correct distance with external_solids
    """
    visited = np.zeros_like(edt,dtype=np.uint8)

    for face in subdomain.faces:
        if face.n_proc > -1:
            arg = face.info['argOrder']
            data = np.copy(external_solids[face.info['ID']])

            for edge in subdomain.edges:
                for f in edge.info['faceIndex']:
                    if f == face.ID and edge.n_proc > -1: 
                        data = np.append(data,external_solids[edge.info['ID']],axis=0)

            for corner in subdomain.corners:
                for f in corner.info['faceIndex']:
                    if f == face.ID and corner.n_proc > -1: 
                        data = np.append(data,external_solids[corner.info['ID']],axis=0)

            tree = KDTree(data)
            face_solids = solids[np.where(solids[:,3]==face.ID)][:,0:3]
            edt,visited = nodes.fixInterfaceCalc(tree,
                                                    edt.shape[arg[0]],
                                                    face.info['dir'],
                                                    face_solids,
                                                    edt,
                                                    visited,
                                                    min(subdomain.domain.voxel),
                                                    subdomain.coords,
                                                    arg)
    return edt

  
    # def genStats(self):
    #     """
    #     Get Information (non-zero min/max) of distance tranform
    #     """
    #     own = self.subdomain.index_own_nodes
    #     ownEDT =  self.EDT[own[0]:own[1],
    #                        own[2]:own[3],
    #                        own[4]:own[5]]
    #     distVals,distCounts  = np.unique(ownEDT,return_counts=True)
    #     EDTData = [self.subdomain.ID,distVals,distCounts]
    #     EDTData = comm.gather(EDTData, root=0)
    #     if self.subdomain.ID == 0:
    #         bins = np.empty([])
    #         for d in EDTData:
    #             if d[0] == 0:
    #                 bins = d[1]
    #             else:
    #                 bins = np.append(bins,d[1],axis=0)
    #             bins = np.unique(bins)

    #         counts = np.zeros_like(bins,dtype=np.int64)
    #         for d in EDTData:
    #             for n in range(0,d[1].size):
    #                 ind = np.where(bins==d[1][n])[0][0]
    #                 counts[ind] = counts[ind] + d[2][n]

    #         stats = np.stack((bins,counts), axis = 1)
    #         self.minD = bins[1]
    #         self.maxD = bins[-1]
    #         distData = [self.minD,self.maxD]
    #         print("Minimum distance:",self.minD,"Maximum distance:",self.maxD)
    #     else:
    #         distData = None
    #     distData = comm.bcast(distData, root=0)
    #     self.minD = distData[0]
    #     self.maxD = distData[1]

def calc_edt(subdomain,grid):
    """

    """
    size = subdomain.domain.num_subdomains
    edt = edt3d(grid, anisotropy = subdomain.domain.voxel)
    if size > 1 or (size == 1 and any(subdomain.boundary_ID == 2)):
        solids = nodes.get_boundary_nodes(grid,0)
        face_solids,edge_solids,corner_solids = utils.partition_boundary_solids(subdomain,solids)
        external_solids = communication.pass_external_data(subdomain,face_solids,edge_solids,corner_solids)
        edt = fixInterface(edt,subdomain,solids,external_solids)
        edt = communication.update_buffer(subdomain,edt)

    return edt

