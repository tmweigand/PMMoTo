import numpy as np
from mpi4py import MPI
from . import communication
import edt
import math
comm = MPI.COMM_WORLD

class Morphology(object):
    def __init__(self,Domain,subDomain,grid,radius):
        self.Domain = Domain
        self.subDomain = subDomain
        self.structElem = None
        self.radius = radius
        self.stuctRatio = np.zeros(6)
        self.gridOut = np.copy(grid)

    def genStructElem(self):

        self.structRatio = np.array([math.ceil(self.radius/self.Domain.dX),math.ceil(self.radius/self.Domain.dX),
                                     math.ceil(self.radius/self.Domain.dY),math.ceil(self.radius/self.Domain.dY),
                                     math.ceil(self.radius/self.Domain.dZ),math.ceil(self.radius/self.Domain.dZ)],
                                     dtype=np.int64)

        # x = np.linspace(-self.structRatio[0]*self.Domain.dX,self.structRatio[0]*self.Domain.dX,self.structRatio[0]*2+1)
        # y = np.linspace(-self.structRatio[1]*self.Domain.dY,self.structRatio[1]*self.Domain.dY,self.structRatio[1]*2+1)
        # z = np.linspace(-self.structRatio[2]*self.Domain.dZ,self.structRatio[2]*self.Domain.dZ,self.structRatio[2]*2+1)
        
        # xg,yg,zg = np.meshgrid(x,y,z,indexing='ij')
        # s = xg**2 + yg**2 + zg**2

        # self.structElem = np.array(s <= self.radius * self.radius)

    def morphAdd(self,phase):
        """
        Perform a morpological addition on a given phase
        """

        ### Convert input grid or multiphase grid to binary for EDT
        grid = np.where(self.haloGrid == phase,0,1)

        ### Perform EDT on haloed grid so no errors on boundaries
        gridEDT = edt.edt3d(grid, anisotropy=(self.Domain.dX, self.Domain.dY, self.Domain.dZ))

        ### Morph Add based on EDT
        gridOut = np.where( (gridEDT <= self.radius),phase,self.haloGrid).astype(np.uint8)

        ### Trim Halo
        dim = gridOut.shape
        gridOut = gridOut[self.halo[0]:dim[0]-self.halo[1],
                          self.halo[2]:dim[1]-self.halo[3],
                          self.halo[4]:dim[2]-self.halo[5]]
        self.gridOut = np.ascontiguousarray(gridOut)


def morph(phase,grid,subDomain,radius):

    sDMorph = Morphology(Domain = subDomain.Domain,subDomain = subDomain, grid = grid, radius = radius)
    sDComm = communication.Comm(Domain = subDomain.Domain,subDomain = subDomain,grid = grid)
    sDMorph.genStructElem()
    sDMorph.haloGrid,sDMorph.halo = sDComm.haloCommunication(sDMorph.structRatio)
    sDMorph.morphAdd(phase)

    return sDMorph.gridOut
