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
        self.Orientation = subDomain.Orientation
        self.structElem = None
        self.radius = radius
        self.stuctRatio = np.zeros(3)
        self.grid = np.copy(grid)
        self.gridOut = np.copy(grid)

    def genStructElem(self):

        self.structRatio = np.array([math.ceil(self.radius/self.Domain.dX),
                                     math.ceil(self.radius/self.Domain.dY),
                                     math.ceil(self.radius/self.Domain.dZ)],dtype=np.int64)

        x = np.linspace(-self.structRatio[0]*self.Domain.dX,self.structRatio[0]*self.Domain.dX,self.structRatio[0]*2+1)
        y = np.linspace(-self.structRatio[1]*self.Domain.dY,self.structRatio[1]*self.Domain.dY,self.structRatio[1]*2+1)
        z = np.linspace(-self.structRatio[2]*self.Domain.dZ,self.structRatio[2]*self.Domain.dZ,self.structRatio[2]*2+1)
        
        xg,yg,zg = np.meshgrid(x,y,z,indexing='ij')
        s = xg**2 + yg**2 + zg**2

        self.structElem = np.array(s <= self.radius * self.radius)


    def morphAdd(self):
        self.gridOutEDT = edt.edt3d(np.logical_not(self.haloGrid), anisotropy=(self.Domain.dX, self.Domain.dY, self.Domain.dZ))
        gridOut = np.where( (self.gridOutEDT <= self.radius),1,0).astype(np.uint8)
        dim = gridOut.shape
        self.gridOut = gridOut[self.halo[1]:dim[0]-self.halo[0],
                               self.halo[3]:dim[1]-self.halo[2],
                               self.halo[5]:dim[2]-self.halo[4]]
        self.gridOutEDT = self.gridOutEDT[self.halo[1]:dim[0]-self.halo[0],
                               self.halo[3]:dim[1]-self.halo[2],
                               self.halo[5]:dim[2]-self.halo[4]]
        self.gridOut = np.ascontiguousarray(self.gridOut)
        self.gridOutEDT  = np.ascontiguousarray(self.gridOutEDT )



def morph(rank,size,Domain,subDomain,grid,radius):
    sDMorph = Morphology(Domain = Domain,subDomain = subDomain, grid = grid, radius = radius)
    sDComm = communication.Comm(Domain = Domain,subDomain = subDomain,grid = grid)
    sDMorph.genStructElem()
    sDMorph.haloGrid,sDMorph.halo = sDComm.haloCommunication(sDMorph.structRatio)
    sDMorph.morphAdd()

    return sDMorph
