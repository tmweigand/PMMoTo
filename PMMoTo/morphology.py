import numpy as np
from mpi4py import MPI
from . import communication
import edt
import math
comm = MPI.COMM_WORLD

from . import dataOutput

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
        self.resPad = np.zeros([6],dtype=np.int64)

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


    def addReservoir(self,inlet):

        ### Add Reservoir of Phase 1
        if np.sum(inlet) > 0:
            c = 0
            for c,n in enumerate(inlet):
                if n == 1:
                    self.resPad[c] = 1

            self.haloGrid = np.pad(self.haloGrid, ((self.resPad[0], self.resPad[1]), 
                                                   (self.resPad[2], self.resPad[3]), 
                                                   (self.resPad[4], self.resPad[5])),
                                                   'constant', constant_values=1)
        if inlet[0] == 1:
            self.haloGrid[0,:,:]  = np.where(self.haloGrid[1,:,:] == 1,1,0)
        if inlet[1] == 1:
            self.haloGrid[-1,:,:] = np.where(self.haloGrid[-2,:,:] == 1,1,0)
        if inlet[2] == 1:
            self.haloGrid[:,0,:]  = np.where(self.haloGrid[:,1,:] == 1,1,0)
        if inlet[3] == 1:
            self.haloGrid[:,-1,:] = np.where(self.haloGrid[:,-2,:] == 1,1,0)
        if inlet[4] == 1:
            self.haloGrid[:,:,0]  = np.where(self.haloGrid[:,:,1] == 1,1,0)
        if inlet[5] == 1:
            self.haloGrid[:,:,-1] = np.where(self.haloGrid[:,:,-1] == 1,1,0)

    def morphAdd(self):


        self.gridOutEDT = edt.edt3d(np.logical_not(self.haloGrid), anisotropy=(self.Domain.dX, self.Domain.dY, self.Domain.dZ))
        gridOut = np.where( (self.gridOutEDT <= self.radius),1,0).astype(np.uint8)
        dim = gridOut.shape
        self.gridOut = gridOut[self.resPad[0]+self.halo[0]:dim[0]-self.halo[1]-self.resPad[1],
                               self.resPad[2]+self.halo[2]:dim[1]-self.halo[3]-self.resPad[3],
                               self.resPad[4]+self.halo[4]:dim[2]-self.halo[5]-self.resPad[5]]
        self.gridOut = np.ascontiguousarray(self.gridOut)



def morph(grid,inlet,subDomain,radius):

    sDMorph = Morphology(Domain = subDomain.Domain,subDomain = subDomain, grid = grid, radius = radius)
    sDComm = communication.Comm(Domain = subDomain.Domain,subDomain = subDomain,grid = grid)
    sDMorph.genStructElem()
    sDMorph.haloGrid,sDMorph.halo = sDComm.haloCommunication(sDMorph.structRatio)
    sDMorph.addReservoir(inlet)
    sDMorph.morphAdd()

    return sDMorph.gridOut
