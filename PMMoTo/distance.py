import numpy as np
from mpi4py import MPI
from pykdtree.kdtree import KDTree
import edt
from . import communication
from _distance import _fixInterfaceCalc
from _distance import _getBoundarySolids

comm = MPI.COMM_WORLD




""" Solid = 0, Pore = 1 """

""" TO DO:
           Allow Mulitple Phases?
           Cython
"""

class EDT(object):

    def __init__(self,ID,subDomain,Domain,Orientation,grid):
        bufferSize         = 1
        self.extendFactor  = 0.7
        self.useIndex      = False
        self.ID            = ID
        self.subDomain     = subDomain
        self.Domain        = Domain
        self.Orientation   = Orientation
        self.EDT = np.zeros_like(self.subDomain.grid)
        self.visited = np.zeros_like(self.subDomain.grid,dtype=np.uint8)
        self.solids = None
        self.nS = 0
        self.faceSolids = []
        self.edgeSolids = []
        self.cornerSolids = []
        self.solidsAll = {self.ID: {'orientID':{}}}

        self.grid = grid
        self.x = self.subDomain.x
        self.y = self.subDomain.y
        self.z = self.subDomain.z
        self.subDomainSize = self.subDomain.subDomainSize
        self.buffer = self.subDomain.buffer
        self.distVals = None
        self.distCounts  = None
        self.minD = 0
        self.maxD = 0

    def genLocalEDT(self,):
        self.EDT = edt.edt3d(self.grid, anisotropy=(self.Domain.dX, self.Domain.dY, self.Domain.dZ))

    def getBoundarySolids(self):

        area = 2*self.grid.shape[0]*self.grid.shape[1] + 2*self.grid.shape[0]*self.grid.shape[2] + 2*self.grid.shape[1]*self.grid.shape[2]
        self.solids = -np.ones([area,4],dtype='int32')
        for fIndex in self.Orientation.faces:
            self.nS = _getBoundarySolids(self,
                               fIndex,
                               self.Orientation.faces[fIndex]['dir'],
                               self.Orientation.faces[fIndex]['argOrder'],
                               self.nS,
                               self.grid,
                               self.solids)

    def getFaceSolids(self):
        """
        Trim to minimize communication and reduce KD Tree. Identify on Surfaces, Edges, and Corners
        """
        self.faceSolids = [[] for _ in range(len(self.Orientation.faces))]
        extend = [self.extendFactor*x for x in self.subDomainSize]
        for fIndex in self.Orientation.faces:
            pointsXYZ = []
            points = self.solids[np.where( (self.solids[:,0]>-1)
                                 & (self.solids[:,1]>-1)
                                 & (self.solids[:,2]>-1)
                                 & ((self.solids[:,3]==fIndex)) )][:,0:3]

            for x,y,z in points:
                pointsXYZ.append([self.x[x],self.y[y],self.z[z]] )
            self.faceSolids[fIndex] = np.asarray(pointsXYZ)
            name = self.Orientation.faces[fIndex]['ID']
            self.solidsAll[self.ID]['orientID'][name] = np.copy(self.faceSolids[fIndex])

    def getEdgeSolids(self):

        self.edgeSolids = [[] for _ in range(len(self.Orientation.edges))]
        for eIndex in self.Orientation.edges:
            edgeID = self.Orientation.edges[eIndex]['ID']
            face1 = self.faceSolids[self.Orientation.edges[eIndex]['faceIndex'][0]]
            face2 = self.faceSolids[self.Orientation.edges[eIndex]['faceIndex'][1]]
            arg1  = self.Orientation.edges[eIndex]['dir'][0]
            arg2  = self.Orientation.edges[eIndex]['dir'][1]
            coords = [self.x,self.y,self.z]
            coord1 = coords[arg1]
            coord2 = coords[arg2]

            extend = [self.extendFactor*x for x in self.subDomainSize]
            plusArg = 0
            if edgeID[arg1] == 1:
                dom11 = coord1[-1] - extend[arg1]
                dom12 = coord1[-1]
            elif (edgeID[arg1] == -1):
                dom11 = coord1[0]
                dom12 = coord1[0] + extend[arg1]
            if edgeID[arg2] == 1:
                dom21 = coord2[-1] - extend[arg2]
                dom22 = coord2[-1]
            elif (edgeID[arg2] == -1):
                dom21 = coord2[0]
                dom22 = coord2[0] + extend[arg2]

            truth1 = dom21 <= face1[:,arg2+plusArg]
            truth2 = face1[:,arg2+plusArg] <= dom22
            self.edgeSolids[eIndex] = face1[np.where( truth1 & truth2 )]
            truth1 = dom11 <= face2[:,arg1+plusArg]
            truth2 = face2[:,arg1+plusArg] <= dom12
            self.edgeSolids[eIndex] = np.append(self.edgeSolids[eIndex],face2[np.where( truth1 & truth2 )],axis=0)
            self.edgeSolids[eIndex] = np.unique(np.array(self.edgeSolids[eIndex]),axis=0)

    def getCornerSolids(self):

        self.cornerSolids = [[] for _ in range(len(self.Orientation.corners))]

        for cIndex in self.Orientation.corners:
            cID = self.Orientation.corners[cIndex]['ID']
            face1 = self.faceSolids[self.Orientation.corners[cIndex]['faceIndex'][0]]
            face2 = self.faceSolids[self.Orientation.corners[cIndex]['faceIndex'][1]]
            face3 = self.faceSolids[self.Orientation.corners[cIndex]['faceIndex'][2]]

            plusArg = 0
            extend = [self.extendFactor*x for x in self.subDomainSize]
            if (cID[0] == 1):
                dom11 = self.x[-1] - extend[0]
                dom12 = self.x[-1]
            elif (cID[0] == -1):
                dom11 = self.x[0]
                dom12 = self.x[0] + extend[0]

            if (cID[1] == 1):
                dom21 = self.y[-1] - extend[1]
                dom22 = self.y[-1]
            elif (cID[1] == -1):
                dom21 = self.y[0]
                dom22 = self.y[0] + extend[1]
            if (cID[2] == 1):
                dom31 = self.z[-1] - extend[2]
                dom32 = self.z[-1]
            elif (cID[2] == -1):
                dom31 = self.z[0]
                dom32 = self.z[0] + extend[2]

            truth1 = (dom21 <= face1[:,1+plusArg]) & (face1[:,1+plusArg] <= dom22)
            truth2 = (dom31 <= face1[:,2+plusArg]) & (face1[:,2+plusArg] <= dom32)
            self.cornerSolids[cIndex] = face1[np.where(truth1 & truth2)]
            truth1 = (dom11 <= face2[:,0+plusArg]) & (face2[:,0+plusArg] <= dom12)
            truth2 = (dom31 <= face2[:,2+plusArg]) & (face2[:,2+plusArg] <= dom32)
            self.cornerSolids[cIndex] = np.append(self.cornerSolids[cIndex],face2[np.where(truth1 & truth2)],axis=0)
            truth1 = (dom11 <= face3[:,0+plusArg]) & (face3[:,0+plusArg] <= dom12)
            truth2 = (dom21 <= face3[:,1+plusArg]) & (face3[:,1+plusArg] <= dom22)
            self.cornerSolids[cIndex] = np.append(self.cornerSolids[cIndex], face3[np.where(truth1 & truth2)],axis=0)

    def fixInterfaceCalc(self,tree,faceID):

        order  = [None]*3
        orderL = [None]*3
        nC  = self.Orientation.faces[faceID]['nC']
        nM  = self.Orientation.faces[faceID]['nM']
        nN  = self.Orientation.faces[faceID]['nN']
        dir = self.Orientation.faces[faceID]['dir']
        coords = [self.x,self.y,self.z]
        minD = min(self.Domain.dX,self.Domain.dY,self.Domain.dZ)

        faceSolids = self.solids[np.where(self.solids[:,3]==faceID)][:,0:3]

        if (dir == 1):
            for i in range(0,faceSolids.shape[0]):

                if faceSolids[i,nC] < 0:
                    endL = self.grid.shape[nC]
                else:
                    endL = faceSolids[i,nC]

                distChanged = True
                l = 0
                while distChanged and l < endL:
                    cL = l
                    cM = faceSolids[i,nM]
                    cN = faceSolids[i,nN]

                    m = cM
                    n = cN

                    order[nC] = coords[nC][cL]
                    order[nM] = coords[nM][cM]
                    order[nN] = coords[nN][cN]
                    orderL[nC] = l
                    orderL[nM] = cM
                    orderL[nN] = cN

                    maxD = self.EDT[orderL[0],orderL[1],orderL[2]]
                    if (maxD > minD):
                        d,ind = tree.query([order],p=2,distance_upper_bound=maxD)
                        if d < maxD:
                            self.EDT[orderL[0],orderL[1],orderL[2]] = d
                            distChanged = True
                            self.visited[orderL[0],orderL[1],orderL[2]] = 1
                        elif self.visited[orderL[0],orderL[1],orderL[2]] == 0:
                            distChanged = False
                    l = l + 1

        if (dir == -1):
            for i in range(0,faceSolids.shape[0]):

                if faceSolids[i,nC] < 0:
                    endL = 0
                else:
                    endL = faceSolids[i,nC]

                distChanged = True
                l = self.grid.shape[nC] - 1

                while distChanged and l > endL:
                    cL = l
                    cM = faceSolids[i,nM]
                    cN = faceSolids[i,nN]
                    m = cM
                    n = cN
                    order[nC] = coords[nC][cL]
                    order[nM] = coords[nM][cM]
                    order[nN] = coords[nN][cN]
                    orderL[nC] = l
                    orderL[nM] = m
                    orderL[nN] = n

                    maxD = self.EDT[orderL[0],orderL[1],orderL[2]]
                    if (maxD > minD):
                        d,ind = tree.query([order],p=2,distance_upper_bound=maxD)
                        if d < maxD:
                            self.EDT[orderL[0],orderL[1],orderL[2]] = d
                            distChanged = True
                            self.visited[orderL[0],orderL[1],orderL[2]] = 1
                        elif self.visited[orderL[0],orderL[1],orderL[2]] == 0:
                            distChanged = False
                    l = l - 1

    def initRecieve(self):

        for neigh in self.subDomain.neighborF:
            if neigh > -1 and neigh != self.ID:
                self.solidsAll[neigh] = {'orientID':{}}

        for neigh in self.subDomain.neighborE:
            if neigh > -1 and neigh != self.ID:
                self.solidsAll[neigh] = {'orientID':{}}

        for neigh in self.subDomain.neighborC:
            if neigh > -1 and neigh != self.ID:
                self.solidsAll[neigh] = {'orientID':{}}

    def fixInterface(self):

        minD = min(self.Domain.dX,self.Domain.dY,self.Domain.dZ)
        coords = [self.x,self.y,self.z]

        for fIndex in self.Orientation.faces:
            orientID = self.Orientation.faces[fIndex]['ID']
            data = np.empty((0,3))
            for procs in self.solidsAll.keys():
                for fID in self.solidsAll[procs]['orientID'].keys():
                    if fID == orientID:
                        data = np.append(data,self.solidsAll[procs]['orientID'][fID],axis=0)

            tree = KDTree(data)

            faceSolids = self.solids[np.where(self.solids[:,3]==fIndex)][:,0:3]

            self.EDT,self.visited = _fixInterfaceCalc(self,
                                                      tree,
                                                      fIndex,
                                                      self.grid.shape[self.Orientation.faces[fIndex]['argOrder'][0]],
                                                      self.Orientation.faces[fIndex]['dir'],
                                                      faceSolids,
                                                      self.EDT,
                                                      self.visited,
                                                      minD,
                                                      coords,
                                                      self.Orientation.faces[fIndex]['argOrder'])

    def EDTCommPack(self):

        self.sendData = {self.subDomain.ID: {'NeighborProcID':{}}}

        for fIndex in self.Orientation.faces:
            neigh = self.subDomain.neighborF[fIndex]
            oppIndex = self.Orientation.faces[fIndex]['oppIndex']
            oppNeigh = self.subDomain.neighborF[oppIndex]
            if oppNeigh > -1 and oppNeigh not in self.sendData[self.subDomain.ID]['NeighborProcID'].keys():
                self.sendData[self.subDomain.ID]['NeighborProcID'][oppNeigh] = {'ID':{}}
            if (oppNeigh > -1 and neigh != self.subDomain.ID and oppNeigh in self.sendData[self.subDomain.ID]['NeighborProcID'].keys()):
                self.sendData[self.subDomain.ID]['NeighborProcID'][oppNeigh]['ID'][fIndex] = self.faceSolids[oppIndex]

        for eIndex in self.Orientation.edges:
            neigh = self.subDomain.neighborE[eIndex]
            oppIndex = self.Orientation.edges[eIndex]['oppIndex']
            oppNeigh = self.subDomain.neighborE[oppIndex]
            if oppNeigh > -1 and oppNeigh not in self.sendData[self.subDomain.ID]['NeighborProcID'].keys():
                self.sendData[self.subDomain.ID]['NeighborProcID'][oppNeigh] = {'ID':{}}
            if (oppNeigh > -1 and neigh != self.subDomain.ID and oppNeigh in self.sendData[self.subDomain.ID]['NeighborProcID'].keys()):
                self.sendData[self.subDomain.ID]['NeighborProcID'][oppNeigh]['ID'][eIndex] = self.edgeSolids[oppIndex]

        for cIndex in self.Orientation.corners:
            neigh = self.subDomain.neighborC[cIndex]
            oppIndex = self.Orientation.corners[cIndex]['oppIndex']
            oppNeigh = self.subDomain.neighborC[oppIndex]
            if oppNeigh > -1 and oppNeigh not in self.sendData[self.subDomain.ID]['NeighborProcID'].keys():
                self.sendData[self.subDomain.ID]['NeighborProcID'][oppNeigh] = {'ID':{}}
            if (oppNeigh > -1 and neigh != self.subDomain.ID and oppNeigh in self.sendData[self.subDomain.ID]['NeighborProcID'].keys()):
                self.sendData[self.subDomain.ID]['NeighborProcID'][oppNeigh]['ID'][cIndex] = self.cornerSolids[oppIndex]

    def EDTComm(self):

        self.dataRecvFace,self.dataRecvEdge,self.dataRecvCorner = communication.subDomainComm(self.Orientation,self.subDomain,self.sendData[self.subDomain.ID]['NeighborProcID'])

    def EDTCommUnpack(self):

        #### FACE ####
        for fIndex in self.Orientation.faces:
            orientID = self.Orientation.faces[fIndex]['ID']
            neigh = self.subDomain.neighborF[fIndex]
            perFace  = self.subDomain.neighborPerF[fIndex]
            perCorrection = perFace*self.Domain.domainLength

            if (neigh > -1 and neigh != self.ID):
                self.solidsAll[neigh]['orientID'][orientID] = self.dataRecvFace[fIndex]['ID'][fIndex]
                if (perFace.any() != 0):
                    self.solidsAll[neigh]['orientID'][orientID] = self.solidsAll[neigh]['orientID'][orientID]-perCorrection
            elif (neigh == self.ID):
                oppIndex = self.Orientation.faces[fIndex]['oppIndex']
                self.solidsAll[neigh]['orientID'][orientID] = np.append(self.solidsAll[neigh]['orientID'][orientID],self.faceSolids[oppIndex]-perCorrection,axis=0)

        # #### EDGES ####
        for eIndex in self.Orientation.edges:
            neigh = self.subDomain.neighborE[eIndex]
            oppIndex = self.Orientation.edges[eIndex]['oppIndex']
            perEdge = self.subDomain.neighborPerE[eIndex]
            perCorrection = perEdge*self.Domain.domainLength
            if (neigh > -1  and neigh != self.ID):
                faceIndex = self.Orientation.edges[eIndex]['faceIndex']
                for fIndex in faceIndex:
                    orientID = self.Orientation.faces[fIndex]['ID']
                    data = self.dataRecvEdge[eIndex]['ID'][eIndex]
                    if orientID in self.solidsAll[neigh]['orientID']:
                        if (perEdge.any() != 0):
                            self.solidsAll[neigh]['orientID'][orientID] = np.append(self.solidsAll[neigh]['orientID'][orientID],data - perCorrection,axis=0)
                        else:
                            self.solidsAll[neigh]['orientID'][orientID] = np.append(self.solidsAll[neigh]['orientID'][orientID],data,axis=0)
                    else:
                        if (perEdge.any() != 0):
                            self.solidsAll[neigh]['orientID'][orientID] = data - perCorrection
                        else:
                            self.solidsAll[neigh]['orientID'][orientID] = data

            elif(neigh == self.ID):
                faceIndex = self.Orientation.edges[eIndex]['faceIndex']
                for fIndex in faceIndex:
                    orientID = self.Orientation.faces[fIndex]['ID']
                    if (perEdge.any() != 0):
                        perCorrection = perEdge*self.Domain.domainLength
                        self.solidsAll[neigh]['orientID'][orientID] = np.append(self.solidsAll[neigh]['orientID'][orientID],self.edgeSolids[oppIndex]-perCorrection,axis=0)
                    else:
                        self.solidsAll[neigh]['orientID'][orientID] = np.append(self.solidsAll[neigh]['orientID'][orientID],self.edgeSolids[oppIndex],axis=0)

        #### CORNERS ####
        for cIndex in self.Orientation.corners:
            neigh = self.subDomain.neighborC[cIndex]
            perCorner = self.subDomain.neighborPerC[cIndex]
            perCorrection = perCorner*self.Domain.domainLength*self.Domain.periodic
            if (neigh > -1 and neigh != self.ID):
                faceIndex = self.Orientation.corners[cIndex]['faceIndex']
                for fIndex in faceIndex:
                    orientID = self.Orientation.faces[fIndex]['ID']
                    data = self.dataRecvCorner[cIndex]['ID'][cIndex]
                    if orientID in self.solidsAll[neigh]['orientID']:
                        if (perCorner.any() != 0):
                            self.solidsAll[neigh]['orientID'][orientID] =np.append(self.solidsAll[neigh]['orientID'][orientID],data-perCorrection,axis=0)
                        else:
                            self.solidsAll[neigh]['orientID'][orientID] =np.append(self.solidsAll[neigh]['orientID'][orientID],data,axis=0)
                    else:
                        self.solidsAll[neigh]['orientID'][orientID] = data
                        if (perCorner.any() != 0):
                            self.solidsAll[neigh]['orientID'][orientID] = self.solidsAll[neigh]['orientID'][orientID]-perCorrection
            elif neigh == self.ID:
                faceIndex = self.Orientation.corners[cIndex]['faceIndex']
                oppIndex = self.Orientation.corners[cIndex]['oppIndex']
                for fIndex in faceIndex:
                    orientID = self.Orientation.faces[fIndex]['ID']
                    if (perCorner.any() != 0):
                        self.solidsAll[neigh]['orientID'][orientID] = np.append(self.solidsAll[neigh]['orientID'][orientID],self.cornerSolids[oppIndex]-perCorrection,axis=0)
                    else:
                        self.solidsAll[neigh]['orientID'][orientID] = np.append(self.solidsAll[neigh]['orientID'][orientID],self.cornerSolids[oppIndex],axis=0)

    def genStats(self):
        own = self.subDomain.ownNodes
        ownEDT =  self.EDT[own[0][0]:own[0][1],
                            own[1][0]:own[1][1],
                            own[2][0]:own[2][1]]
        self.distVals,self.distCounts  = np.unique(ownEDT,return_counts=True)

def calcEDT(rank,size,domain,subDomain,grid,stats=False):

    sDEDT = EDT(Domain = domain, ID = rank, subDomain = subDomain, Orientation = subDomain.Orientation, grid = grid)
    sDEDT.genLocalEDT()
    if size > 1:
        sDEDT.getBoundarySolids()
        sDEDT.getFaceSolids()
        sDEDT.getEdgeSolids()
        sDEDT.getCornerSolids()
        sDEDT.initRecieve()
        sDEDT.EDTCommPack()
        sDEDT.EDTComm()
        sDEDT.EDTCommUnpack()
        sDEDT.fixInterface()
    if stats:
        sDEDT.genStats()
        EDTData = [sDEDT.ID,sDEDT.distVals,sDEDT.distCounts]
        EDTData = comm.gather(EDTData, root=0)
        if rank == 0:
            bins = np.empty([])
            for d in EDTData:
                if d[0] == 0:
                    bins = d[1]
                else:
                    bins = np.append(bins,d[1],axis=0)
                bins = np.unique(bins)

            counts = np.zeros_like(bins,dtype=np.int64)
            for d in EDTData:
                for n in range(0,d[1].size):
                    ind = np.where(bins==d[1][n])[0][0]
                    counts[ind] = counts[ind] + d[2][n]

            stats = np.stack((bins,counts), axis = 1)
            sDEDT.minD = bins[1]
            sDEDT.maxD = bins[-1]
            distData = [sDEDT.minD,sDEDT.maxD]
            print("Minimum distance:",sDEDT.minD,"Maximum distance:",sDEDT.maxD)
        else:
            distData = None
        distData = comm.bcast(distData, root=0)
        sDEDT.minD = distData[0]
        sDEDT.maxD = distData[1]


    return sDEDT
