import numpy as np
from mpi4py import MPI
import sys
comm = MPI.COMM_WORLD

def raiseError():
    MPI.Finalize()
    sys.exit()

class Comm(object):
    def __init__(self,Domain,subDomain,grid = None):
        self.Domain = Domain
        self.subDomain = subDomain
        self.Orientation = subDomain.Orientation
        self.grid = grid
        self.dim = self.grid.shape
        self.halo = np.zeros([6],dtype=np.int64)
        self.haloData = {self.subDomain.ID: {'NeighborProcID':{}}}
   

    def bufferCommPack(self):
        """
        Grab The Slices (based on Buffer Size [1,1,1]) to pack and send to Neighbors
        for faces, edges and corners
        """

        self.Orientation.getSendBufferSlices(self.subDomain.buffer,self.dim)

        self.slices = self.Orientation.sendFSlices
        for fIndex in self.Orientation.faces:
            neigh = self.subDomain.neighborF[fIndex]
            if neigh > -1:
                if neigh not in self.haloData[self.subDomain.ID]['NeighborProcID'].keys():
                    self.haloData[self.subDomain.ID]['NeighborProcID'][neigh] = {'Index':{}}
                self.haloData[self.subDomain.ID]['NeighborProcID'][neigh]['Index'][fIndex] = self.grid[self.slices[fIndex,0],self.slices[fIndex,1],self.slices[fIndex,2]]

        self.slices = self.Orientation.sendESlices
        for eIndex in self.Orientation.edges:
            neigh = self.subDomain.neighborE[eIndex]
            if neigh > -1:
                if neigh not in self.haloData[self.subDomain.ID]['NeighborProcID'].keys():
                    self.haloData[self.subDomain.ID]['NeighborProcID'][neigh] = {'Index':{}}
                self.haloData[self.subDomain.ID]['NeighborProcID'][neigh]['Index'][eIndex] = self.grid[self.slices[eIndex,0],self.slices[eIndex,1],self.slices[eIndex,2]]

        self.slices = self.Orientation.sendCSlices
        for cIndex in self.Orientation.corners:
            neigh = self.subDomain.neighborC[cIndex]
            if neigh > -1:
                if neigh not in self.haloData[self.subDomain.ID]['NeighborProcID'].keys():
                    self.haloData[self.subDomain.ID]['NeighborProcID'][neigh] = {'Index':{}}
                self.haloData[self.subDomain.ID]['NeighborProcID'][neigh]['Index'][cIndex] = self.grid[self.slices[cIndex,0],self.slices[cIndex,1],self.slices[cIndex,2]]

        self.haloGrid = np.copy(self.grid)

    def bufferComm(self):
        self.dataRecvFace,self.dataRecvEdge,self.dataRecvCorner = subDomainComm(self.Orientation,self.subDomain,self.haloData[self.subDomain.ID]['NeighborProcID'])

    def bufferCommUnpack(self):

        self.Orientation.getRecieveBufferSlices(self.subDomain.buffer,self.haloGrid.shape)

        #### Faces ####
        self.slices = self.Orientation.recvFSlices
        for fIndex in self.Orientation.faces:
            neigh = self.subDomain.neighborF[fIndex]
            oppIndex = self.Orientation.faces[fIndex]['oppIndex']
            if (neigh > -1 and neigh != self.subDomain.ID):
                if neigh in self.haloData[self.subDomain.ID]['NeighborProcID'].keys():
                    if neigh not in self.haloData:
                        self.haloData[neigh] = {'NeighborProcID':{}}
                    self.haloData[neigh]['NeighborProcID'][neigh] = self.dataRecvFace[fIndex]['Index'][oppIndex]
                    self.haloGrid[self.slices[fIndex,0],self.slices[fIndex,1],self.slices[fIndex,2]] = self.haloData[neigh]['NeighborProcID'][neigh]

        #### Edges ####
        self.slices = self.Orientation.recvESlices
        for eIndex in self.Orientation.edges:
            neigh = self.subDomain.neighborE[eIndex]
            oppIndex = self.Orientation.edges[eIndex]['oppIndex']
            if (neigh > -1 and neigh != self.subDomain.ID):
                if neigh in self.haloData[self.subDomain.ID]['NeighborProcID'].keys():
                    if neigh not in self.haloData:
                        self.haloData[neigh] = {'NeighborProcID':{}}
                    self.haloData[neigh]['NeighborProcID'][neigh] = self.dataRecvEdge[eIndex]['Index'][oppIndex]
                    self.haloGrid[self.slices[eIndex,0],self.slices[eIndex,1],self.slices[eIndex,2]] = self.haloData[neigh]['NeighborProcID'][neigh]

        #### Corners ####
        self.slices = self.Orientation.recvCSlices
        for cIndex in self.Orientation.corners:
            neigh = self.subDomain.neighborC[cIndex]
            oppIndex = self.Orientation.corners[cIndex]['oppIndex']
            if (neigh > -1 and neigh != self.subDomain.ID):
                if neigh in self.haloData[self.subDomain.ID]['NeighborProcID'].keys():
                    if neigh not in self.haloData:
                        self.haloData[neigh] = {'NeighborProcID':{}}
                    self.haloData[neigh]['NeighborProcID'][neigh] = self.dataRecvCorner[cIndex]['Index'][oppIndex]
                    self.haloGrid[self.slices[cIndex,0],self.slices[cIndex,1],self.slices[cIndex,2]] = self.haloData[neigh]['NeighborProcID'][neigh]


    def haloCommPack(self,haloIn):
        """
        Grab The Slices (based on Size) to pack and send to Neighbors
        for faces, edges and corners
        """

        self.Orientation.getSendSlices(haloIn,self.subDomain.buffer,self.dim)

        self.slices = self.Orientation.sendFSlices
        for fIndex in self.Orientation.faces:
            neigh = self.subDomain.neighborF[fIndex]
            fID = self.Orientation.faces[fIndex]['ID']
            if neigh > -1:
                self.halo[fIndex]= haloIn[fIndex]
                if neigh not in self.haloData[self.subDomain.ID]['NeighborProcID'].keys():
                    self.haloData[self.subDomain.ID]['NeighborProcID'][neigh] = {'Index':{}}
                self.haloData[self.subDomain.ID]['NeighborProcID'][neigh]['Index'][fIndex] = self.grid[self.slices[fIndex,0],self.slices[fIndex,1],self.slices[fIndex,2]]


        self.slices = self.Orientation.sendESlices
        for eIndex in self.Orientation.edges:
            neigh = self.subDomain.neighborE[eIndex]
            if neigh > -1:
                if neigh not in self.haloData[self.subDomain.ID]['NeighborProcID'].keys():
                    self.haloData[self.subDomain.ID]['NeighborProcID'][neigh] = {'Index':{}}
                self.haloData[self.subDomain.ID]['NeighborProcID'][neigh]['Index'][eIndex] = self.grid[self.slices[eIndex,0],self.slices[eIndex,1],self.slices[eIndex,2]] 

        self.slices = self.Orientation.sendCSlices
        for cIndex in self.Orientation.corners:
            neigh = self.subDomain.neighborC[cIndex]
            if neigh > -1:
                if neigh not in self.haloData[self.subDomain.ID]['NeighborProcID'].keys():
                    self.haloData[self.subDomain.ID]['NeighborProcID'][neigh] = {'Index':{}}
                self.haloData[self.subDomain.ID]['NeighborProcID'][neigh]['Index'][cIndex] = self.grid[self.slices[cIndex,0],self.slices[cIndex,1],self.slices[cIndex,2]]

        self.haloGrid = np.pad(self.grid, ( (self.halo[0], self.halo[1]), (self.halo[2], self.halo[3]), (self.halo[4], self.halo[5]) ), 'constant', constant_values=255)

    def haloComm(self):
        self.dataRecvFace,self.dataRecvEdge,self.dataRecvCorner = subDomainComm(self.Orientation,self.subDomain,self.haloData[self.subDomain.ID]['NeighborProcID'])

    def haloCommUnpack(self):


        self.Orientation.getRecieveSlices(self.halo,self.subDomain.buffer,self.haloGrid.shape)

        #### Faces ####
        self.slices = self.Orientation.recvFSlices
        for fIndex in self.Orientation.faces:
            neigh = self.subDomain.neighborF[fIndex]
            oppIndex = self.Orientation.faces[fIndex]['oppIndex']
            if (neigh > -1 and neigh != self.subDomain.ID):
                if neigh in self.haloData[self.subDomain.ID]['NeighborProcID'].keys():
                    if neigh not in self.haloData:
                        self.haloData[neigh] = {'NeighborProcID':{}}
                    self.haloData[neigh]['NeighborProcID'][neigh] = self.dataRecvFace[fIndex]['Index'][oppIndex]
                    self.haloGrid[self.slices[fIndex,0],self.slices[fIndex,1],self.slices[fIndex,2]] = self.haloData[neigh]['NeighborProcID'][neigh]

        #### Edges ####
        self.slices = self.Orientation.recvESlices
        for eIndex in self.Orientation.edges:
            neigh = self.subDomain.neighborE[eIndex]
            oppIndex = self.Orientation.edges[eIndex]['oppIndex']
            if (neigh > -1 and neigh != self.subDomain.ID):
                if neigh in self.haloData[self.subDomain.ID]['NeighborProcID'].keys():
                    if neigh not in self.haloData:
                        self.haloData[neigh] = {'NeighborProcID':{}}
                    self.haloData[neigh]['NeighborProcID'][neigh] = self.dataRecvEdge[eIndex]['Index'][oppIndex]
                    self.haloGrid[self.slices[eIndex,0],self.slices[eIndex,1],self.slices[eIndex,2]] = self.haloData[neigh]['NeighborProcID'][neigh]

        #### Corners ####
        self.slices = self.Orientation.recvCSlices
        for cIndex in self.Orientation.corners:
            neigh = self.subDomain.neighborC[cIndex]
            oppIndex = self.Orientation.corners[cIndex]['oppIndex']
            if (neigh > -1 and neigh != self.subDomain.ID):
                if neigh in self.haloData[self.subDomain.ID]['NeighborProcID'].keys():
                    if neigh not in self.haloData:
                        self.haloData[neigh] = {'NeighborProcID':{}}
                    self.haloData[neigh]['NeighborProcID'][neigh] = self.dataRecvCorner[cIndex]['Index'][oppIndex]
                    self.haloGrid[self.slices[cIndex,0],self.slices[cIndex,1],self.slices[cIndex,2]] = self.haloData[neigh]['NeighborProcID'][neigh]

    def EDTCommPack(self,faceSolids,edgeSolids,cornerSolids):

        self.sendData = {self.subDomain.ID: {'NeighborProcID':{}}}

        for fIndex in self.Orientation.faces:
            neigh = self.subDomain.neighborF[fIndex]
            oppIndex = self.Orientation.faces[fIndex]['oppIndex']
            oppNeigh = self.subDomain.neighborF[oppIndex]
            if oppNeigh > -1 and oppNeigh not in self.sendData[self.subDomain.ID]['NeighborProcID'].keys():
                self.sendData[self.subDomain.ID]['NeighborProcID'][oppNeigh] = {'Index':{}}
            if (oppNeigh > -1 and neigh != self.subDomain.ID and oppNeigh in self.sendData[self.subDomain.ID]['NeighborProcID'].keys()):
                self.sendData[self.subDomain.ID]['NeighborProcID'][oppNeigh]['Index'][fIndex] = faceSolids[oppIndex]

        for eIndex in self.Orientation.edges:
            neigh = self.subDomain.neighborE[eIndex]
            oppIndex = self.Orientation.edges[eIndex]['oppIndex']
            oppNeigh = self.subDomain.neighborE[oppIndex]
            if oppNeigh > -1 and oppNeigh not in self.sendData[self.subDomain.ID]['NeighborProcID'].keys():
                self.sendData[self.subDomain.ID]['NeighborProcID'][oppNeigh] = {'Index':{}}
            if (oppNeigh > -1 and neigh != self.subDomain.ID and oppNeigh in self.sendData[self.subDomain.ID]['NeighborProcID'].keys()):
                self.sendData[self.subDomain.ID]['NeighborProcID'][oppNeigh]['Index'][eIndex] = edgeSolids[oppIndex]

        for cIndex in self.Orientation.corners:
            neigh = self.subDomain.neighborC[cIndex]
            oppIndex = self.Orientation.corners[cIndex]['oppIndex']
            oppNeigh = self.subDomain.neighborC[oppIndex]
            if oppNeigh > -1 and oppNeigh not in self.sendData[self.subDomain.ID]['NeighborProcID'].keys():
                self.sendData[self.subDomain.ID]['NeighborProcID'][oppNeigh] = {'Index':{}}
            if (oppNeigh > -1 and neigh != self.subDomain.ID and oppNeigh in self.sendData[self.subDomain.ID]['NeighborProcID'].keys()):
                self.sendData[self.subDomain.ID]['NeighborProcID'][oppNeigh]['Index'][cIndex] = cornerSolids[oppIndex]

    def EDTComm(self):
        self.dataRecvFace,self.dataRecvEdge,self.dataRecvCorner = subDomainComm(self.Orientation,self.subDomain,self.sendData[self.subDomain.ID]['NeighborProcID'])

    def EDTCommUnpack(self,solidsAll,faceSolids,edgeSolids,cornerSolids):

        #### FACE ####
        for fIndex in self.Orientation.faces:
            orientID = self.Orientation.faces[fIndex]['ID']
            neigh = self.subDomain.neighborF[fIndex]
            perFace  = self.subDomain.neighborPerF[fIndex]
            perCorrection = perFace*self.Domain.domainLength

            if (neigh > -1 and neigh != self.subDomain.ID):
                solidsAll[neigh]['orientID'][orientID] = self.dataRecvFace[fIndex]['Index'][fIndex]
                if (perFace.any() != 0):
                    solidsAll[neigh]['orientID'][orientID] = solidsAll[neigh]['orientID'][orientID]-perCorrection
            elif (neigh == self.subDomain.ID):
                oppIndex = self.Orientation.faces[fIndex]['oppIndex']
                solidsAll[neigh]['orientID'][orientID] = np.append(solidsAll[neigh]['orientID'][orientID],faceSolids[oppIndex]-perCorrection,axis=0)

        # #### EDGES ####
        for eIndex in self.Orientation.edges:
            neigh = self.subDomain.neighborE[eIndex]
            oppIndex = self.Orientation.edges[eIndex]['oppIndex']
            perEdge = self.subDomain.neighborPerE[eIndex]
            perCorrection = perEdge*self.Domain.domainLength
            if (neigh > -1  and neigh != self.subDomain.ID):
                faceIndex = self.Orientation.edges[eIndex]['faceIndex']
                for fIndex in faceIndex:
                    orientID = self.Orientation.faces[fIndex]['ID']
                    data = self.dataRecvEdge[eIndex]['Index'][eIndex]
                    if orientID in solidsAll[neigh]['orientID']:
                        if (perEdge.any() != 0):
                            solidsAll[neigh]['orientID'][orientID] = np.append(solidsAll[neigh]['orientID'][orientID],data - perCorrection,axis=0)
                        else:
                            solidsAll[neigh]['orientID'][orientID] = np.append(solidsAll[neigh]['orientID'][orientID],data,axis=0)
                    else:
                        if (perEdge.any() != 0):
                            solidsAll[neigh]['orientID'][orientID] = data - perCorrection
                        else:
                            solidsAll[neigh]['orientID'][orientID] = data

            elif(neigh == self.subDomain.ID):
                faceIndex = self.Orientation.edges[eIndex]['faceIndex']
                for fIndex in faceIndex:
                    orientID = self.Orientation.faces[fIndex]['ID']
                    if (perEdge.any() != 0):
                        perCorrection = perEdge*self.Domain.domainLength
                        solidsAll[neigh]['orientID'][orientID] = np.append(solidsAll[neigh]['orientID'][orientID],edgeSolids[oppIndex]-perCorrection,axis=0)
                    else:
                        solidsAll[neigh]['orientID'][orientID] = np.append(solidsAll[neigh]['orientID'][orientID],edgeSolids[oppIndex],axis=0)

        #### CORNERS ####
        for cIndex in self.Orientation.corners:
            neigh = self.subDomain.neighborC[cIndex]
            perCorner = self.subDomain.neighborPerC[cIndex]
            perCorrection = perCorner*self.Domain.domainLength  #*self.Domain.periodic #TMW MAYBE BUG?
            if (neigh > -1 and neigh != self.subDomain.ID):
                faceIndex = self.Orientation.corners[cIndex]['faceIndex']
                for fIndex in faceIndex:
                    orientID = self.Orientation.faces[fIndex]['ID']
                    data = self.dataRecvCorner[cIndex]['Index'][cIndex]
                    if orientID in solidsAll[neigh]['orientID']:
                        if (perCorner.any() != 0):
                            solidsAll[neigh]['orientID'][orientID] =np.append(solidsAll[neigh]['orientID'][orientID],data-perCorrection,axis=0)
                        else:
                            solidsAll[neigh]['orientID'][orientID] =np.append(solidsAll[neigh]['orientID'][orientID],data,axis=0)
                    else:
                        solidsAll[neigh]['orientID'][orientID] = data
                        if (perCorner.any() != 0):
                            solidsAll[neigh]['orientID'][orientID] = solidsAll[neigh]['orientID'][orientID]-perCorrection
            elif neigh == self.subDomain.ID:
                faceIndex = self.Orientation.corners[cIndex]['faceIndex']
                oppIndex = self.Orientation.corners[cIndex]['oppIndex']
                for fIndex in faceIndex:
                    orientID = self.Orientation.faces[fIndex]['ID']
                    if (perCorner.any() != 0):
                        solidsAll[neigh]['orientID'][orientID] = np.append(solidsAll[neigh]['orientID'][orientID],cornerSolids[oppIndex]-perCorrection,axis=0)
                    else:
                        solidsAll[neigh]['orientID'][orientID] = np.append(solidsAll[neigh]['orientID'][orientID],cornerSolids[oppIndex],axis=0)


    def updateBuffer(self):
        self.bufferCommPack()
        self.bufferComm()
        self.bufferCommUnpack()
        return self.haloGrid

    def haloCommunication(self,haloIn):
        self.haloCommPack(haloIn)
        self.haloComm()
        self.haloCommUnpack()
        return self.haloGrid,self.halo

    def EDTCommunication(self,solidsAll,faceSolids,edgeSolids,cornerSolids):
        self.EDTCommPack(faceSolids,edgeSolids,cornerSolids)
        self.EDTComm()
        self.EDTCommUnpack(solidsAll,faceSolids,edgeSolids,cornerSolids)
        return solidsAll
    

def subDomainComm(Orientation,subDomain,sendData):

    #### FACE ####
    reqs = [None]*Orientation.numFaces
    reqr = [None]*Orientation.numFaces
    recvDataFace = [None]*Orientation.numFaces
    for fIndex in Orientation.faces:
        neigh = subDomain.neighborF[fIndex]
        oppIndex = Orientation.faces[fIndex]['oppIndex']
        oppNeigh = subDomain.neighborF[oppIndex]
        if (oppNeigh > -1 and neigh != subDomain.ID and oppNeigh in sendData.keys() ):
            reqs[fIndex] = comm.isend(sendData[oppNeigh],dest=oppNeigh)
        if (neigh > -1 and neigh != subDomain.ID and neigh in sendData.keys() ):
            reqr[fIndex] = comm.recv(source=neigh)

    reqs = [i for i in reqs if i]
    MPI.Request.waitall(reqs)

    for fIndex in Orientation.faces:
        neigh = subDomain.neighborF[fIndex]
        if (neigh > -1 and neigh != subDomain.ID and neigh in sendData.keys() ):
            recvDataFace[fIndex] = reqr[fIndex]

    #### EDGES ####
    reqs = [None]*Orientation.numEdges
    reqr = [None]*Orientation.numEdges
    recvDataEdge = [None]*Orientation.numEdges
    for eIndex in Orientation.edges:
        neigh = subDomain.neighborE[eIndex]
        oppIndex = Orientation.edges[eIndex]['oppIndex']
        oppNeigh = subDomain.neighborE[oppIndex]
        if (oppNeigh > -1 and neigh != subDomain.ID and oppNeigh in sendData.keys() ):
            reqs[eIndex] = comm.isend(sendData[oppNeigh],dest=oppNeigh)
        if (neigh > -1 and neigh != subDomain.ID and neigh in sendData.keys() ):
            reqr[eIndex] = comm.recv(source=neigh)

    reqs = [i for i in reqs if i]
    MPI.Request.waitall(reqs)

    for eIndex in Orientation.edges:
        neigh = subDomain.neighborE[eIndex]
        oppIndex = Orientation.edges[eIndex]['oppIndex']
        oppNeigh = subDomain.neighborE[oppIndex]
        if (neigh > -1 and neigh != subDomain.ID and neigh in sendData.keys() ):
            recvDataEdge[eIndex] = reqr[eIndex]

    #### CORNERS ####
    reqs = [None]*Orientation.numCorners
    reqr = [None]*Orientation.numCorners
    recvDataCorner = [None]*Orientation.numCorners
    for cIndex in Orientation.corners:
        neigh = subDomain.neighborC[cIndex]
        oppIndex = Orientation.corners[cIndex]['oppIndex']
        oppNeigh = subDomain.neighborC[oppIndex]
        if (oppNeigh > -1 and neigh != subDomain.ID and oppNeigh in sendData.keys() ):
            reqs[cIndex] = comm.isend(sendData[oppNeigh],dest=oppNeigh)
        if (neigh > -1 and neigh != subDomain.ID and neigh in sendData.keys() ):
            reqr[cIndex] = comm.recv(source=neigh)

    reqs = [i for i in reqs if i]
    MPI.Request.waitall(reqs)

    for cIndex in Orientation.corners:
        neigh = subDomain.neighborC[cIndex]
        if (neigh > -1 and neigh != subDomain.ID and neigh in sendData.keys() ):
            recvDataCorner[cIndex] = reqr[cIndex]

    return recvDataFace,recvDataEdge,recvDataCorner

def setCOMMNEW(Orientation,subDomain,data):
  """
  Transmit data to Neighboring Processors
  """
  dataRecvFace,dataRecvEdge,dataRecvCorner = subDomainComm(Orientation,subDomain,data[subDomain.ID]['nProcID'])

  #############
  ### Faces ###
  #############
  for fIndex in Orientation.faces:
    neigh = subDomain.neighborF[fIndex]
    if (neigh > -1 and neigh != subDomain.ID):
      if neigh in data[subDomain.ID]['nProcID'].keys():
        if neigh not in data:
          data[neigh] = {'nProcID':{}}
        data[neigh]['nProcID'][neigh] = dataRecvFace[fIndex]

  #############
  ### Edges ###
  #############
  for eIndex in Orientation.edges:
    neigh = subDomain.neighborE[eIndex]
    if (neigh > -1 and neigh != subDomain.ID):
      if neigh in data[subDomain.ID]['nProcID'].keys():
        if neigh not in data:
          data[neigh] = {'nProcID':{}}
        data[neigh]['nProcID'][neigh] = dataRecvEdge[eIndex]

  ###############
  ### Corners ###
  ###############
  for cIndex in Orientation.corners:
    neigh = subDomain.neighborC[cIndex]
    if (neigh > -1 and neigh != subDomain.ID):
      if neigh in data[subDomain.ID]['nProcID'].keys():
        if neigh not in data:
          data[neigh] = {'nProcID':{}}
        data[neigh]['nProcID'][neigh] = dataRecvCorner[cIndex]

  return data



def setCOMM(Orientation,subDomain,data):
  """
  Transmit data to Neighboring Processors
  """
  dataRecvFace,dataRecvEdge,dataRecvCorner = subDomainComm(Orientation,subDomain,data[subDomain.ID]['NeighborProcID'])

  #############
  ### Faces ###
  #############
  for fIndex in Orientation.faces:
    neigh = subDomain.neighborF[fIndex]
    if (neigh > -1 and neigh != subDomain.ID):
      if neigh in data[subDomain.ID]['NeighborProcID'].keys():
        if neigh not in data:
          data[neigh] = {'NeighborProcID':{}}
        data[neigh]['NeighborProcID'][neigh] = dataRecvFace[fIndex]

  #############
  ### Edges ###
  #############
  for eIndex in Orientation.edges:
    neigh = subDomain.neighborE[eIndex]
    if (neigh > -1 and neigh != subDomain.ID):
      if neigh in data[subDomain.ID]['NeighborProcID'].keys():
        if neigh not in data:
          data[neigh] = {'NeighborProcID':{}}
        data[neigh]['NeighborProcID'][neigh] = dataRecvEdge[eIndex]

  ###############
  ### Corners ###
  ###############
  for cIndex in Orientation.corners:
    neigh = subDomain.neighborC[cIndex]
    if (neigh > -1 and neigh != subDomain.ID):
      if neigh in data[subDomain.ID]['NeighborProcID'].keys():
        if neigh not in data:
          data[neigh] = {'NeighborProcID':{}}
        data[neigh]['NeighborProcID'][neigh] = dataRecvCorner[cIndex]

  return data