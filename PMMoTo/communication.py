from . import Orientation
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
        self.grid = grid
        self.dim = self.grid.shape
        self.halo = np.zeros([6],dtype=np.int64)
        self.haloData = {self.subDomain.ID: {'NeighborProcID':{}}} 

    def buffer_pack(self):
        """
        Grab The Slices (based on Buffer Size [1,1,1]) to pack and send to Neighbors
        for faces, edges and corners.
        """

        sendFSlices,sendESlices,sendCSlices = Orientation.getSendBufferSlices(self.subDomain.buffer,self.dim)

        halo_data = {}
        for n_proc in self.subDomain.n_procs:
            if n_proc > -1 and n_proc != self.subDomain.ID:
                halo_data[n_proc] = {'ID':{}}
                for feature in self.subDomain.n_procs[n_proc]:
                    halo_data[n_proc]['ID'][feature] = None

        slices = sendFSlices
        for face in self.subDomain.faces:
            if face.n_proc > -1 and face.n_proc != self.subDomain.ID:
                s = slices[face.ID,:]
                halo_data[face.n_proc]['ID'][face.info['ID']] = self.grid[s[0],s[1],s[2]]

        slices = sendESlices
        for edge in self.subDomain.edges:
            if edge.n_proc > -1 and edge.n_proc != self.subDomain.ID:
                s = slices[edge.ID,:]
                halo_data[edge.n_proc]['ID'][edge.info['ID']] = self.grid[s[0],s[1],s[2]]

        slices = sendCSlices
        for corner in self.subDomain.corners:
            if corner.n_proc > -1 and corner.n_proc != self.subDomain.ID:
                s = slices[corner.ID,:]
                halo_data[corner.n_proc]['ID'][corner.info['ID']] = self.grid[s[0],s[1],s[2]]

        return halo_data


    def bufferComm(self,data):

        face_recv,edge_recv,corner_recv = subDomainComm(self.subDomain,data)
        return face_recv,edge_recv,corner_recv

    def buffer_unpack(self,face_recv,edge_recv,corner_recv):
        """
        Unpack buffer information and account for serial periodic boundary conditions
        """
        halo_grid = np.copy(self.grid)
        recvFSlices,recvESlices,recvCSlices = Orientation.getRecieveBufferSlices(self.subDomain.buffer,halo_grid.shape)
        sendFSlices,sendESlices,sendCSlices = Orientation.getSendBufferSlices(self.subDomain.buffer,self.dim)

        #### Faces ####
        r_slices = recvFSlices
        s_slices = sendFSlices
        for face in self.subDomain.faces:

            if (face.n_proc > -1 and face.n_proc != self.subDomain.ID):
                r_s = r_slices[face.ID,:]
                halo_grid[r_s[0],r_s[1],r_s[2]] = face_recv[face.ID]['ID'][face.opp_info['ID']]

            elif face.n_proc == self.subDomain.ID:
                r_s = r_slices[face.ID,:]
                s_s = s_slices[face.info['oppIndex'],:]
                halo_grid[r_s[0],r_s[1],r_s[2]] = self.grid[s_s[0],s_s[1],s_s[2]]

        #### Edges ####
        r_slices = recvESlices
        s_slices = sendESlices
        for edge in self.subDomain.edges:

            if (edge.n_proc > -1 and edge.n_proc != self.subDomain.ID):
                r_s = r_slices[edge.ID,:]
                halo_grid[r_s[0],r_s[1],r_s[2]] = edge_recv[edge.ID]['ID'][edge.opp_info['ID']]

            elif edge.n_proc == self.subDomain.ID:
                r_s = r_slices[edge.ID,:]
                s_s = s_slices[edge.info['oppIndex'],:]
                halo_grid[r_s[0],r_s[1],r_s[2]] = self.grid[s_s[0],s_s[1],s_s[2]]

        #### Corners ####
        r_slices = recvCSlices
        s_slices = sendCSlices
        for corner in self.subDomain.corners:

            if (corner.n_proc > -1 and corner.n_proc != self.subDomain.ID):
                r_s = r_slices[corner.ID,:]
                halo_grid[r_s[0],r_s[1],r_s[2]] = corner_recv[corner.ID]['ID'][corner.opp_info['ID']]

            elif corner.n_proc == self.subDomain.ID:
                r_s = r_slices[corner.ID,:]
                s_s = s_slices[corner.info['oppIndex'],:]
                halo_grid[r_s[0],r_s[1],r_s[2]] = self.grid[s_s[0],s_s[1],s_s[2]]
        
        return halo_grid

    def haloCommPack(self,haloIn):
        """
        Grab The Slices (based on Size) to pack and send to Neighbors
        for faces, edges and corners
        """

        sendFSlices,sendESlices,sendCSlices = Orientation.getSendSlices(haloIn,self.subDomain.buffer,self.dim)

        self.slices = Orientation.sendFSlices
        for fIndex in Orientation.faces:
            neigh = self.subDomain.neighborF[fIndex]
            if neigh > -1:
                self.halo[fIndex]= haloIn[fIndex]
                if neigh not in self.haloData[self.subDomain.ID]['NeighborProcID'].keys():
                    self.haloData[self.subDomain.ID]['NeighborProcID'][neigh] = {'Index':{}}
                self.haloData[self.subDomain.ID]['NeighborProcID'][neigh]['Index'][fIndex] = self.grid[self.slices[fIndex,0],self.slices[fIndex,1],self.slices[fIndex,2]]


        self.slices = Orientation.sendESlices
        for eIndex in Orientation.edges:
            neigh = self.subDomain.neighborE[eIndex]
            if neigh > -1:
                if neigh not in self.haloData[self.subDomain.ID]['NeighborProcID'].keys():
                    self.haloData[self.subDomain.ID]['NeighborProcID'][neigh] = {'Index':{}}
                self.haloData[self.subDomain.ID]['NeighborProcID'][neigh]['Index'][eIndex] = self.grid[self.slices[eIndex,0],self.slices[eIndex,1],self.slices[eIndex,2]] 

        self.slices = Orientation.sendCSlices
        for cIndex in Orientation.corners:
            neigh = self.subDomain.neighborC[cIndex]
            if neigh > -1:
                if neigh not in self.haloData[self.subDomain.ID]['NeighborProcID'].keys():
                    self.haloData[self.subDomain.ID]['NeighborProcID'][neigh] = {'Index':{}}
                self.haloData[self.subDomain.ID]['NeighborProcID'][neigh]['Index'][cIndex] = self.grid[self.slices[cIndex,0],self.slices[cIndex,1],self.slices[cIndex,2]]

        self.haloGrid = np.pad(self.grid, ( (self.halo[0], self.halo[1]), (self.halo[2], self.halo[3]), (self.halo[4], self.halo[5]) ), 'constant', constant_values=255)

    def haloComm(self):
        self.dataRecvFace,self.dataRecvEdge,self.dataRecvCorner = subDomainComm(self.subDomain,self.haloData[self.subDomain.ID]['NeighborProcID'])

    def haloCommUnpack(self):
        Orientation.getRecieveSlices(self.halo,self.subDomain.buffer,self.haloGrid.shape)

        #### Faces ####
        self.slices = Orientation.recvFSlices
        for fIndex in Orientation.faces:
            neigh = self.subDomain.neighborF[fIndex]
            oppIndex = Orientation.faces[fIndex]['oppIndex']
            if (neigh > -1 and neigh != self.subDomain.ID):
                if neigh in self.haloData[self.subDomain.ID]['NeighborProcID'].keys():
                    if neigh not in self.haloData:
                        self.haloData[neigh] = {'NeighborProcID':{}}
                    self.haloData[neigh]['NeighborProcID'][neigh] = self.dataRecvFace[fIndex]['Index'][oppIndex]
                    self.haloGrid[self.slices[fIndex,0],self.slices[fIndex,1],self.slices[fIndex,2]] = self.haloData[neigh]['NeighborProcID'][neigh]

        #### Edges ####
        self.slices = Orientation.recvESlices
        for eIndex in Orientation.edges:
            neigh = self.subDomain.neighborE[eIndex]
            oppIndex = Orientation.edges[eIndex]['oppIndex']
            if (neigh > -1 and neigh != self.subDomain.ID):
                if neigh in self.haloData[self.subDomain.ID]['NeighborProcID'].keys():
                    if neigh not in self.haloData:
                        self.haloData[neigh] = {'NeighborProcID':{}}
                    self.haloData[neigh]['NeighborProcID'][neigh] = self.dataRecvEdge[eIndex]['Index'][oppIndex]
                    self.haloGrid[self.slices[eIndex,0],self.slices[eIndex,1],self.slices[eIndex,2]] = self.haloData[neigh]['NeighborProcID'][neigh]

        #### Corners ####
        self.slices = Orientation.recvCSlices
        for cIndex in Orientation.corners:
            neigh = self.subDomain.neighborC[cIndex]
            oppIndex = Orientation.corners[cIndex]['oppIndex']
            if (neigh > -1 and neigh != self.subDomain.ID):
                if neigh in self.haloData[self.subDomain.ID]['NeighborProcID'].keys():
                    if neigh not in self.haloData:
                        self.haloData[neigh] = {'NeighborProcID':{}}
                    self.haloData[neigh]['NeighborProcID'][neigh] = self.dataRecvCorner[cIndex]['Index'][oppIndex]
                    self.haloGrid[self.slices[cIndex,0],self.slices[cIndex,1],self.slices[cIndex,2]] = self.haloData[neigh]['NeighborProcID'][neigh]

    def EDTCommPack(self,face_solids,edge_solids,corner_solids):
        """
        Send boundary solids data. Send data is OPPOSITE from ID. 
        """
        self.send_data = {}
        for n_proc in self.subDomain.n_procs:
            if n_proc != self.subDomain.ID:
                self.send_data[n_proc] = {'ID':{}}
                for feature in self.subDomain.n_procs[n_proc]:
                    self.send_data[n_proc]['ID'][feature] = None
        
        for face in self.subDomain.faces:
            if face.n_proc != self.subDomain.ID:
                self.send_data[face.n_proc]['ID'][face.info['ID']] = face_solids[face.ID]

        for edge in self.subDomain.edges:
            if edge.n_proc != self.subDomain.ID:
                self.send_data[edge.n_proc]['ID'][edge.info['ID']] = edge_solids[edge.ID]

        for corner in self.subDomain.corners:
            if corner.n_proc != self.subDomain.ID:
                self.send_data[corner.n_proc]['ID'][corner.info['ID']] = corner_solids[corner.ID]

    def EDTComm(self):
        
        if self.send_data:
            self.dataRecvFace,self.dataRecvEdge,self.dataRecvCorner = subDomainComm(self.subDomain,self.send_data)

    def EDTCommUnpackNEW(self,external_solids,face_solids,edge_solids,corner_solids):

        #### FACE ####
        for face in self.subDomain.faces:
            period_correction = face.periodic_correction*self.Domain.domainLength
            if (face.n_proc > -1 and face.n_proc  != self.subDomain.ID):
                opp_ID = Orientation.faces[face.info['oppIndex']]['ID']
                external_solids[face.info['ID']] = self.dataRecvFace[face.ID]['ID'][opp_ID] - period_correction
            elif (face.n_proc == self.subDomain.ID):
                external_solids[face.info['ID']] = face_solids[face.info['oppIndex']] - period_correction

        #### EDGE ####
        for edge in self.subDomain.edges:
            period_correction = edge.periodic_correction*self.Domain.domainLength
            if (edge.n_proc > -1 and edge.n_proc != self.subDomain.ID):
                opp_ID = Orientation.edges[edge.info['oppIndex']]['ID']
                external_solids[edge.info['ID']] = self.dataRecvEdge[edge.ID]['ID'][opp_ID] - period_correction
            elif (edge.n_proc == self.subDomain.ID):
                external_solids[edge.info['ID']] = edge_solids[edge.info['oppIndex']] - period_correction

        #### Corner ####
        for corner in self.subDomain.corners:
            period_correction = corner.periodic_correction*self.Domain.domainLength
            if (corner.n_proc > -1 and corner.n_proc != self.subDomain.ID):
                opp_ID = Orientation.corners[corner.info['oppIndex']]['ID']
                external_solids[corner.info['ID']] = self.dataRecvCorner[corner.ID]['ID'][opp_ID] - period_correction
            elif (corner.n_proc == self.subDomain.ID):
                external_solids[corner.info['ID']] = corner_solids[corner.info['oppIndex']] - period_correction


    def update_buffer(self):
        halo_data = self.buffer_pack()
        f,e,c = self.bufferComm(halo_data)
        halo_grid = self.buffer_unpack(f,e,c)
        return halo_grid

    def haloCommunication(self,haloIn):
        self.haloCommPack(haloIn)
        self.haloComm()
        self.haloCommUnpack()
        return self.haloGrid,self.halo

    def EDTCommunication(self,external_solids,face_solids,edge_solids,corner_solids):
        self.EDTCommPack(face_solids,edge_solids,corner_solids)
        self.EDTComm()
        self.EDTCommUnpackNEW(external_solids,face_solids,edge_solids,corner_solids)
        return external_solids
    

def subDomainComm(subDomain,sendData):

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

def set_COMM(subDomain,data):
  """
  Transmit data to Neighboring Processors
  """
  dataRecvFace,dataRecvEdge,dataRecvCorner = subDomainComm(subDomain,data[subDomain.ID]['nProcID'])

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