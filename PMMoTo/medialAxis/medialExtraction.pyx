# cython: profile=True
# cython: linetrace=True

# distutils: language = c++
import numpy as np
from mpi4py import MPI
from .. import communication
comm = MPI.COMM_WORLD

from . cimport medialExtractionFunctions as mEFunc
from libcpp.vector cimport vector

import numpy as np
from numpy cimport npy_intp, npy_float32
cimport cython

class medialExtraction(object):
    """
    Perform  Medial Extraction 
    """

    def __init__(self,Domain,subDomain,grid,edt):
        self.Domain = Domain
        self.subDomain = subDomain
        self.grid = grid
        self.Orientation = subDomain.Orientation
        self.MA = np.copy(self.grid)
        self.edt = edt

    def getMALoopInfo(self,boundaryID,buffer,halo,grid):
        """
        Determine slices of face, edge, and corner internal boundary slices 
        """

        self.innerLoop = np.zeros([3,2],dtype=np.int64)
        self.fLoopB = np.zeros([self.Orientation.numFaces,3,2],dtype=np.int64)
        self.eLoopB = np.zeros([self.Orientation.numEdges,3,2],dtype=np.int64)
        self.cLoopB = np.zeros([self.Orientation.numCorners,3,2],dtype=np.int64)

        dim = grid.shape

        ################
        ### Internal ###
        ################
        for fIndex in self.Orientation.faces:
            fID = self.Orientation.faces[fIndex]['ID']
            for n,_ in enumerate(fID):
                self.innerLoop[n] = [buffer[n*2]+halo[n*2]+1, dim[n]-buffer[n*2+1]-halo[n*2+1]-1]

        #############
        ### Faces ###
        #############
        for fIndex in self.Orientation.faces:
            fID = self.Orientation.faces[fIndex]['ID']
            for n,ID in enumerate(fID):
                if ID == 1:
                    ### Periodic or Internal Boundary
                    if boundaryID[n*2+1] == -1 or boundaryID[n*2+1] == 2:
                        self.fLoopB[fIndex,n] = [dim[n]-halo[n*2+1]-buffer[n*2+1]-1,dim[n]-halo[n*2+1]]
                    ### Wall Boundary
                    if boundaryID[n*2+1] == 1:
                        self.fLoopB[fIndex,n] = [dim[n]-2,dim[n]-1]
                    ### No Assumption
                    if boundaryID[n*2+1] == 0:
                        self.fLoopB[fIndex,n] = [dim[n]-1,dim[n]]
                elif ID == -1:
                    ### Periodic or Internal Boundary
                    if boundaryID[n*2] == -1 or boundaryID[n*2] == 2:
                        self.fLoopB[fIndex,n] = [halo[n*2],halo[n*2]+buffer[n*2]+1]
                    ### Wall Boundary
                    if boundaryID[n*2] == 1:
                        self.fLoopB[fIndex,n] = [1,2]
                    ### No Assumption
                    if boundaryID[n*2] == 0:
                        self.fLoopB[fIndex,n] = [0,1]
                else:
                    ### Periodic or Internal Boundary
                    if boundaryID[n*2] == 2 or boundaryID[n*2] == -1:
                        self.fLoopB[fIndex,n,0] = buffer[n*2]+2*halo[n*2]
                    if boundaryID[n*2+1] == 2 or boundaryID[n*2+1] == -1:
                        self.fLoopB[fIndex,n,1] = dim[n]-buffer[n*2+1]-2*halo[n*2+1]
                    ### Wall Boundary or No Assumption
                    if boundaryID[n*2] == 1 or boundaryID[n*2] == 0:
                        self.fLoopB[fIndex,n,0] = buffer[n*2]+1
                    if boundaryID[n*2+1] == 1 or boundaryID[n*2+1] == 0:
                        self.fLoopB[fIndex,n,1] = dim[n]-buffer[n*2+1]-1
        #############

        #############
        ### Edges ###
        #############
        for eIndex in self.Orientation.edges:
            eID = self.Orientation.edges[eIndex]['ID']
            for n,ID in enumerate(eID):
                if ID == 1:
                    ### Periodic or Internal Boundary
                    if boundaryID[n*2+1] == -1 or boundaryID[n*2+1] == 2:
                        self.eLoopB[eIndex,n] = [dim[n]-buffer[n*2+1]-2*halo[n*2+1],dim[n]-buffer[n*2+1]]
                    ### Wall Boundary
                    if boundaryID[n*2+1] == 1:
                        self.eLoopB[eIndex,n] = [dim[n]-2,dim[n]-1]
                    ### No Assumption
                    if boundaryID[n*2+1] == 0:
                        self.eLoopB[eIndex,n] = [dim[n]-1,dim[n]]
                elif ID == -1:
                    ### Periodic or Internal Boundary
                    if boundaryID[n*2] == -1 or boundaryID[n*2] == 2:
                        self.eLoopB[eIndex,n] = [buffer[n*2],buffer[n*2]+2*halo[n*2]] 
                    ### Wall Boundary
                    if boundaryID[n*2] == 1:
                        self.eLoopB[eIndex,n] = [1,2]
                    ### No Assumption
                    if boundaryID[n*2] == 0:
                        self.eLoopB[eIndex,n] = [0,1]
                else:
                    ### Periodic or Internal Boundary
                    if boundaryID[n*2] == -1 or boundaryID[n*2] == 2:
                        self.eLoopB[eIndex,n,0] = buffer[n*2]+2*halo[n*2]
                    if boundaryID[n*2+1] == -1 or boundaryID[n*2+1] == 2:
                        self.eLoopB[eIndex,n,1] = dim[n]-buffer[n*2+1]-2*halo[n*2+1]
                    ### Wall Boundary or No Assumption
                    if boundaryID[n*2] == 0 or boundaryID[n*2] == 1:
                        self.eLoopB[eIndex,n,0] = buffer[n*2]+1
                    if boundaryID[n*2+1] == 0 or boundaryID[n*2+1] == 1:
                        self.eLoopB[eIndex,n,1] = dim[n]-buffer[n*2+1]-1
        ######################

        ########################
        ### Boundary Corners ###
        ########################
        for cIndex in self.Orientation.corners:
            cID = self.Orientation.corners[cIndex]['ID']
            for n,ID in enumerate(cID):
                if ID == 1:
                    ### Periodic or Internal Boundary
                    if boundaryID[n*2+1] == -1 or boundaryID[n*2+1] == 2:
                        self.cLoopB[cIndex,n] = [dim[n]-buffer[n*2+1]-2*halo[n*2+1],dim[n]-buffer[n*2+1]]
                    ### Wall Boundary
                    if boundaryID[n*2+1] == 1:
                        self.cLoopB[cIndex,n] = [dim[n]-2,dim[n]-1]
                    ### No Assumption
                    if boundaryID[n*2+1] == 0:
                        self.cLoopB[cIndex,n] = [dim[n]-1,dim[n]]
                elif ID == -1:
                    ### Periodic or Internal Boundary
                    if boundaryID[n*2] == -1 or boundaryID[n*2] == 2:
                        self.cLoopB[cIndex,n] = [buffer[n*2],buffer[n*2]+2*halo[n*2]]
                    ### Wall Boundary
                    if boundaryID[n*2] == 1:
                        self.cLoopB[cIndex,n] = [1,2]
                    ### No Assumption
                    if boundaryID[n*2] == 0:
                        self.cLoopB[cIndex,n] = [0,1]
        ########################
            

    def extractMedialAxis(self,connect = False):
    
        ## Send Buffer for MA and EDT
        sDComm = communication.Comm(Domain = self.Domain,subDomain = self.subDomain,grid = self.MA)
        haloIn = np.array([2,2,2,2,2,2],dtype=np.int8)
        self.MA,self.halo = sDComm.haloCommunication(haloIn)

        sDComm = communication.Comm(Domain = self.Domain,subDomain = self.subDomain,grid = self.edt)
        self.edt,_ = sDComm.haloCommunication(haloIn)

        ### Get Loop Info
        self.getMALoopInfo(self.subDomain.boundaryID,self.subDomain.buffer,self.halo,self.MA)

        ### Update Buffer for Communication
        ### Should Do This A Different Way bc have to Remove Later
        for n in range(self.Orientation.numFaces):
            self.subDomain.buffer[n] += self.halo[n]

        ### Update EDT Buffer. Issue If Dont!
        sDComm = communication.Comm(Domain = self.Domain,subDomain = self.subDomain,grid = self.edt)
        self.edt = sDComm.updateBuffer()

        ### External Boundaries - Wall and No Assumption
        converged = False
        while not converged:

            ### Loop through Faces
            unchangedBorders = 0
            iter = 0
            faceConverged = False
            while unchangedBorders < self.Orientation.numFaces:
                unchangedBorders = 0
                for fErode in range(0,self.Orientation.numFaces):
                    self.MA,noChange = self.erodeGrid(self.MA,self.edt,fErode,externalFaces=True)
                    if noChange:
                        unchangedBorders += 1
                iter += 1

            if iter == 1:
                faceConverged = True

            ### Update Halo
            sDComm = communication.Comm(Domain = self.Domain,subDomain = self.subDomain,grid = self.MA)
            self.MA = sDComm.updateBuffer()

            ### Loop Throgh Edges
            unchangedBorders = 0
            iter = 0
            edgeConverged = False
            while unchangedBorders < self.Orientation.numFaces:
                unchangedBorders = 0
                for fErode in range(0,self.Orientation.numFaces):
                    self.MA,noChange = self.erodeGrid(self.MA,self.edt,fErode,externalEdges=True)
                    if noChange:
                        unchangedBorders += 1
                iter += 1

            if iter == 1:
                edgeConverged = True

            ### Update Halo
            sDComm = communication.Comm(Domain = self.Domain,subDomain = self.subDomain,grid = self.MA)
            self.MA = sDComm.updateBuffer()

            ### Loop Throgh Corners
            unchangedBorders = 0
            iter = 0
            cornerConverged = True
            while unchangedBorders < self.Orientation.numFaces:
                unchangedBorders = 0
                for fErode in range(0,self.Orientation.numFaces):
                    self.MA,noChange = self.erodeGrid(self.MA,self.edt,fErode,externalCorners = True)
                    if noChange:
                        unchangedBorders += 1
                iter += 1

            if iter == 1:
                cornerConverged = True

            ### Update Halo
            sDComm = communication.Comm(Domain = self.Domain,subDomain = self.subDomain,grid = self.MA)
            self.MA = sDComm.updateBuffer()

            ### Check For Convergence for own Proc
            procConverged = False
            if faceConverged and edgeConverged and cornerConverged:
                procConverged = True

            ### Check For Convergence for all Procs
            allProcsConverged = comm.gather(procConverged, root=0)
            if self.subDomain.ID == 0:
                if np.sum(allProcsConverged) == len(allProcsConverged):
                    converged = True
            comm.barrier()
            converged = comm.bcast(converged, root=0)

        ### Internal Nodes and Periodic / Internal Voxels
        converged = False
        while not converged:

            ### Loop Through Only External Boundary Conditions
            if 0 in self.subDomain.boundaryID or 1 in self.subDomain.boundaryID:
                unchangedBorders = 0
                while unchangedBorders < self.Orientation.numFaces:
                    unchangedBorders = 0
                    for fErode in range(self.Orientation.numFaces):
                        self.MA,noChange = self.erodeGrid(self.MA,self.edt,fErode)
                        if noChange:
                            unchangedBorders += 1



            ### Loop Through Internal Voxels
            unchangedBorders = 0
            iter = 0
            internalConverged = False
            while unchangedBorders < self.Orientation.numFaces:
                unchangedBorders = 0
                for fErode in range(self.Orientation.numFaces):
                    self.MA,noChange = self.erodeGrid(self.MA,self.edt,fErode,internal=True)
                    if noChange:
                        unchangedBorders += 1
                iter += 1

            if iter == 1:
                internalConverged = True

            ### Update Halo
            sDComm = communication.Comm(Domain = self.Domain,subDomain = self.subDomain,grid = self.MA)
            self.MA = sDComm.updateBuffer()

            ### Loop through Faces
            unchangedBorders = 0
            iter = 0
            faceConverged = False
            while unchangedBorders < self.Orientation.numFaces:
                unchangedBorders = 0
                for fErode in range(0,self.Orientation.numFaces):
                    self.MA,noChange = self.erodeGrid(self.MA,self.edt,fErode,faces=True)
                    if noChange:
                        unchangedBorders += 1
                iter += 1

            if iter == 1:
                faceConverged = True

            ### Update Halo
            sDComm = communication.Comm(Domain = self.Domain,subDomain = self.subDomain,grid = self.MA)
            self.MA = sDComm.updateBuffer()

            ### Loop Throgh Edges
            unchangedBorders = 0
            iter = 0
            edgeConverged = False
            while unchangedBorders < self.Orientation.numFaces:
                unchangedBorders = 0
                for fErode in range(0,self.Orientation.numFaces):
                    self.MA,noChange = self.erodeGrid(self.MA,self.edt,fErode,edges=True)
                    if noChange:
                        unchangedBorders += 1
                iter += 1

            if iter == 1:
                edgeConverged = True

            ### Update Halo
            sDComm = communication.Comm(Domain = self.Domain,subDomain = self.subDomain,grid = self.MA)
            self.MA = sDComm.updateBuffer()

            ### Loop Throgh Corners
            unchangedBorders = 0
            iter = 0
            cornerConverged = True
            while unchangedBorders < self.Orientation.numFaces:
                unchangedBorders = 0
                for fErode in range(0,self.Orientation.numFaces):
                    self.MA,noChange = self.erodeGrid(self.MA,self.edt,fErode,corners=True)
                    if noChange:
                        unchangedBorders += 1
                iter += 1

            if iter == 1:
                cornerConverged = True

            ### Update Halo
            sDComm = communication.Comm(Domain = self.Domain,subDomain = self.subDomain,grid = self.MA)
            self.MA = sDComm.updateBuffer()

            ### Check For Convergence for own Proc
            procConverged = False
            if internalConverged and faceConverged and edgeConverged and cornerConverged:
                procConverged = True

            ### Check For Convergence for all Procs
            allProcsConverged = comm.gather(procConverged, root=0)
            if self.subDomain.ID == 0:
                if np.sum(allProcsConverged) == len(allProcsConverged):
                    converged = True
            comm.barrier()
            converged = comm.bcast(converged, root=0)
        
        ### Reset Buffer
        for n in range(self.Orientation.numFaces):
            self.subDomain.buffer[n] -= self.halo[n]

        ### Grab Medial Axis with Single and Two Buffer to 
        haloPadNeigh = np.zeros_like(self.halo)
        haloPadNeighNot = np.ones_like(self.halo)
        for n in range(self.Orientation.numFaces):
            if self.halo[n] > 0:
                haloPadNeigh[n] = 1
                haloPadNeighNot[n] = 0

        dim = self.MA.shape

        if not connect:
            self.MA = self.MA[self.halo[0]:dim[0]-self.halo[1],
                              self.halo[2]:dim[1]-self.halo[3],
                              self.halo[4]:dim[2]-self.halo[5]]

            return np.ascontiguousarray(self.MA)

        else:
            self.MA = self.MA[self.halo[0] - haloPadNeigh[0] : dim[0] - self.halo[1] + haloPadNeigh[1],
                              self.halo[2] - haloPadNeigh[2] : dim[1] - self.halo[3] + haloPadNeigh[3],
                              self.halo[4] - haloPadNeigh[4] : dim[2] - self.halo[5] + haloPadNeigh[5]]
            
            return np.ascontiguousarray(self.MA),haloPadNeigh,haloPadNeighNot



    @cython.boundscheck(False)
    @cython.wraparound(False)
    def erodeGrid(self,
                mEFunc.pixel_type[:, :, ::1] img not None,
                npy_float32 [:, :, ::1] edt,
                int fErode, 
                internal = False,
                faces = False,
                edges = False,
                corners = False,
                externalFaces = False,
                externalEdges = False,
                externalCorners = False):

        cdef:
            npy_intp x, y, z, ID
            bint no_change = True

            # list simple_border_points
            vector[mEFunc.coordinate] simple_border_points
            mEFunc.coordinate point
            Py_ssize_t num_border_points, i, j
            mEFunc.pixel_type neighb[27]

        # ### Get Face Simple Points 
        if internal:
            mEFunc.findSimplePoints(img,edt,fErode,self.innerLoop,simple_border_points)

        if faces:
            for f in self.Orientation.faces:
                if self.subDomain.neighborF[f] > -1:
                    mEFunc.findSimplePoints(img,edt,fErode,self.fLoopB[f],simple_border_points)

        if edges:
            for e in self.Orientation.edges:
                if self.subDomain.neighborE[e] > -1:
                    mEFunc.findSimplePoints(img,edt,fErode,self.eLoopB[e],simple_border_points)

        if corners:
            for c in self.Orientation.corners:
                if self.subDomain.neighborC[c] > -1:
                    mEFunc.findSimplePoints(img,edt,fErode,self.cLoopB[c],simple_border_points)

        if externalFaces:
            for f in self.Orientation.faces:
                if self.subDomain.neighborF[f] == -1:
                    mEFunc.findSimplePoints(img,edt,fErode,self.fLoopB[f],simple_border_points)
                if self.subDomain.neighborF[f] == -2:
                    faceSimplePoints(f,img,edt,fErode,self.fLoopB[f],simple_border_points)

        if externalEdges:
            for e in self.Orientation.edges:
                if self.subDomain.neighborE[e] == -1:
                    mEFunc.findSimplePoints(img,edt,fErode,self.eLoopB[e],simple_border_points)
                if self.subDomain.neighborE[e] == -2:
                    edgeSimplePoints(e,img,edt,fErode,self.eLoopB[e],simple_border_points)
                if self.subDomain.neighborE[e] == -3:
                    faceSimplePoints(self.subDomain.externalE[e],img,edt,fErode,self.eLoopB[e],simple_border_points)

        if externalCorners:
            for c in self.Orientation.corners:
                if self.subDomain.neighborC[c] == -1:
                    mEFunc.findSimplePoints(img,edt,fErode,self.cLoopB[c],simple_border_points)
                if self.subDomain.neighborC[c] == -2:
                    cornerSimplePoints(c,img,edt,fErode,self.cLoopB[c],simple_border_points)
                if self.subDomain.neighborC[c] == -3:
                    faceSimplePoints(self.subDomain.externalC[c],img,edt,fErode,self.cLoopB[c],simple_border_points)
                if self.subDomain.neighborC[c] == -4:
                    edgeSimplePoints(self.subDomain.externalC[c],img,edt,fErode,self.cLoopB[c],simple_border_points)


        num_border_points = simple_border_points.size()        
        simple_border_points = sorted(simple_border_points, key=lambda d: d['edt'])
        for i in range(num_border_points):
            point = simple_border_points[i]
            x = point.x
            y = point.y
            z = point.z
            ID = point.ID

            if ID == 0:
                mEFunc.get_neighborhood(img, x, y, z, neighb)
            elif ID > 0:
                mEFunc.get_neighborhood_limited(img, x, y, z, ID, neighb)

            if mEFunc.is_simple_point(neighb):
                img[x, y, z] = 0
                no_change = False


        return np.ascontiguousarray(img),no_change



cdef void faceSimplePoints(int ID,
                           mEFunc.pixel_type[:, :, ::1] img,
                           npy_float32[:, :, ::1] edt,
                           int fErode,
                           npy_intp[:,:] loop,
                           vector[mEFunc.coordinate] & simple_border_points):


    if ID == 0:
        mEFunc.find_simple_point_candidates_faces_0(img,edt,fErode,loop,simple_border_points)
    if ID == 1:
        mEFunc.find_simple_point_candidates_faces_1(img,edt,fErode,loop,simple_border_points)
    if ID == 2:
        mEFunc.find_simple_point_candidates_faces_2(img,edt,fErode,loop,simple_border_points)
    if ID == 3:
        mEFunc.find_simple_point_candidates_faces_3(img,edt,fErode,loop,simple_border_points)
    if ID == 4:
        mEFunc.find_simple_point_candidates_faces_4(img,edt,fErode,loop,simple_border_points)
    if ID == 5:
        mEFunc.find_simple_point_candidates_faces_5(img,edt,fErode,loop,simple_border_points)


cdef void edgeSimplePoints(int ID,
                           mEFunc.pixel_type[:, :, ::1] img,
                           npy_float32[:, :, ::1] edt,
                           int fErode,
                           npy_intp[:,:] loop,
                           vector[mEFunc.coordinate] & simple_border_points):

    if ID == 0:
        mEFunc.find_simple_point_candidates_edges_0(img,edt,fErode,loop,simple_border_points)
    if ID == 1:
        mEFunc.find_simple_point_candidates_edges_1(img,edt,fErode,loop,simple_border_points)
    if ID == 2:
        mEFunc.find_simple_point_candidates_edges_2(img,edt,fErode,loop,simple_border_points)
    if ID == 3:
        mEFunc.find_simple_point_candidates_edges_3(img,edt,fErode,loop,simple_border_points)
    if ID == 4:
        mEFunc.find_simple_point_candidates_edges_4(img,edt,fErode,loop,simple_border_points)
    if ID == 5:
        mEFunc.find_simple_point_candidates_edges_5(img,edt,fErode,loop,simple_border_points)
    if ID == 6:
        mEFunc.find_simple_point_candidates_edges_6(img,edt,fErode,loop,simple_border_points)
    if ID == 7:
        mEFunc.find_simple_point_candidates_edges_7(img,edt,fErode,loop,simple_border_points)
    if ID == 8:
        mEFunc.find_simple_point_candidates_edges_8(img,edt,fErode,loop,simple_border_points)
    if ID == 9:
        mEFunc.find_simple_point_candidates_edges_9(img,edt,fErode,loop,simple_border_points)
    if ID == 10:
        mEFunc.find_simple_point_candidates_edges_10(img,edt,fErode,loop,simple_border_points)
    if ID == 11:
        mEFunc.find_simple_point_candidates_edges_11(img,edt,fErode,loop,simple_border_points)


cdef void cornerSimplePoints(int ID,
                             mEFunc.pixel_type[:, :, ::1] img,
                             npy_float32[:, :, ::1] edt,
                             int fErode,
                             npy_intp[:,:] loop,
                             vector[mEFunc.coordinate] & simple_border_points):


    if ID == 0:
        mEFunc.find_simple_point_candidates_corners_0(img,edt,fErode,loop,simple_border_points)
    if ID == 1:
        mEFunc.find_simple_point_candidates_corners_1(img,edt,fErode,loop,simple_border_points)
    if ID == 2:
        mEFunc.find_simple_point_candidates_corners_2(img,edt,fErode,loop,simple_border_points)
    if ID == 3:
        mEFunc.find_simple_point_candidates_corners_3(img,edt,fErode,loop,simple_border_points)
    if ID == 4:
        mEFunc.find_simple_point_candidates_corners_4(img,edt,fErode,loop,simple_border_points)
    if ID == 5:
        mEFunc.find_simple_point_candidates_corners_5(img,edt,fErode,loop,simple_border_points)
    if ID == 6:
        mEFunc.find_simple_point_candidates_corners_6(img,edt,fErode,loop,simple_border_points)
    if ID == 7:
        mEFunc.find_simple_point_candidates_corners_7(img,edt,fErode,loop,simple_border_points)