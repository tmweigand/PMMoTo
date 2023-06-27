# distutils: language = c++
import numpy as np
from mpi4py import MPI
from .. import communication
from .. import nodes
from .. import sets
from .. import dataOutput
comm = MPI.COMM_WORLD
import math

from . cimport medialExtractionFunctions as mEFunc
from libc.string cimport memcpy
from libcpp.vector cimport vector
from libc.stdio cimport printf
from libcpp cimport bool

import numpy as np
from numpy cimport npy_intp, npy_int8, npy_uint8, ndarray, npy_float32
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
        self.numIterations = 4

        self.fLoopB,self.fLoopI,self.eLoopB,self.eLoopI,self.cLoopB,self.cLoopI = self.Orientation.getMALoopInfo(self.subDomain.buffer,self.grid)

    def medialLoopScheme(self):
        """
        Create Looping Scheme for Procs to Avoid OverLap
        Alternate Between Face and Face/Edges/Two Corners
        Four Iterations to Cover All Vocels 
        Differentiate Between Face Direction (+1/-1)
        """
        self.fInfo = np.zeros([self.Orientation.numFaces],dtype=np.int8)
        self.eInfo = np.zeros([self.numIterations,self.Orientation.numFaces,self.Orientation.numEdges],dtype=np.int8)
        self.cInfo = np.zeros([self.numIterations,self.Orientation.numFaces,self.Orientation.numCorners],dtype=np.int8)

    
        for f in range(self.Orientation.numFaces):

            erodeAxis = self.Orientation.faces[f]['argOrder']
            dir = self.Orientation.faces[f]['dir']

            #### Determine if Proc is Even or Odd
            subID = self.subDomain.subID[erodeAxis[1]] + self.subDomain.subID[erodeAxis[2]]
            if subID % 2 == 0:
                evenID = True
            else:
                evenID = False


            # Perform MA Extraction on Internal and External Boundaries
            if self.subDomain.boundaryID[f] != 0: #Padded?
                self.fInfo[f] = 1

            ### Get The Edges of the Erode Face
            for nI in range(self.numIterations):

                if dir < 0:
                    ### Even Procs
                    if nI % 2 == 0 and evenID:
                        for e in self.Orientation.edges:
                            if f in self.Orientation.edges[e]['faceIndex']:
                                self.eInfo[nI,f,e] = 1

                    ### Odd Procs
                    elif nI % 2 != 0  and not evenID:
                        for e in self.Orientation.edges:
                            if f in self.Orientation.edges[e]['faceIndex']:
                               self.eInfo[nI,f,e] = 1 
                else:
                    ### Odd Procs
                    if nI % 2 == 0 and not evenID:
                        for e in self.Orientation.edges:
                            if f in self.Orientation.edges[e]['faceIndex']:
                                self.eInfo[nI,f,e] = 1

                    ### Even Procs
                    elif nI % 2 != 0 and evenID:
                        for e in self.Orientation.edges:
                            if f in self.Orientation.edges[e]['faceIndex']:
                                self.eInfo[nI,f,e] = 1 
            
            ### Get The Corners of the Erode Face
            for nI in range(self.numIterations):

                if dir < 0:
                    ### Even Proc
                    if nI % 2 == 0 and evenID:
                        if nI == 0:
                            for c in self.Orientation.corners:
                                if f in self.Orientation.corners[c]['faceIndex']:
                                    if self.Orientation.corners[c]['ID'][erodeAxis[1]] < 0:
                                        self.cInfo[nI,f,c] = 1
                        if nI == 2:
                            for c in self.Orientation.corners:
                                if f in self.Orientation.corners[c]['faceIndex']:
                                    if self.Orientation.corners[c]['ID'][erodeAxis[1]] > 0:
                                        self.cInfo[nI,f,c] = 1

                    ### Odd Proc
                    elif nI % 2 != 0  and not evenID:
                        if nI == 1:
                            for c in self.Orientation.corners:
                                if f in self.Orientation.corners[c]['faceIndex']:
                                    if self.Orientation.corners[c]['ID'][erodeAxis[1]] < 0:
                                        self.cInfo[nI,f,c] = 1
                        if nI == 3:
                            for c in self.Orientation.corners:
                                if f in self.Orientation.corners[c]['faceIndex']:
                                    if self.Orientation.corners[c]['ID'][erodeAxis[1]] > 0:
                                        self.cInfo[nI,f,c] = 1
                
                else:
                    ### Odd Proc
                    if nI % 2 == 0 and not evenID:
                        if nI == 0:
                            for c in self.Orientation.corners:
                                if f in self.Orientation.corners[c]['faceIndex']:
                                    if self.Orientation.corners[c]['ID'][erodeAxis[1]] < 0:
                                        self.cInfo[nI,f,c] = 1
                        if nI == 2:
                            for c in self.Orientation.corners:
                                if f in self.Orientation.corners[c]['faceIndex']:
                                    if self.Orientation.corners[c]['ID'][erodeAxis[1]] > 0:
                                        self.cInfo[nI,f,c] = 1

                    ### Even Proc
                    elif nI % 2 != 0  and evenID:
                        if nI == 1:
                            for c in self.Orientation.corners:
                                if f in self.Orientation.corners[c]['faceIndex']:
                                    if self.Orientation.corners[c]['ID'][erodeAxis[1]] < 0:
                                        self.cInfo[nI,f,c] = 1
                        if nI == 3:
                            for c in self.Orientation.corners:
                                if f in self.Orientation.corners[c]['faceIndex']:
                                    if self.Orientation.corners[c]['ID'][erodeAxis[1]] > 0:
                                        self.cInfo[nI,f,c] = 1 


    def medialLoopSchemeNEW(self):
        """
        Create Looping Scheme for Procs to Avoid OverLap
        Alternate Between Face and Face/Edges/Two Corners
        Four Iterations to Cover All Vocels 
        Differentiate Between Face Direction (+1/-1)
        """

        self.fInfo = np.zeros([self.Orientation.numFaces,self.Orientation.numFaces],dtype=np.int8)
        self.eInfo = np.zeros([self.Orientation.numFaces,self.Orientation.numEdges],dtype=np.int8)
        self.cInfo = np.zeros([self.Orientation.numFaces,self.Orientation.numCorners],dtype=np.int8)



        for fErode in range(self.Orientation.numFaces):

            erodeAxis = self.Orientation.faces[fErode]['argOrder'][0]
            ID = self.Orientation.faces[fErode]['ID'][erodeAxis]

            for f in range(self.Orientation.numFaces):
                if f != fErode:
                    self.fInfo[fErode,f] = 1

            for e in range(self.Orientation.numEdges):
                if self.Orientation.edges[e]['ID'][erodeAxis] == ID:
                    pass
                else:
                    self.eInfo[fErode,e] = 1

            for c in range(self.Orientation.numCorners):
                if self.Orientation.corners[c]['ID'][erodeAxis] == ID:
                    pass
                else:
                    self.cInfo[fErode,c] = 1

            

    def skeletonizeAxis_test(self):

        #self.medialLoopScheme()
        self.medialLoopSchemeNEW()
    
        ## Send Buffer for MA and EDT
        sDComm = communication.Comm(Domain = self.Domain,subDomain = self.subDomain,grid = self.MA)
        haloIn = np.array([2,2,2,2,2,2],dtype=np.int8)
        self.MA,self.halo = sDComm.haloCommunication(haloIn)
        sDComm = communication.Comm(Domain = self.Domain,subDomain = self.subDomain,grid = self.edt)
        self.edt,_ = sDComm.haloCommunication(haloIn)

        ### Get Loop Info
        self.fLoopB,self.fLoopI,self.eLoopB,self.eLoopI,self.cLoopB,self.cLoopI = self.Orientation.getMALoopInfoTEST(self.subDomain.buffer,haloIn,self.MA)
        self.subDomain.buffer = (haloIn[0]+1)*np.ones(6,dtype=np.int8)

        sDComm = communication.Comm(Domain = self.Domain,subDomain = self.subDomain,grid = self.edt)
        self.edt = sDComm.updateBuffer()

        converged = False
        while not converged:

            ### Loop Through Internal Voxels
            unchangedBorders = 0
            iter = 0
            internalConverged = False
            while unchangedBorders < self.Orientation.numFaces:
                unchangedBorders = 0
                for fErode in range(self.Orientation.numFaces):
                    self.MA,noChange = self.getInternalBoundariesNO(self.MA,self.edt,fErode,boundary = 0,internal=True)
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
                    self.MA,noChange = self.getInternalBoundariesNO(self.MA,self.edt,fErode,boundary = 0,faces=True)
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
                    self.MA,noChange = self.getInternalBoundariesNO(self.MA,self.edt,fErode,boundary = 0,edges=True)
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
                    self.MA,noChange = self.getInternalBoundariesNO(self.MA,self.edt,fErode,boundary = 0,corners=True)
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
                print(allProcsConverged)
                if np.sum(allProcsConverged) == len(allProcsConverged):
                    converged = True
            comm.barrier()
            converged = comm.bcast(converged, root=0)
            
        self.subDomain.buffer = np.ones(6,dtype=np.int8)  
        dim = self.MA.shape
        self.MA = self.MA[self.halo[0]:dim[0]-self.halo[1],
                          self.halo[2]:dim[1]-self.halo[3],
                          self.halo[4]:dim[2]-self.halo[5]]

        return np.ascontiguousarray(self.MA)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def getInternalBoundariesNO(self,
                                mEFunc.pixel_type[:, :, ::1] img not None,
                                npy_float32 [:, :, ::1] edt,
                                int fErode, 
                                int boundary,
                                internal = False,
                                faces = False,
                                edges = False,
                                corners = False):

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
            mEFunc.find_simple_point_candidates_TEST2(img,edt,fErode,self.fLoopI,simple_border_points)

        if faces:
            for f in self.Orientation.faces:
                mEFunc.find_simple_point_candidates_TEST2(img,edt,fErode,self.fLoopB[f],simple_border_points)

        if edges:
            for e in self.Orientation.edges:
                mEFunc.find_simple_point_candidates_TEST2(img,edt,fErode,self.eLoopI[e],simple_border_points)

        if corners:
            for c in self.Orientation.corners:
                mEFunc.find_simple_point_candidates_TEST2(img,edt,fErode,self.cLoopI[c],simple_border_points)

        # if boundary:
        #     for f in self.Orientation.faces:
        #         if f == 0 and self.fInfo[fErode,f]:
        #             mEFunc.find_simple_point_candidates_faces_0(img,edt,fErode,self.fLoopB[f],simple_border_points)
        #         if f == 1 and self.fInfo[fErode,f]:
        #             mEFunc.find_simple_point_candidates_faces_1(img,edt,fErode,self.fLoopB[f],simple_border_points)
        #         if f == 2 and self.fInfo[fErode,f]:
        #             mEFunc.find_simple_point_candidates_faces_2(img,edt,fErode,self.fLoopB[f],simple_border_points)
        #         if f == 3 and self.fInfo[fErode,f]:
        #             mEFunc.find_simple_point_candidates_faces_3(img,edt,fErode,self.fLoopB[f],simple_border_points)
        #         if f == 4 and self.fInfo[fErode,f]:
        #             mEFunc.find_simple_point_candidates_faces_4(img,edt,fErode,self.fLoopB[f],simple_border_points)
        #         if f == 5 and self.fInfo[fErode,f]:
        #             mEFunc.find_simple_point_candidates_faces_5(img,edt,fErode,self.fLoopB[f],simple_border_points)

        #     for e in self.Orientation.edges:
        #         if e == 0 and self.eInfo[fErode,e]:
        #             mEFunc.find_simple_point_candidates_edges_0(img,edt,fErode,self.eLoopB[e],simple_border_points)
        #         if e == 1 and self.eInfo[fErode,e]:
        #             mEFunc.find_simple_point_candidates_edges_1(img,edt,fErode,self.eLoopB[e],simple_border_points)
        #         if e == 2 and self.eInfo[fErode,e]:
        #             mEFunc.find_simple_point_candidates_edges_2(img,edt,fErode,self.eLoopB[e],simple_border_points)
        #         if e == 3 and self.eInfo[fErode,e]:
        #             mEFunc.find_simple_point_candidates_edges_3(img,edt,fErode,self.eLoopB[e],simple_border_points)
        #         if e == 4 and self.eInfo[fErode,e]:
        #             mEFunc.find_simple_point_candidates_edges_4(img,edt,fErode,self.eLoopB[e],simple_border_points)
        #         if e == 5 and self.eInfo[fErode,e]:
        #             mEFunc.find_simple_point_candidates_edges_5(img,edt,fErode,self.eLoopB[e],simple_border_points)
        #         if e == 6 and self.eInfo[fErode,e]:
        #             mEFunc.find_simple_point_candidates_edges_6(img,edt,fErode,self.eLoopB[e],simple_border_points)
        #         if e == 7 and self.eInfo[fErode,e]:
        #             mEFunc.find_simple_point_candidates_edges_7(img,edt,fErode,self.eLoopB[e],simple_border_points)
        #         if e == 8 and self.eInfo[fErode,e]:
        #             mEFunc.find_simple_point_candidates_edges_8(img,edt,fErode,self.eLoopB[e],simple_border_points)
        #         if e == 9 and self.eInfo[fErode,e]:
        #             mEFunc.find_simple_point_candidates_edges_9(img,edt,fErode,self.eLoopB[e],simple_border_points)
        #         if e == 10 and self.eInfo[fErode,e]:
        #             mEFunc.find_simple_point_candidates_edges_10(img,edt,fErode,self.eLoopB[e],simple_border_points)
        #         if e == 11 and self.eInfo[fErode,e]:
        #             mEFunc.find_simple_point_candidates_edges_11(img,edt,fErode,self.eLoopB[e],simple_border_points)

        #     for c in self.Orientation.corners:
        #         if c == 0 and self.cInfo[fErode,c]:
        #             mEFunc.find_simple_point_candidates_corners_0(img,edt,fErode,self.cLoopB[c],simple_border_points)
        #         if c == 1 and self.cInfo[fErode,c]:
        #             mEFunc.find_simple_point_candidates_corners_1(img,edt,fErode,self.cLoopB[c],simple_border_points)
        #         if c == 2 and self.cInfo[fErode,c]:
        #             mEFunc.find_simple_point_candidates_corners_2(img,edt,fErode,self.cLoopB[c],simple_border_points)
        #         if c == 3 and self.cInfo[fErode,c]:
        #             mEFunc.find_simple_point_candidates_corners_3(img,edt,fErode,self.cLoopB[c],simple_border_points)
        #         if c == 4 and self.cInfo[fErode,c]:
        #             mEFunc.find_simple_point_candidates_corners_4(img,edt,fErode,self.cLoopB[c],simple_border_points)
        #         if c == 5 and self.cInfo[fErode,c]:
        #             mEFunc.find_simple_point_candidates_corners_5(img,edt,fErode,self.cLoopB[c],simple_border_points)
        #         if c == 6 and self.cInfo[fErode,c]:
        #             mEFunc.find_simple_point_candidates_corners_6(img,edt,fErode,self.cLoopB[c],simple_border_points)
        #         if c == 7 and self.cInfo[fErode,c]:
        #             mEFunc.find_simple_point_candidates_corners_7(img,edt,fErode,self.cLoopB[c],simple_border_points)

        num_border_points = simple_border_points.size()
        
        #simple_border_points = sorted(simple_border_points, key=lambda d: d['faceCount'],reverse=True)
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
