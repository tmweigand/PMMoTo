import numpy as np

class Orientation(object):
    """
    Orientation of the voxels broken into face, edge, and corner neighbors
    """
    def __init__(self):
        self.numFaces = 6
        self.numEdges = 12
        self.numCorners = 8
        self.numNeighbors = 26

        self.sendFSlices = np.empty([self.numFaces,3],dtype=object)
        self.recvFSlices = np.empty([self.numFaces,3],dtype=object)
        self.sendESlices = np.empty([self.numEdges,3],dtype=object)
        self.recvESlices = np.empty([self.numEdges,3],dtype=object)
        self.sendCSlices = np.empty([self.numCorners,3],dtype=object)
        self.recvCSlices = np.empty([self.numCorners,3],dtype=object)


        self.faces=  {0:{'ID':(-1, 0, 0),'oppIndex':1, 'MAInd':10, 'argOrder':np.array([0,1,2],dtype=np.uint8), 'dir': 1},
                      1:{'ID':( 1, 0, 0),'oppIndex':0, 'MAInd':11, 'argOrder':np.array([0,1,2],dtype=np.uint8), 'dir':-1},
                      2:{'ID':( 0,-1, 0),'oppIndex':3, 'MAInd':12, 'argOrder':np.array([1,0,2],dtype=np.uint8), 'dir': 1},
                      3:{'ID':( 0, 1, 0),'oppIndex':2, 'MAInd':13, 'argOrder':np.array([1,0,2],dtype=np.uint8), 'dir':-1},
                      4:{'ID':( 0, 0,-1),'oppIndex':5, 'MAInd':14, 'argOrder':np.array([2,0,1],dtype=np.uint8), 'dir': 1},
                      5:{'ID':( 0, 0, 1),'oppIndex':4, 'MAInd':15, 'argOrder':np.array([2,0,1],dtype=np.uint8), 'dir':-1}
                      }
        
        self.edges = {0 :{'ID':(-1, 0,-1), 'oppIndex':5, 'MAInd':20, 'faceIndex':(0,4), 'dir':(0,2)},
                      1 :{'ID':(-1, 0, 1), 'oppIndex':4, 'MAInd':21, 'faceIndex':(0,5), 'dir':(0,2)},
                      2 :{'ID':(-1,-1, 0), 'oppIndex':7, 'MAInd':22, 'faceIndex':(0,2), 'dir':(0,1)},
                      3 :{'ID':(-1, 1, 0), 'oppIndex':6, 'MAInd':23, 'faceIndex':(0,3), 'dir':(0,1)},
                      4 :{'ID':( 1, 0,-1), 'oppIndex':1, 'MAInd':24, 'faceIndex':(1,4), 'dir':(0,2)},
                      5 :{'ID':( 1, 0, 1), 'oppIndex':0, 'MAInd':25, 'faceIndex':(1,5), 'dir':(0,2)},
                      6 :{'ID':( 1,-1, 0), 'oppIndex':3, 'MAInd':26, 'faceIndex':(1,2), 'dir':(0,1)},
                      7 :{'ID':( 1, 1, 0), 'oppIndex':2, 'MAInd':27, 'faceIndex':(1,3), 'dir':(0,1)},
                      8 :{'ID':( 0,-1,-1), 'oppIndex':11,'MAInd':28, 'faceIndex':(2,4), 'dir':(1,2)},
                      9 :{'ID':( 0,-1, 1), 'oppIndex':10,'MAInd':29, 'faceIndex':(2,5), 'dir':(1,2)},
                      10:{'ID':( 0, 1,-1), 'oppIndex':9, 'MAInd':30, 'faceIndex':(3,4), 'dir':(1,2)},
                      11:{'ID':( 0, 1, 1), 'oppIndex':8, 'MAInd':31, 'faceIndex':(3,5), 'dir':(1,2)},
                    }

        self.corners = {0:{'ID':(-1,-1,-1),'oppIndex':7, 'MAInd':47, 'faceIndex':(0,2,4)},
                        1:{'ID':(-1,-1, 1),'oppIndex':6, 'MAInd':45, 'faceIndex':(0,2,5)},
                        2:{'ID':(-1, 1,-1),'oppIndex':5, 'MAInd':46, 'faceIndex':(0,3,4)},
                        3:{'ID':(-1, 1, 1),'oppIndex':4, 'MAInd':44, 'faceIndex':(0,3,5)},
                        4:{'ID':( 1,-1,-1),'oppIndex':3, 'MAInd':43, 'faceIndex':(1,2,4)}, 
                        5:{'ID':( 1,-1, 1),'oppIndex':2, 'MAInd':41, 'faceIndex':(1,2,5)},
                        6:{'ID':( 1, 1,-1),'oppIndex':1, 'MAInd':42, 'faceIndex':(1,3,4)}, 
                        7:{'ID':( 1, 1, 1),'oppIndex':0, 'MAInd':40, 'faceIndex':(1,3,5)}
                        }
        

        self.directions ={0 :{'ID':[-1,-1,-1],'index': 0 ,'oppIndex': 25},
                          1 :{'ID':[-1,-1,0], 'index': 1 ,'oppIndex': 24},
                          2 :{'ID':[-1,-1,1], 'index': 2 ,'oppIndex': 23},
                          3 :{'ID':[-1,0,-1], 'index': 3 ,'oppIndex': 22},
                          4 :{'ID':[-1,0,0],  'index': 4 ,'oppIndex': 21},
                          5 :{'ID':[-1,0,1],  'index': 5 ,'oppIndex': 20},
                          6 :{'ID':[-1,1,-1], 'index': 6 ,'oppIndex': 19},
                          7 :{'ID':[-1,1,0],  'index': 7 ,'oppIndex': 18},
                          8 :{'ID':[-1,1,1],  'index': 8 ,'oppIndex': 17},
                          9 :{'ID':[0,-1,-1], 'index': 9 ,'oppIndex': 16},
                          10:{'ID':[0,-1,0],  'index': 10 ,'oppIndex': 15},
                          11:{'ID':[0,-1,1],  'index': 11 ,'oppIndex': 14},
                          12:{'ID':[0,0,-1],  'index': 12 ,'oppIndex': 13},
                          13:{'ID':[0,0,1],   'index': 13 ,'oppIndex': 12},
                          14:{'ID':[0,1,-1],  'index': 14 ,'oppIndex': 11},
                          15:{'ID':[0,1,0],   'index': 15 ,'oppIndex': 10},
                          16:{'ID':[0,1,1],   'index': 16 ,'oppIndex': 9},
                          17:{'ID':[1,-1,-1], 'index': 17 ,'oppIndex': 8},
                          18:{'ID':[1,-1,0],  'index': 18 ,'oppIndex': 7},
                          19:{'ID':[1,-1,1],  'index': 19 ,'oppIndex': 6},
                          20:{'ID':[1,0,-1],  'index': 20 ,'oppIndex': 5},
                          21:{'ID':[1,0,0],   'index': 21 ,'oppIndex': 4},
                          22:{'ID':[1,0,1],   'index': 22 ,'oppIndex': 3},
                          23:{'ID':[1,1,-1],  'index': 23 ,'oppIndex': 2},
                          24:{'ID':[1,1,0],   'index': 24 ,'oppIndex': 1},
                          25:{'ID':[1,1,1],   'index': 25 ,'oppIndex': 0},
                         }

    def getLoopInfo(self,grid,subDomain,inlet,outlet,resPad):
        """
        Grap  Loop Information to Cycle through the Boundary Faces and Internal Nodes
        Reservois are Treated as Entire Face  
        """

        loopInfo = np.zeros([self.numFaces+1,3,2],dtype = np.int64)

        rangeInfo = 2*np.ones([6],dtype=np.uint8)
        for fIndex in self.faces:
            if subDomain.boundaryID[fIndex] == 0:
                rangeInfo[fIndex] = rangeInfo[fIndex] - 1
            if inlet[fIndex] > 0:
                rangeInfo[fIndex] = rangeInfo[fIndex] + resPad
            if outlet[fIndex] > 0:
                rangeInfo[fIndex] = rangeInfo[fIndex] + resPad

        for fIndex in self.faces:
            face = self.faces[fIndex]['argOrder'][0]

            if self.faces[fIndex]['dir'] == -1:
                if face == 0:
                    loopInfo[fIndex,0] = [grid.shape[0]-rangeInfo[1],grid.shape[0]]
                    loopInfo[fIndex,1] = [0,grid.shape[1]]
                    loopInfo[fIndex,2] = [0,grid.shape[2]]
                elif face == 1:
                    loopInfo[fIndex,0] = [rangeInfo[0],grid.shape[0]-rangeInfo[1]]
                    loopInfo[fIndex,1] = [grid.shape[1]-rangeInfo[3],grid.shape[1]]
                    loopInfo[fIndex,2] = [0,grid.shape[2]]
                elif face == 2:
                    loopInfo[fIndex,0] = [rangeInfo[0],grid.shape[0]-rangeInfo[1]]
                    loopInfo[fIndex,1] = [rangeInfo[2],grid.shape[1]-rangeInfo[3]]
                    loopInfo[fIndex,2] = [grid.shape[2]-rangeInfo[5],grid.shape[2]]

            elif self.faces[fIndex]['dir'] == 1:
                if face == 0:
                    loopInfo[fIndex,0] = [0,rangeInfo[0]]
                    loopInfo[fIndex,1] = [0,grid.shape[1]]
                    loopInfo[fIndex,2] = [0,grid.shape[2]]
                elif face == 1:
                    loopInfo[fIndex,0] = [rangeInfo[0],grid.shape[0]-rangeInfo[1]]
                    loopInfo[fIndex,1] = [0,rangeInfo[2]]
                    loopInfo[fIndex,2] = [0,grid.shape[2]]
                elif face == 2:
                    loopInfo[fIndex,0] = [rangeInfo[0],grid.shape[0]-rangeInfo[1]]
                    loopInfo[fIndex,1] = [rangeInfo[2],grid.shape[1]-rangeInfo[3]]
                    loopInfo[fIndex,2] = [0,rangeInfo[4]]

        loopInfo[self.numFaces][0] = [rangeInfo[0],grid.shape[0]-rangeInfo[1]]
        loopInfo[self.numFaces][1] = [rangeInfo[2],grid.shape[1]-rangeInfo[3]]
        loopInfo[self.numFaces][2] = [rangeInfo[4],grid.shape[2]-rangeInfo[5]]

        return loopInfo

    def getSendSlices(self,structRatio,buffer,dim):
        """
        Determine slices of face, edge, and corner neighbor to send data 
        structRatio is size of voxel window to send and is [nx,ny,nz]
        buffer is the subDomain.buffer
        dim is grid.shape
        Buffer is always updated on edges and corners due to geometry contraints
        """

        #############
        ### Faces ###
        #############
        for fIndex in self.faces:
            fID = self.faces[fIndex]['ID']
            for n in range(len(fID)):
                if fID[n] != 0:
                    if fID[n] > 0:
                        self.sendFSlices[fIndex,n] = slice(dim[n]-structRatio[n*2+1]-buffer[n*2+1]-1,dim[n]-buffer[n*2+1]-1)
                    else:
                        self.sendFSlices[fIndex,n] = slice(buffer[n*2]+1,buffer[n*2]+structRatio[n*2]+1)
                else:
                    self.sendFSlices[fIndex,n] = slice(None,None)
        #############

        #############
        ### Edges ###
        #############
        for eIndex in self.edges:
            eID = self.edges[eIndex]['ID']
            for n in range(len(eID)):
                if eID[n] != 0:
                    if eID[n] > 0:
                        self.sendESlices[eIndex,n] = slice(dim[n]-structRatio[n*2+1]-buffer[n*2+1]-1,dim[n]-1)
                    else:
                        self.sendESlices[eIndex,n] = slice(buffer[n*2],buffer[n*2]+structRatio[n*2]+1)
                else:
                    self.sendESlices[eIndex,n] = slice(None,None)
        #############

        ###############
        ### Corners ###
        ###############
        for cIndex in self.corners:
            cID = self.corners[cIndex]['ID']
            for n in range(len(cID)):
                if cID[n] > 0:
                    self.sendCSlices[cIndex,n] = slice(dim[n]-structRatio[n*2+1]-buffer[n*2+1]-1,dim[n]-1)
                else:
                    self.sendCSlices[cIndex,n] = slice(buffer[n*2],buffer[n*2]+structRatio[n*2]+1)
        ###############



    def getSendBufferSlices(self,buffer,dim):
        """
        Determine slices of face, edge, and corner neighbor to send data 
        structRatio is size of voxel window to send and is [nx,ny,nz]
        buffer is the subDomain.buffer
        dim is grid.shape
        Buffer is always updated on edges and corners due to geometry contraints
        """

        #############
        ### Faces ###
        #############
        for fIndex in self.faces:
            fID = self.faces[fIndex]['ID']
            for n in range(len(fID)):
                if fID[n] != 0:
                    if fID[n] > 0:
                        self.sendFSlices[fIndex,n] = slice(dim[n]-2*buffer[n*2+1],dim[n]-buffer[n*2+1])
                    else:
                        self.sendFSlices[fIndex,n] = slice(buffer[n*2],2*buffer[n*2])
                else:
                    self.sendFSlices[fIndex,n] = slice(buffer[n*2],dim[n]-buffer[n*2+1])
        #############

        #############
        ### Edges ###
        #############
        for eIndex in self.edges:
            eID = self.edges[eIndex]['ID']
            for n in range(len(eID)):
                if eID[n] != 0:
                    if eID[n] > 0:
                        self.sendESlices[eIndex,n] = slice(dim[n]-2*buffer[n*2+1],dim[n]-buffer[n*2+1])
                    else:
                        self.sendESlices[eIndex,n] = slice(buffer[n*2],2*buffer[n*2])
                else:
                    self.sendESlices[eIndex,n] = slice(buffer[n*2],dim[n]-buffer[n*2+1])
        #############

        ###############
        ### Corners ###
        ###############
        for cIndex in self.corners:
            cID = self.corners[cIndex]['ID']
            for n in range(len(cID)):
                if cID[n] > 0:
                    self.sendCSlices[cIndex,n] = slice(dim[n]-2*buffer[n*2+1],dim[n]-buffer[n*2+1])
                else:
                    self.sendCSlices[cIndex,n] = slice(buffer[n*2],2*buffer[n*2])
        ###############


    def getRecieveSlices(self,halo,buffer,dim):
        """
        Determine slices of face, edge, and corner neighbor to recieve data 
        Buffer is always updated on edges and corners due to geometry contraints
        """

        #############
        ### Faces ###
        #############
        for fIndex in self.faces:
            fID = self.faces[fIndex]['ID']
            for n in range(len(fID)):
                if fID[n] != 0:
                    if fID[n] > 0:
                        self.recvFSlices[fIndex,n] = slice(dim[n]-halo[n*2+1],dim[n])
                    else:
                        self.recvFSlices[fIndex,n] = slice(None,halo[n*2])
                else:
                    self.recvFSlices[fIndex,n] = slice(halo[n*2],dim[n]-halo[n*2+1])
        #############

        #############
        ### Edges ###
        #############
        for eIndex in self.edges:
            eID = self.edges[eIndex]['ID']
            for n in range(len(eID)):
                if eID[n] != 0:
                    if eID[n] > 0:
                        self.recvESlices[eIndex,n] = slice(dim[n]-halo[n*2+1]-buffer[n*2+1],dim[n])
                    else:
                        self.recvESlices[eIndex,n] = slice(None,halo[n*2]+buffer[n*2])
                else:
                    self.recvESlices[eIndex,n] = slice(halo[n*2],dim[n]-halo[n*2+1])
        #############

        ###############
        ### Corners ###
        ###############
        for cIndex in self.corners:
            cID = self.corners[cIndex]['ID']
            for n in range(len(cID)):
                if cID[n] > 0:
                    self.recvCSlices[cIndex,n] = slice(dim[n]-halo[n*2+1]-buffer[n*2+1],dim[n])
                else:
                    self.recvCSlices[cIndex,n] = slice(None,halo[n*2]+buffer[n*2])
        ###############

    def getRecieveBufferSlices(self,buffer,dim):
        """
        Determine slices of face, edge, and corner neighbor to recieve data 
        Buffer is always updated on edges and corners due to geometry contraints
        """

        #############
        ### Faces ###
        #############
        for fIndex in self.faces:
            fID = self.faces[fIndex]['ID']
            for n in range(len(fID)):
                if fID[n] != 0:
                    if fID[n] > 0:
                        self.recvFSlices[fIndex,n] = slice(dim[n]-buffer[n*2+1],dim[n])
                    else:
                        self.recvFSlices[fIndex,n] = slice(None,buffer[n*2])
                else:
                    self.recvFSlices[fIndex,n] = slice(buffer[n*2],dim[n]-buffer[n*2+1])
        #############

        #############
        ### Edges ###
        #############
        for eIndex in self.edges:
            eID = self.edges[eIndex]['ID']
            for n in range(len(eID)):
                if eID[n] != 0:
                    if eID[n] > 0:
                        self.recvESlices[eIndex,n] = slice(dim[n]-buffer[n*2+1],dim[n])
                    else:
                        self.recvESlices[eIndex,n] = slice(None,buffer[n*2])
                else:
                    self.recvESlices[eIndex,n] = slice(buffer[n*2],dim[n]-buffer[n*2+1])
        #############

        ###############
        ### Corners ###
        ###############
        for cIndex in self.corners:
            cID = self.corners[cIndex]['ID']
            for n in range(len(cID)):
                if cID[n] > 0:
                    self.recvCSlices[cIndex,n] = slice(dim[n]-buffer[n*2+1],dim[n])
                else:
                    self.recvCSlices[cIndex,n] = slice(None,buffer[n*2])
        ###############



    def getMALoopInfo(self,buffer,grid):
        """
        Determine slices of face, edge, and corner internal boundary slices 
        """

        FLoopI = np.empty([self.numFaces,3,2],dtype=np.int64)
        FLoopB = np.empty([self.numFaces,3,2],dtype=np.int64)
        ELoopI = np.empty([self.numFaces,self.numEdges,3,2],dtype=np.int64)
        ELoopB = np.empty([self.numEdges,3,2],dtype=np.int64)
        CLoopI = np.empty([self.numFaces,self.numCorners,3,2],dtype=np.int64)
        CLoopB = np.empty([self.numCorners,3,2],dtype=np.int64)

        dim = grid.shape

        #############
        ### Faces ###
        #############
        for fIndex in self.faces:
            fID = self.faces[fIndex]['ID']
            for n in range(len(fID)):
                if fID[n] != 0:
                    if fID[n] > 0:
                        FLoopB[fIndex,n] = [dim[n]-buffer[n*2+1]-1,dim[n]-buffer[n*2+1]]
                        FLoopI[fIndex,n] = [buffer[n*2+1]+1,dim[n]-buffer[n*2+1]-1]
                    else:
                        FLoopB[fIndex,n] = [buffer[n*2],buffer[n*2]+1]
                        FLoopI[fIndex,n] = [buffer[n*2]+1,dim[n]-buffer[n*2]-1]
                else:
                    FLoopB[fIndex,n] = [buffer[n*2]+1,dim[n]-buffer[n*2+1]-1]
                    FLoopI[fIndex,n] = [buffer[n*2]+1,dim[n]-buffer[n*2+1]-1]

        #############

        ######################
        ### Boundary Edges ###
        ######################
        for eIndex in self.edges:
            eID = self.edges[eIndex]['ID']
            for n in range(len(eID)):
                if eID[n] != 0:
                    if eID[n] > 0:
                        bufID = n*2+1
                        ELoopB[eIndex,n] = [dim[n]-buffer[bufID]-1,dim[n]-buffer[bufID]]
                    else:
                        bufID = n*2
                        ELoopB[eIndex,n] = [buffer[bufID],buffer[bufID]+1]
                else:
                    ELoopB[eIndex,n] = [buffer[n*2]+1,dim[n]-buffer[n*2+1]-1]
        ######################

        ######################
        ### Internal Edges ###
        ######################
        for fIndex in self.faces:
            erodeAxis = self.faces[fIndex]['argOrder']
            for eIndex in self.edges:
                eID = self.edges[eIndex]['ID']
                for n in range(len(eID)):
                    if n == erodeAxis[0]:
                        if eID[n] != 0:
                            if eID[n] > 0:
                                bufID = n*2+1
                                ELoopI[fIndex,eIndex,n] = [buffer[bufID],dim[n]-buffer[bufID]-2]
                            else:
                                bufID = n*2
                                ELoopI[fIndex,eIndex,n] = [buffer[bufID]+1,dim[n]-buffer[bufID]-1] 
                    else:                       
                        if eID[n] != 0:
                            if eID[n] > 0:
                                bufID = n*2+1
                                ELoopI[fIndex,eIndex,n] = [dim[n]-buffer[bufID]-1,dim[n]-buffer[bufID]]
                            else:
                                bufID = n*2
                                ELoopI[fIndex,eIndex,n] = [buffer[bufID],buffer[bufID]+1]
                        else:
                            ELoopI[fIndex,eIndex,n] = [buffer[n*2]+1,dim[n]-buffer[n*2+1]-1]
        ######################


        ########################
        ### Boundary Corners ###
        ########################
        for cIndex in self.corners:
            cID = self.corners[cIndex]['ID']
            for n in range(len(cID)):
                if cID[n] > 0:
                    bufID = n*2+1
                    CLoopB[cIndex,n] = [dim[n]-buffer[bufID]-1,dim[n]-buffer[bufID]]
                else:
                    bufID = n*2
                    CLoopB[cIndex,n] = [buffer[bufID],buffer[bufID]+1]
        ########################

        ########################
        ### Internal Corners ###
        ########################
        for fIndex in self.faces:
            erodeAxis = self.faces[fIndex]['argOrder']
            for cIndex in self.corners:
                cID = self.corners[cIndex]['ID']
                for n in range(len(cID)):
                    if n == erodeAxis[0]:
                        if cID[n] > 0:
                            bufID = n*2+1
                            CLoopI[fIndex,cIndex,n] = [buffer[bufID],dim[n]-buffer[bufID]-2]
                        else:
                            bufID = n*2
                            CLoopI[fIndex,cIndex,n] = [buffer[bufID]+1,dim[n]-buffer[bufID]-1]
                    else:
                        if cID[n] > 0:
                            bufID = n*2+1
                            CLoopI[fIndex,cIndex,n] = [dim[n]-buffer[bufID]-1,dim[n]-buffer[bufID]]
                        else:
                            bufID = n*2
                            CLoopI[fIndex,cIndex,n] = [buffer[bufID],1+buffer[bufID]]
                
        ########################

        return FLoopB,FLoopI,ELoopB,ELoopI,CLoopB,CLoopI

    def getMALoopInfoALL(self,grid):
        """
        Determine slices of face, edge, and corner internal boundary slices 
        """

        FLoopI = np.empty([self.numFaces,3,2],dtype=np.int64)
        FLoopB = np.empty([self.numFaces,3,2],dtype=np.int64)
        ELoopI = np.empty([self.numFaces,self.numEdges,3,2],dtype=np.int64)
        ELoopB = np.empty([self.numEdges,3,2],dtype=np.int64)
        CLoopI = np.empty([self.numFaces,self.numCorners,3,2],dtype=np.int64)
        CLoopB = np.empty([self.numCorners,3,2],dtype=np.int64)

        dim = grid.shape

        #############
        ### Faces ###
        #############
        for fIndex in self.faces:
            fID = self.faces[fIndex]['ID']
            for n in range(len(fID)):
                if fID[n] != 0:
                    if fID[n] > 0:
                        FLoopB[fIndex,n] = [dim[n]-1,dim[n]]
                        FLoopI[fIndex,n] = [1,dim[n]-1]
                    else:
                        FLoopB[fIndex,n] = [0,1]
                        FLoopI[fIndex,n] = [1,dim[n]-1]
                else:
                    FLoopB[fIndex,n] = [1,dim[n]-1]
                    FLoopI[fIndex,n] = [1,dim[n]-1]

        #############

        ######################
        ### Boundary Edges ###
        ######################
        for eIndex in self.edges:
            eID = self.edges[eIndex]['ID']
            for n in range(len(eID)):
                if eID[n] != 0:
                    if eID[n] > 0:
                        ELoopB[eIndex,n] = [dim[n]-1,dim[n]]
                    else:
                        ELoopB[eIndex,n] = [0,1]
                else:
                    ELoopB[eIndex,n] = [1,dim[n]-1]
        ######################

        ######################
        ### Internal Edges ###
        ######################
        for fIndex in self.faces:
            erodeAxis = self.faces[fIndex]['argOrder']
            for eIndex in self.edges:
                eID = self.edges[eIndex]['ID']
                for n in range(len(eID)):
                    if n == erodeAxis[0]:
                        if eID[n] != 0:
                            if eID[n] > 0:
                                ELoopI[fIndex,eIndex,n] = [1,dim[n]-2]
                            else:
                                ELoopI[fIndex,eIndex,n] = [1,dim[n]-2] 
                    else:                       
                        if eID[n] != 0:
                            if eID[n] > 0:
                                ELoopI[fIndex,eIndex,n] = [dim[n]-2,dim[n]-1]
                            else:
                                ELoopI[fIndex,eIndex,n] = [1,2]
                        else:
                            ELoopI[fIndex,eIndex,n] = [1,dim[n]-2]
        ######################


        ########################
        ### Boundary Corners ###
        ########################
        for cIndex in self.corners:
            cID = self.corners[cIndex]['ID']
            for n in range(len(cID)):
                if cID[n] > 0:
                    CLoopB[cIndex,n] = [dim[n]-1,dim[n]]
                else:
                    CLoopB[cIndex,n] = [0,1]
        ########################

        ########################
        ### Internal Corners ###
        ########################
        for fIndex in self.faces:
            erodeAxis = self.faces[fIndex]['argOrder']
            for cIndex in self.corners:
                cID = self.corners[cIndex]['ID']
                for n in range(len(cID)):
                    if n == erodeAxis[0]:
                        if cID[n] > 0:
                            CLoopI[fIndex,cIndex,n] = [1,dim[n]-2]
                        else:
                            CLoopI[fIndex,cIndex,n] = [1,dim[n]-2]
                    else:
                        if cID[n] > 0:
                            CLoopI[fIndex,cIndex,n] = [dim[n]-2,dim[n]-1]
                        else:
                            CLoopI[fIndex,cIndex,n] = [1,2]
                
        ########################

        return FLoopB,FLoopI,ELoopB,ELoopI,CLoopB,CLoopI
    





    def getMALoopInfoTEST(self,buffer,padding,grid):
        """
        Determine slices of face, edge, and corner internal boundary slices 
        """

        FLoopI = np.zeros([3,2],dtype=np.int64)
        FLoopB = np.zeros([self.numFaces,3,2],dtype=np.int64)
        ELoopI = np.zeros([self.numEdges,3,2],dtype=np.int64)
        ELoopB = np.zeros([self.numEdges,3,2],dtype=np.int64)
        CLoopI = np.zeros([self.numCorners,3,2],dtype=np.int64)
        CLoopB = np.zeros([self.numCorners,3,2],dtype=np.int64)

        dim = grid.shape

        #############
        ### Faces ###
        #############
        for fIndex in self.faces:
            fID = self.faces[fIndex]['ID']
            for n in range(len(fID)):
                if fID[n] != 0:
                    if fID[n] > 0:
                        FLoopB[fIndex,n] = [dim[n]-buffer[fIndex]-padding[fIndex]-1,dim[n]-buffer[fIndex]-1]
                    else:
                        FLoopB[fIndex,n] = [buffer[fIndex]+1,buffer[fIndex]+padding[fIndex]+1]
                else:
                    FLoopB[fIndex,n] = [buffer[n*2]+padding[n*2]+2,dim[n]-buffer[n*2+1]-padding[n*2+1]-2]
                FLoopI[n] = [buffer[n*2]+padding[n*2]+1,dim[n]-buffer[n*2+1]-padding[n*2+1]-1]

        #############

        ######################
        ### Boundary Edges ###
        ######################
        for eIndex in self.edges:
            eID = self.edges[eIndex]['ID']
            for n in range(len(eID)):
                if eID[n] != 0:
                    if eID[n] > 0:
                        ELoopB[eIndex,n] = [dim[n]-buffer[n*2+1]-2*padding[n*2+1],dim[n]-buffer[n*2+1]]
                    else:
                        ELoopB[eIndex,n] = [buffer[n*2],buffer[n*2]+2*padding[n*2]] 
                else:
                    ELoopB[eIndex,n] = [buffer[n*2]+padding[n*2]+2,dim[n]-buffer[n*2+1]-padding[n*2+1]-2]
        ######################

        ######################
        ### Internal Edges ###
        ######################
        for eIndex in self.edges:
            eID = self.edges[eIndex]['ID']
            for n in range(len(eID)):
                    if eID[n] != 0:
                        if eID[n] > 0:
                            ELoopI[eIndex,n] = [dim[n]-buffer[n*2+1]-2*padding[n*2+1],dim[n]-buffer[n*2+1]]
                        else:
                            ELoopI[eIndex,n] = [buffer[n*2],buffer[n*2]+2*padding[n*2]] 
                    else:
                        ELoopI[eIndex,n] = [buffer[n*2]+padding[n*2]+2,dim[n]-buffer[n*2+1]-padding[n*2+1]-2]
        ######################


        ########################
        ### Boundary Corners ###
        ########################
        for cIndex in self.corners:
            cID = self.corners[cIndex]['ID']
            for n in range(len(cID)):
                if cID[n] > 0:
                    CLoopB[cIndex,n] = [dim[n]-buffer[n*2+1]-2*padding[n*2+1],dim[n]-buffer[n*2+1]]
                else:
                    CLoopB[cIndex,n] = [buffer[n*2],buffer[n*2]+2*padding[n*2]]
        ########################

        ########################
        ### Internal Corners ###
        ########################
        for cIndex in self.corners:
            cID = self.corners[cIndex]['ID']
            for n in range(len(cID)):
                    if cID[n] > 0:
                        CLoopI[cIndex,n] = [dim[n]-buffer[n*2+1]-2*padding[n*2+1],dim[n]-buffer[n*2+1]]
                    else:
                        CLoopI[cIndex,n] = [buffer[n*2],buffer[n*2]+2*padding[n*2]]
        ########################

        return FLoopB,FLoopI,ELoopB,ELoopI,CLoopB,CLoopI