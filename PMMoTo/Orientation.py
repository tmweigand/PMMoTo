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

    def getSendSlices(self,structRatio,buffer,updateBuffer=False):
        """
        Determine slices of face, edge, and corner neighbor to send data 
        structRatio is size of voxel window to send
        buffer is XXX
        """

        ### Determine if function is to update Buffer or add Halo
        factor = 2
        if updateBuffer:
            factor = 1

        #############
        ### Faces ###
        #############
        for fIndex in self.faces:
            fID = self.faces[fIndex]['ID']
            for n in range(len(fID)):
                if fID[n] != 0:
                    if fID[n] > 0:
                        buf = None
                        if buffer[fIndex] > 0:
                            buf = -buffer[fIndex]*factor
                        self.sendFSlices[fIndex,n] = slice(-structRatio[n]-buffer[fIndex]*factor,buf)
                    else:
                        buf = None
                        if buffer[fIndex] > 0:
                            buf = buffer[fIndex]*factor
                        self.sendFSlices[fIndex,n] = slice(buf,structRatio[n]+buffer[fIndex]*factor)
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
                        buf = None
                        if buffer[fIndex] > 0:
                            buf = -buffer[fIndex]*factor
                        self.sendESlices[eIndex,n] = slice(-structRatio[n]-buffer[fIndex]*factor,buf)
                    else:
                        buf = None
                        if buffer[fIndex] > 0:
                            buf = buffer[fIndex]*factor
                        self.sendESlices[eIndex,n] = slice(buf,structRatio[n]+buffer[fIndex]*factor)
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
                    buf = None
                    if buffer[fIndex] > 0:
                        buf = -buffer[fIndex]*factor
                    self.sendCSlices[cIndex,n] = slice(-structRatio[n]-buffer[fIndex]*factor,buf)
                else:
                    buf = None
                    if buffer[fIndex] > 0:
                        buf = buffer[fIndex]*factor
                    self.sendCSlices[cIndex,n] = slice(buf,structRatio[n]+buffer[fIndex]*factor)
        ###############

    def getRecieveSlices(self,structRatio,pad,arr):
        """
        Determine slices of face, edge, and corner neighbor to recieve data 
        structRatio is 
        pad is amout arr has increased 
        arr is 
        """

        dim = arr.shape
        if pad.shape != [3,2]:
            pad = pad.reshape([3,2])

        #############
        ### Faces ###
        #############
        for fIndex in self.faces:
            fID = self.faces[fIndex]['ID']
            for n in range(len(fID)):
                if fID[n] != 0:
                    if fID[n] > 0:
                        self.recvFSlices[fIndex,n] = slice(-structRatio[n],None)
                    else:
                        self.recvFSlices[fIndex,n] = slice(None,structRatio[n])
                else:
                    self.recvFSlices[fIndex,n] = slice(0+pad[n,0],dim[n]-pad[n,1])
        #############

        #############
        ### Edges ###
        #############
        for eIndex in self.edges:
            eID = self.edges[eIndex]['ID']
            for n in range(len(eID)):
                if eID[n] != 0:
                    if eID[n] > 0:
                        self.recvESlices[eIndex,n] = slice(-structRatio[n],None)
                    else:
                        self.recvESlices[eIndex,n] = slice(None,structRatio[n])
                else:
                    self.recvESlices[eIndex,n] = slice(0+pad[n,0],dim[n]-pad[n,1])
        #############

        ###############
        ### Corners ###
        ###############
        for cIndex in self.corners:
            cID = self.corners[cIndex]['ID']
            for n in range(len(cID)):
                if cID[n] > 0:
                    self.recvCSlices[cIndex,n] = slice(-structRatio[n],None)
                else:
                    self.recvCSlices[cIndex,n] = slice(None,structRatio[n])
        ###############
