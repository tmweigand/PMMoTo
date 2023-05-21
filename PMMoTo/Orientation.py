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


        self.faces=  {0:{'ID':(-1, 0, 0),'oppIndex':1, 'argOrder':np.array([0,1,2],dtype=np.uint8), 'dir': 1},
                      1:{'ID':( 1, 0, 0),'oppIndex':0, 'argOrder':np.array([0,1,2],dtype=np.uint8), 'dir':-1},
                      2:{'ID':( 0,-1, 0),'oppIndex':3, 'argOrder':np.array([1,0,2],dtype=np.uint8), 'dir': 1},
                      3:{'ID':( 0, 1, 0),'oppIndex':2, 'argOrder':np.array([1,0,2],dtype=np.uint8), 'dir':-1},
                      4:{'ID':( 0, 0,-1),'oppIndex':5, 'argOrder':np.array([2,0,1],dtype=np.uint8), 'dir': 1},
                      5:{'ID':( 0, 0, 1),'oppIndex':4, 'argOrder':np.array([2,0,1],dtype=np.uint8), 'dir':-1}
                      }
        self.edges = {0 :{'ID':(-1, 0,-1), 'oppIndex':5, 'faceIndex':(0,4), 'dir':(0,2)},
                      1 :{'ID':(-1, 0, 1), 'oppIndex':4, 'faceIndex':(0,5), 'dir':(0,2)},
                      2 :{'ID':(-1,-1, 0), 'oppIndex':7, 'faceIndex':(0,2), 'dir':(0,1)},
                      3 :{'ID':(-1, 1, 0), 'oppIndex':6, 'faceIndex':(0,3), 'dir':(0,1)},
                      4 :{'ID':( 1, 0,-1), 'oppIndex':1, 'faceIndex':(1,4), 'dir':(0,2)},
                      5 :{'ID':( 1, 0, 1), 'oppIndex':0, 'faceIndex':(1,5), 'dir':(0,2)},
                      6 :{'ID':( 1,-1, 0), 'oppIndex':3, 'faceIndex':(1,2), 'dir':(0,1)},
                      7 :{'ID':( 1, 1, 0), 'oppIndex':2, 'faceIndex':(1,3), 'dir':(0,1)},
                      8 :{'ID':( 0,-1,-1), 'oppIndex':11,'faceIndex':(2,4), 'dir':(1,2)},
                      9 :{'ID':( 0,-1, 1), 'oppIndex':10,'faceIndex':(2,5), 'dir':(1,2)},
                      10:{'ID':( 0, 1,-1), 'oppIndex':9, 'faceIndex':(3,4), 'dir':(1,2)},
                      11:{'ID':( 0, 1, 1), 'oppIndex':8, 'faceIndex':(3,5), 'dir':(1,2)},
                    }

        self.corners = {0:{'ID':(-1,-1,-1),'oppIndex':7, 'faceIndex':(0,2,4)},
                        1:{'ID':(-1,-1, 1),'oppIndex':6, 'faceIndex':(0,2,5)},
                        2:{'ID':(-1, 1,-1),'oppIndex':5, 'faceIndex':(0,3,4)},
                        3:{'ID':(-1, 1, 1),'oppIndex':4, 'faceIndex':(0,3,5)},
                        4:{'ID':( 1,-1,-1),'oppIndex':3, 'faceIndex':(1,2,4)}, 
                        5:{'ID':( 1,-1, 1),'oppIndex':2, 'faceIndex':(1,2,5)},
                        6:{'ID':( 1, 1,-1),'oppIndex':1, 'faceIndex':(1,3,4)}, 
                        7:{'ID':( 1, 1,1 ),'oppIndex':0, 'faceIndex':(1,3,5)}
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

    def getSendSlices(self,structRatio,buffer):
        """
        Determine slices of face, edge, and corner neighbor to send data 
        structRatio is size of voxel window to send
        buffer is XXX
        """

        #############
        ### Faces ###
        #############
        for fIndex in self.faces:
            fID = self.faces[fIndex]['ID']
            for n in range(len(fID)):
                if fID[n] != 0:
                    if fID[n] > 0:
                        buf = None
                        if buffer[n][1] > 0:
                            buf = -buffer[n][1]*2
                        self.sendFSlices[fIndex,n] = slice(-structRatio[n]-buffer[n][1]*2,buf)
                    else:
                        buf = None
                        if buffer[n][0] > 0:
                            buf = buffer[n][0]*2
                        self.sendFSlices[fIndex,n] = slice(buf,structRatio[n]+buffer[n][0]*2)
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
                        if buffer[n][1] > 0:
                            buf = -buffer[n][1]*2
                        self.sendESlices[eIndex,n] = slice(-structRatio[n]-buffer[n][1]*2,buf)
                    else:
                        buf = None
                        if buffer[n][0] > 0:
                            buf = buffer[n][0]*2
                        self.sendESlices[eIndex,n] = slice(buf,structRatio[n]+buffer[n][0]*2)
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
                    if buffer[n][1] > 0:
                        buf = -buffer[n][1]*2
                    self.sendCSlices[cIndex,n] = slice(-structRatio[n]-buffer[n][1]*2,buf)
                else:
                    buf = None
                    if buffer[n][0] > 0:
                        buf = buffer[n][0]*2
                    self.sendCSlices[cIndex,n] = slice(buf,structRatio[n]+buffer[n][0]*2)
        ###############

    def getRecieveSlices(self,structRatio,pad,arr):
        """
        Determine slices of face, edge, and corner neighbor to recieve data 
        structRatio is 
        pad is XXX
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
                    self.recvFSlices[fIndex,n] = slice(0+pad[n,1],dim[n]-pad[n,0])
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
                    self.recvESlices[eIndex,n] = slice(0+pad[n,1],dim[n]-pad[n,0])
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
