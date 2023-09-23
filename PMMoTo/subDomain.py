import numpy as np
from mpi4py import MPI
comm = MPI.COMM_WORLD

from .domainGeneration import domainGenINK
from .domainGeneration import domainGen
from .domainGeneration import domainGenVerlet
from . import communication
from . import Orientation
from . import Domain
from . import porousMedia

""" Solid = 0, Pore = 1 """

""" TO DO:
           Switch to pass periodic info and not generate from samples??
           Redo Domain decomposition - Maybe
"""

class subDomain(object):
    def __init__(self,ID,subDomains,Domain,Orientation):
        self.ID          = ID
        self.size        = np.prod(subDomains)
        self.subDomains  = subDomains
        self.Domain      = Domain
        self.Orientation = Orientation
        self.boundary    = False
        self.boundaryID  = -np.ones([6],dtype = np.int8)
        self.buffer      = np.ones([6],dtype = np.int8)
        self.nodes       = np.zeros([3],dtype = np.int64)
        self.ownNodes    = np.zeros([3],dtype = np.int64)
        self.ownNodesIndex = np.zeros([6],dtype = np.int64)
        self.indexStart  = np.zeros([3],dtype = np.int64)
        self.subDomainSize = np.zeros([3])
        self.subID  = np.zeros([3],dtype = np.int64)
        self.lookUpID = np.zeros(subDomains,dtype=np.int64)
        self.neighborF  = -np.ones(self.Orientation.numFaces,dtype = np.int64)
        self.neighborE = -np.ones(self.Orientation.numEdges,dtype = np.int64)
        self.neighborC = -np.ones(self.Orientation.numCorners,dtype = np.int64)
        self.externalE = -np.ones(self.Orientation.numEdges,dtype = np.int64)
        self.externalC = -np.ones(self.Orientation.numCorners,dtype = np.int64)
        self.neighborPerF =  np.zeros([self.Orientation.numFaces,3],dtype = np.int64)
        self.neighborPerE =  np.zeros([self.Orientation.numEdges,3],dtype = np.int64)
        self.neighborPerC =  np.zeros([self.Orientation.numCorners,3],dtype = np.int64)

    def getInfo(self):
        """
        Gather information for each subDomain including:
        ID, boundary information,number of nodes, global index start, buffer
        """
        n = 0
        for i in range(0,self.subDomains[0]):
            for j in range(0,self.subDomains[1]):
                for k in range(0,self.subDomains[2]):
                    self.lookUpID[i,j,k] = n
                    if n == self.ID:
                        if (i == 0):
                            self.boundary = True
                            self.boundaryID[0] = self.Domain.boundaries[0][0]
                        if (i == self.subDomains[0]-1):
                            self.boundary = True
                            self.boundaryID[1] = self.Domain.boundaries[0][1]
                        if (j == 0):
                            self.boundary = True
                            self.boundaryID[2] = self.Domain.boundaries[1][0]
                        if (j == self.subDomains[1]-1):
                            self.boundary = True
                            self.boundaryID[3] = self.Domain.boundaries[1][1]
                        if (k == 0):
                            self.boundary = True
                            self.boundaryID[4] = self.Domain.boundaries[2][0]
                        if (k == self.subDomains[2]-1):
                            self.boundary = True
                            self.boundaryID[5] = self.Domain.boundaries[2][1]

                        self.subID[0] = i
                        self.subID[1] = j
                        self.subID[2] = k
                        self.nodes[0] = self.Domain.subNodes[0]
                        self.nodes[1] = self.Domain.subNodes[1]
                        self.nodes[2] = self.Domain.subNodes[2]
                        self.ownNodes[0] = self.Domain.subNodes[0]
                        self.ownNodes[1] = self.Domain.subNodes[1]
                        self.ownNodes[2] = self.Domain.subNodes[2]
                        self.indexStart[0] = i * self.Domain.subNodes[0]
                        self.indexStart[1] = j * self.Domain.subNodes[1]
                        self.indexStart[2] = k * self.Domain.subNodes[2]
                        if (i == self.subDomains[0]-1):
                            self.nodes[0] += self.Domain.subNodesRem[0]
                            self.ownNodes[0] += self.Domain.subNodesRem[0]
                        if (j == self.subDomains[1]-1):
                            self.nodes[1] += self.Domain.subNodesRem[1]
                            self.ownNodes[1] += self.Domain.subNodesRem[1]
                        if (k == self.subDomains[2]-1):
                            self.nodes[2] += self.Domain.subNodesRem[2]
                            self.ownNodes[2] += self.Domain.subNodesRem[2]
                    n = n + 1

        ### If boundaryID == 0, buffer is not added
        for f in range(0,self.Orientation.numFaces):
            if self.boundaryID[f] == 0:
                self.buffer[f] = 0

    def getXYZ(self,pad):
        """
        Determine actual coordinate information (x,y,z)
        If boundaryID and Domain.boundary == 0, buffer is not added
        Everywhere else a buffer is added
        """

        self.x = np.zeros([self.nodes[0] + pad[0] + pad[1]],dtype=np.double)
        self.y = np.zeros([self.nodes[1] + pad[2] + pad[3]],dtype=np.double)
        self.z = np.zeros([self.nodes[2] + pad[4] + pad[5]],dtype=np.double)

        for c,i in enumerate(range(-pad[0], self.nodes[0] + pad[1])):
            self.x[c] = self.Domain.domainSize[0,0] + (self.indexStart[0] + i)*self.Domain.dX + self.Domain.dX/2

        for c,j in enumerate(range(-pad[2], self.nodes[1] + pad[3])):
            self.y[c] = self.Domain.domainSize[1,0] + (self.indexStart[1] + j)*self.Domain.dY + self.Domain.dY/2

        for c,k in enumerate(range(-pad[4], self.nodes[2] + pad[5])):
            self.z[c] = self.Domain.domainSize[2,0] + (self.indexStart[2] + k)*self.Domain.dZ + self.Domain.dZ/2

        self.ownNodesIndex[0] = pad[0] + self.ownNodesIndex[0]
        self.ownNodesIndex[1] = self.ownNodesIndex[0] +self.ownNodes[0]
        self.ownNodesIndex[2] = pad[2] + self.ownNodesIndex[2]
        self.ownNodesIndex[3] = self.ownNodesIndex[2] + self.ownNodes[1]
        self.ownNodesIndex[4] = pad[4] + self.ownNodesIndex[4]
        self.ownNodesIndex[5] = self.ownNodesIndex[4] + self.ownNodes[2]

        self.nodes[0] = self.nodes[0] + pad[0] + pad[1]
        self.nodes[1] = self.nodes[1] + pad[2] + pad[3]
        self.nodes[2] = self.nodes[2] + pad[4] + pad[5]

        self.indexStart[0] = self.indexStart[0] - pad[0]
        self.indexStart[1] = self.indexStart[1] - pad[2]
        self.indexStart[2] = self.indexStart[2] - pad[4]

        self.subDomainSize = [self.x[-1] - self.x[0],
                              self.y[-1] - self.y[0],
                              self.z[-1] - self.z[0]]


    def getNeighbors(self):
        """
        Get the Face, Edge, and Corner Neighbors for Each Domain
        -4 Represents No Assumed Boundary Condition External Edge but subDomain Corner
        -3 Reperensets No Assumed Boundary Condition External Face but subDomain Edge
        -2 Represents No Assumed Boundary Condition Face / Internal Edge / Internal Corner
        -1 Assumed Wall Boundary Condition
        >=0 is neighbor Proc ID
        """
        lookIDPad = np.pad(self.lookUpID, ( (1, 1), (1, 1), (1, 1)), 'constant', constant_values=-1)
        lookPerI = np.zeros_like(lookIDPad)
        lookPerJ = np.zeros_like(lookIDPad)
        lookPerK = np.zeros_like(lookIDPad)

        if (self.boundaryID[0] == 0):
            lookIDPad[0,:,:] = -2
        if (self.boundaryID[1] == 0):
            lookIDPad[-1,:,:] = -2
        if (self.boundaryID[2] == 0):
            lookIDPad[:,0,:] = -2
        if (self.boundaryID[3] == 0):
            lookIDPad[:,-1,:] = -2
        if (self.boundaryID[4] == 0):
            lookIDPad[:,:,0] = -2
        if (self.boundaryID[5] == 0):
            lookIDPad[:,:,-1] = -2

        if (self.Domain.boundaries[0][0] == 2):
            lookIDPad[0,:,:]  = lookIDPad[-2,:,:]
            lookIDPad[-1,:,:] = lookIDPad[1,:,:]
            lookPerI[0,:,:] = 1
            lookPerI[-1,:,:] = -1

        if (self.Domain.boundaries[1][0] == 2):
            lookIDPad[:,0,:]  = lookIDPad[:,-2,:]
            lookIDPad[:,-1,:] = lookIDPad[:,1,:]
            lookPerJ[:,0,:] = 1
            lookPerJ[:,-1,:] = -1

        if (self.Domain.boundaries[2][0] == 2):
            lookIDPad[:,:,0]  = lookIDPad[:,:,-2]
            lookIDPad[:,:,-1] = lookIDPad[:,:,1]
            lookPerK[:,:,0] = 1
            lookPerK[:,:,-1] = -1

        for cc,f in enumerate(self.Orientation.faces.values()):
            cx = f['ID'][0] + self.subID[0] + 1
            cy = f['ID'][1] + self.subID[1] + 1
            cz = f['ID'][2] + self.subID[2] + 1
            self.neighborF[cc]      = lookIDPad[cx,cy,cz]
            self.neighborPerF[cc,0] = lookPerI[cx,cy,cz]
            self.neighborPerF[cc,1] = lookPerJ[cx,cy,cz]
            self.neighborPerF[cc,2] = lookPerK[cx,cy,cz]

        for cc,e in enumerate(self.Orientation.edges.values()):
            cx = e['ID'][0] + self.subID[0] + 1
            cy = e['ID'][1] + self.subID[1] + 1
            cz = e['ID'][2] + self.subID[2] + 1
            self.neighborE[cc]      = lookIDPad[cx,cy,cz]
            self.neighborPerE[cc,0] = lookPerI[cx,cy,cz]
            self.neighborPerE[cc,1] = lookPerJ[cx,cy,cz]
            self.neighborPerE[cc,2] = lookPerK[cx,cy,cz]
            
            ### Determine if External Face or Internal Edge
            if self.neighborE[cc] == -2:
                sum = 0
                for f in e['faceIndex']:
                    if self.neighborF[f] == -2:
                        self.externalE[cc] = f ## External Face ID
                    sum += self.neighborF[f]
                if sum != -4:
                    self.neighborE[cc] = -3 ## Internal Edge / External Face
                else:
                    self.externalE[cc] = -1 


        for cc,c in enumerate(self.Orientation.corners.values()):
            cx = c['ID'][0] + self.subID[0] + 1
            cy = c['ID'][1] + self.subID[1] + 1
            cz = c['ID'][2] + self.subID[2] + 1
            self.neighborC[cc]      = lookIDPad[cx,cy,cz]
            self.neighborPerC[cc,0] = lookPerI[cx,cy,cz]
            self.neighborPerC[cc,1] = lookPerJ[cx,cy,cz]
            self.neighborPerC[cc,2] = lookPerK[cx,cy,cz]

            ### Determine if External Edge / Internal Corner or External Face / Internal Corner
            if self.neighborC[cc] == -2:
                sum = 0
                for f in c['faceIndex']:
                    if self.neighborF[f] < 0:
                        sum += self.neighborF[f]
                if sum == -4: ## Internal Corner / External Edge
                    self.neighborC[cc] = -4 
                    for e in c['edgeIndex']:
                        if self.neighborE[e] == -2:
                            self.externalC[cc] = e ## External Edge ID
                if sum == -2: ## Internal Corner / External Face
                    self.neighborC[cc] = -3 
                    for f in c['faceIndex']:
                      if self.neighborF[f] == -2:
                          self.externalC[cc] = f ## External Face ID


        self.lookUpID = lookIDPad

    def loadBalancing(self):
        loadData = [self.ID,np.prod(self.ownNodes)]
        loadData = comm.gather(loadData, root=0)
        if self.ID == 0:
            sumTotalNodes = 0
            for ld in loadData:
                sumTotalNodes = sumTotalNodes + ld[2]
            print("Total Nodes",sumTotalNodes,"Pore Nodes",self.totalPoreNodes)
            print("Ideal Load Balancing is %2.1f%%" %(1./self.size*100.))
            for ld in loadData:
                p = ld[1]/self.totalPoreNodes*100.
                t = ld[2]/sumTotalNodes*100.
                print("Rank: %i has %2.1f%% of the Pore Nodes and %2.1f%% of the total Nodes" %(ld[0],p,t))

    def trimSphereData(self,sphereData):
        numObj = sphereData.shape[1]
        xList = []
        yList = []
        zList = []
        rList = []
        for i in range(numObj):
            x = sphereData[0,i]
            y = sphereData[1,i]
            z = sphereData[2,i]
            r = sphereData[3,i]
            r_sqrt = np.sqrt(sphereData[3,i])
            xCheck = self.x[0] - self.Domain.dX - r_sqrt <= x <= self.x[-1] + self.Domain.dX + r_sqrt
            yCheck = self.y[0] - self.Domain.dY - r_sqrt <= y <= self.y[-1] + self.Domain.dY + r_sqrt
            zCheck = self.z[0] - self.Domain.dZ - r_sqrt <= z <= self.z[-1] + self.Domain.dZ + r_sqrt
            if xCheck and yCheck and zCheck:
                xList.append(x)
                yList.append(y)
                zList.append(z)
                rList.append(r)
        return np.array([xList,yList,zList,rList])


def genDomainSubDomain(rank,size,subDomains,nodes,boundaries,inlet,outlet,dataFormat,file,dataRead,dataReadkwargs=None):

    numSubDomains = np.prod(subDomains)

    if (size != numSubDomains):
        if rank==0: 
            print("Number of Subdomains Must Equal Number of Processors!...Exiting")
        communication.raiseError()
        
    ### Get Domain INFO for All Procs ###
    if file is not None:
        if dataReadkwargs is None:
            domainSize,sphereData = dataRead(file)
        else:
            domainSize,sphereData = dataRead(file,dataReadkwargs)
    if file is None:
        domainSize = np.array([[0.,14.],[-1.5,1.5],[-1.5,1.5]])
    domain = Domain.Domain(nodes = nodes, domainSize = domainSize, subDomains = subDomains, boundaries = boundaries, inlet=inlet, outlet=outlet)
    domain.getdXYZ()
    domain.getSubNodes()

    orient = Orientation.Orientation()

    sD = subDomain(Domain = domain, ID = rank, subDomains = subDomains, Orientation = orient)
    sD.getInfo()
    sD.getNeighbors()
    sD.getXYZ(sD.buffer)
    
    if file is not None:
        sphereData = sD.trimSphereData(sphereData)
        pM = porousMedia.genPorousMedia(sD,dataFormat,sphereData,resSize = 1)
    else:
        pM = porousMedia.genPorousMedia(sD,dataFormat,resSize = 1)


    return domain,sD,pM
