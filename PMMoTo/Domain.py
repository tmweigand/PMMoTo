import numpy as np
from . import communication

class Domain(object):
    """
    Determine information for entire Domain
    """
    def __init__(self,
                 nodes,
                 domainSize,
                 subDomains = [1,1,1],
                 boundaries = [[0,0],[0,0],[0,0]],
                 inlet = [[0,0],[0,0],[0,0]],
                 outlet  =[[0,0],[0,0],[0,0]]):
        self.nodes        = nodes
        self.domainSize   = domainSize
        self.boundaries   = boundaries
        self.subDomains   = subDomains
        self.numSubDomains = np.prod(subDomains)
        self.subNodes     = np.zeros([3],dtype=np.uint64)
        self.subNodesRem  = np.zeros([3],dtype=np.uint64)
        self.domainLength = np.zeros([3])
        self.inlet = inlet
        self.outlet = outlet
        self.dX = 0
        self.dY = 0
        self.dZ = 0
        self.inputChecks()

    def getdXYZ(self):
        """
        Get domain length and voxel size
        """
        self.domainLength[0] = (self.domainSize[0,1]-self.domainSize[0,0])
        self.domainLength[1] = (self.domainSize[1,1]-self.domainSize[1,0])
        self.domainLength[2] = (self.domainSize[2,1]-self.domainSize[2,0])
        self.dX = self.domainLength[0]/self.nodes[0]
        self.dY = self.domainLength[1]/self.nodes[1]
        self.dZ = self.domainLength[2]/self.nodes[2]

    def getSubNodes(self):
        """
        Calculate number of voxels in each subDomain
        """
        self.subNodes[0],self.subNodesRem[0] = divmod(self.nodes[0],self.subDomains[0])
        self.subNodes[1],self.subNodesRem[1] = divmod(self.nodes[1],self.subDomains[1])
        self.subNodes[2],self.subNodesRem[2] = divmod(self.nodes[2],self.subDomains[2])

    def inputChecks(self):
        """
        Ensure Input Parameters are Valid
        """
        error = False

        ### Check Nodes are Positive
        for n in self.nodes:
            if n <= 0:
                error = True
                print("Error: Nodes must be positive integer!")
        
        ### Check subDomain Size
        for n in self.subDomains:
            if n <= 0:
                error = True
                print("Error: Number of Subdomains must be positive integer!")

        ### Check Boundaries and Boundary Pairs
        for dir in self.boundaries:
            for n in dir:
                if n < 0 or n > 2:
                    error = True
                    print("Error: Allowable Boundary IDs are (0) None (1) Walls (2) Periodic")    
                

        ### Check Inlet Condition
        sN = 0
        for dirI,dirB in zip(self.inlet,self.boundaries):
            for nI,nB in zip(dirI,dirB):
                if nI !=0:
                    sN = sN + 1
                    if nB != 0:
                        error = True
                        print("Error: Boundary must be type (0) None at Inlet")
        if sN > 1:
            error = True
            print("Error: Only 1 Inlet Allowed")

        ### Check Outlet Condition
        sN = 0
        for dirI,dirB in zip(self.outlet,self.boundaries):
            for nI,nB in zip(dirI,dirB): 
                if nI !=0:
                    sN = sN + 1
                    if nB != 0:
                        error = True
                        print("Error: Boundary must be type (0) None at Outlet")
        if sN > 1:
            error = True
            print("Error: Only 1 Outlet Allowed")  

        if error:
          communication.raiseError()
    
    def periodicImageSphereData(self,sphereData):
        numObj = sphereData.shape[1]
        replicateSphereData = [[],[],[],[]]
        newSphereData = [[],[],[],[]]
        for i in range(numObj):
            x = sphereData[0,i]
            y = sphereData[1,i]
            z = sphereData[2,i]
            r = sphereData[3,i]
            r_sqrt = np.sqrt(sphereData[3,i])
            nearBotX = x-r_sqrt <= self.domainSize[0,0]
            nearTopX = x+r_sqrt >= self.domainSize[0,1]
            nearBotY = y-r_sqrt <= self.domainSize[1,0]
            nearTopY = y+r_sqrt >= self.domainSize[1,1]
            nearBotZ = z-r_sqrt <= self.domainSize[2,0]
            nearTopZ = z+r_sqrt >= self.domainSize[2,1]
            checks = [nearBotX, nearTopX,
                    nearBotY, nearTopY,
                    nearBotZ, nearTopZ]
            coords = [x,y,z]
            if not any(checks):
                continue
            if self.boundaries[2]==[2,2]:
                if checks[4]:
                    shiftedZ = z+self.domainLength[2]
                    replicateSphereData[0].append(x)
                    replicateSphereData[1].append(y)
                    replicateSphereData[2].append(shiftedZ)
                    replicateSphereData[3].append(r)                
                elif checks [5]:
                    shiftedZ = z-self.domainLength[2]
                    replicateSphereData[0].append(x)
                    replicateSphereData[1].append(y)
                    replicateSphereData[2].append(shiftedZ)
                    replicateSphereData[3].append(r)
                        
            if self.boundaries[0]==[2,2]:
                if checks[0]:
                    shiftedX = x+self.domainLength[0]
                    replicateSphereData[0].append(shiftedX)
                    replicateSphereData[1].append(y)
                    replicateSphereData[2].append(z)
                    replicateSphereData[3].append(r)                
                elif checks[1]:
                    shiftedX = x-self.domainLength[0]
                    replicateSphereData[0].append(shiftedX)
                    replicateSphereData[1].append(y)
                    replicateSphereData[2].append(z)
                    replicateSphereData[3].append(r)
                else:
                    shiftedX = x
            
                if self.boundaries[1]==[2,2]:
                    if checks[2]:
                        shiftedY = y+self.domainLength[1]
                        replicateSphereData[0].append(shiftedX)
                        replicateSphereData[1].append(shiftedY)
                        replicateSphereData[2].append(z)
                        replicateSphereData[3].append(r)
                    elif checks [3]:
                        shiftedY = y-self.domainLength[1]
                        replicateSphereData[0].append(shiftedX)
                        replicateSphereData[1].append(shiftedY)
                        replicateSphereData[2].append(z)
                        replicateSphereData[3].append(r)
                    else:
                        shiftedY = y
            
                    if self.boundaries[2]==[2,2]:
                        if checks[4]:
                            shiftedZ = z+self.domainLength[2]
                            replicateSphereData[0].append(shiftedX)
                            replicateSphereData[1].append(shiftedY)
                            replicateSphereData[2].append(shiftedZ)
                            replicateSphereData[3].append(r)
                        elif checks [5]:
                            shiftedZ = z-self.domainLength[2]
                            replicateSphereData[0].append(shiftedX)
                            replicateSphereData[1].append(shiftedY)
                            replicateSphereData[2].append(shiftedZ)
                            replicateSphereData[3].append(r)
                if self.boundaries[2]==[2,2]:
                    if checks[4]:
                        shiftedZ = z+self.domainLength[2]
                        replicateSphereData[0].append(shiftedX)
                        replicateSphereData[1].append(y)
                        replicateSphereData[2].append(shiftedZ)
                        replicateSphereData[3].append(r)
                    elif checks [5]:
                        shiftedZ = z-self.domainLength[2]
                        replicateSphereData[0].append(shiftedX)
                        replicateSphereData[1].append(y)
                        replicateSphereData[2].append(shiftedZ)
                        replicateSphereData[3].append(r)
            if self.boundaries[1]==[2,2]:
                if checks[2]:
                    shiftedY = y+self.domainLength[1]
                    replicateSphereData[0].append(x)
                    replicateSphereData[1].append(shiftedY)
                    replicateSphereData[2].append(z)
                    replicateSphereData[3].append(r)                
                elif checks [3]:
                    shiftedY = y-self.domainLength[1]
                    replicateSphereData[0].append(x)
                    replicateSphereData[1].append(shiftedY)
                    replicateSphereData[2].append(z)
                    replicateSphereData[3].append(r)
                else:
                    shiftedY = y
                if self.boundaries[2]==[2,2]:
                    if checks[4]:
                        shiftedZ = z+self.domainLength[2]
                        replicateSphereData[0].append(x)
                        replicateSphereData[1].append(shiftedY)
                        replicateSphereData[2].append(shiftedZ)
                        replicateSphereData[3].append(r)
                    elif checks [5]:
                        shiftedZ = z-self.domainLength[2]
                        replicateSphereData[0].append(x)
                        replicateSphereData[1].append(shiftedY)
                        replicateSphereData[2].append(shiftedZ)
                        replicateSphereData[3].append(r)
        newSphereData[0] = sphereData[0].tolist()+replicateSphereData[0]
        newSphereData[1] = sphereData[1].tolist()+replicateSphereData[1]
        newSphereData[2] = sphereData[2].tolist()+replicateSphereData[2]
        newSphereData[3] = sphereData[3].tolist()+replicateSphereData[3]
        return np.array(newSphereData)