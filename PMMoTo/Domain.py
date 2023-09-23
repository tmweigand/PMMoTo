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
    