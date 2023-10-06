import numpy as np
from mpi4py import MPI
from .. import communication
from . import medialExtraction
from .. import nodes
from. import medialNodes
from . import medialSets
comm = MPI.COMM_WORLD


class medialAxis(object):
    """
    Calculate Medial Axis and PostProcess
    Nodes -> Sets -> Paths
    Sets are broken into Reaches -> Medial Nodes -> Medial Clusters
    """

    def __init__(self,Domain,subDomain,grid):
        self.Domain = Domain
        self.subDomain = subDomain
        self.grid = grid
        self.Orientation = subDomain.Orientation
        self.padding = np.zeros([3],dtype=np.int64)
        self.haloGrid = None
        self.halo = np.zeros(6)
        self.haloPadNeigh = np.zeros(6)
        self.haloPadNeighNot = np.zeros(6)
        self.MA = None

    def genMAArrays(self):
        """
        Generate Trimmed MA arrays to get nodeInfo and Correct Neighbor Counts for Boundary Nodes
        """
        dim = self.MA.shape
        tempMA = self.MA[self.haloPadNeigh[0] : dim[0] - self.haloPadNeigh[1],
                         self.haloPadNeigh[2] : dim[1] - self.haloPadNeigh[3],
                         self.haloPadNeigh[4] : dim[2] - self.haloPadNeigh[5]]
                
        neighMA = np.pad(self.MA, ( (self.haloPadNeighNot[0], self.haloPadNeighNot[1]), 
                                    (self.haloPadNeighNot[2], self.haloPadNeighNot[3]), 
                                    (self.haloPadNeighNot[4], self.haloPadNeighNot[5]) ), 
                                    'constant', constant_values=0)
        return tempMA,neighMA

def medialAxisEval(subDomain,porousMedia,grid,distance,connect = False, trim = False):

    rank = subDomain.ID
    size = subDomain.Domain.numSubDomains

    grid = porousMedia.grid

    ### Initialize Classes
    sDMA = medialAxis(Domain = subDomain.Domain, subDomain = subDomain, grid = grid)

    ### Extract MA
    mE = medialExtraction.medialExtraction(Domain = subDomain.Domain, subDomain = subDomain, grid = grid, edt = distance)

    if not connect:
        sDMA.MA = mE.extractMedialAxis(connect)

    if connect:

        ### CLEAN this up with Array and MA Neighbors
        sDMA.MA,sDMA.haloPadNeigh,sDMA.haloPadNeighNot = mE.extractMedialAxis(connect)

        ### Get Info for Medial Axis Nodes and Get Connected Sets and Boundary Sets
        tempMA,neighMA = sDMA.genMAArrays()


        #print(sDMA.subDomain.ID,porousMedia.inlet,porousMedia.outlet,subDomain.Domain,porousMedia.loopInfo,subDomain,subDomain.Orientation)

        Nodes = nodes.getNodeInfo(rank,tempMA,1,porousMedia.inlet,porousMedia.outlet,subDomain.Domain,porousMedia.loopInfo,subDomain,subDomain.Orientation)
        maNodesType = medialNodes.updateMANeighborCount(neighMA,porousMedia,subDomain.Orientation,Nodes[0])
        sDMA.MA = np.ascontiguousarray(tempMA)
        
        mSets = medialNodes.getConnectedMedialAxis(subDomain,sDMA.MA,Nodes,maNodesType)

        sDMA.Sets = mSets


        mSets.get_boundary_sets()
        mSets.pack_boundary_data()

        boundaryData = communication.setCOMMNEW(subDomain.Orientation,subDomain,mSets.boundaryData)

        mSets.unpack_boundary_data(boundaryData)
        mSets.match_boundary_sets()
        mSets.pack_matched_sets()

        ### Generate global information for sets 
        allMatchedSetData = comm.gather(mSets.matchedSetData, root=0)
        mSets.organize_matched_sets(allMatchedSetData)

        ### Generate and Update global ID information
        globalIDInfo = comm.gather([mSets.setCount,mSets.boundarySetCount], root=0)
        mSets.organize_globalSetID(globalIDInfo)
        mSets.update_globalSetID()
        mSets.update_connected_sets()


        ### Collect paths from medialSets
        mPaths = mSets.collect_paths()
        mPaths.get_boundary_paths()
        mPaths.pack_boundary_data()
        boundaryData = communication.setCOMMNEW(subDomain.Orientation,subDomain,mPaths.boundaryData)

        mPaths.unpack_boundary_data(boundaryData)
        mPaths.match_boundary_paths()
        mPaths.pack_matched_paths()

        ### Generate global information for paths 
        allMatchedPathData = comm.gather(mPaths.matchedPathData, root=0)
        mPaths.organize_matched_paths(allMatchedPathData)

        ### Generate and Update global ID information
        globalIDInfo = comm.gather([mPaths.pathCount,mPaths.boundaryPathCount], root=0)
        mPaths.organize_globalPathID(globalIDInfo)
        mPaths.update_globalPathID()


        if trim:
            if sDMA.subDomain.ID == 0:
                print("Trimming...")

            ### Trim paths not connected to inlet and outlet
            mPaths.trim_paths()

            ### Trim sets that are not
            mSets.trim_sets()
            mSets.update_trimmed_connected_sets()
            mSets.pack_untrimmed_sets()
            allTrimSetData = comm.gather(mSets.trimSetData, root=0)
            setInfo,indexMap = mSets.unpack_untrimmed_sets(allTrimSetData)
            setInfo = mSets.serial_trim_sets(setInfo,indexMap)
            setInfo = mSets.repack_global_trimmed_sets(setInfo)



    return sDMA


def medialAxisTrim(sDMA,porousMedia,subDomain,distance,cutoffs):

    rank = subDomain.ID
    size = subDomain.Domain.numSubDomains

    return sDMA
