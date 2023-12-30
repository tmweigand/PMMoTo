import numpy as np
from mpi4py import MPI
from scipy.ndimage import distance_transform_edt
import edt
import time
import PMMoTo

def my_function():

    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()

    subDomains = [1,1,1] # Specifies how Domain is broken among procs
    nodes = [300,300,300] # Total Number of Nodes in Domain

    ## Ordering for Inlet/Outlet ( (-x,+x) , (-y,+y) , (-z,+z) )
    boundaries = [[2,2],[2,2],[2,2]] # 0: Nothing Assumed  1: Walls 2: Periodic
    inlet  = [[0,0],[0,0],[0,0]]
    outlet = [[0,0],[0,0],[0,0]]

    file = './testDomains/50pack.out'


    domain,sDL,pML = PMMoTo.genDomainSubDomain(rank,size,subDomains,nodes,boundaries,inlet,outlet,"Sphere",file,PMMoTo.readPorousMediaXYZR)


    morph_grid = PMMoTo.morph_add(subdomain = sDL,grid =pML.grid,phase = 0,radius =domain.voxel[0]*4)
    morph_grid2 = PMMoTo.morph_add(subdomain = sDL,grid =pML.grid,phase = 1,radius =domain.voxel[0]*4)


    ### Save Grid Data where kwargs are used for saving other grid data (i.e. EDT, Medial Axis)
    PMMoTo.saveGridData("dataOut/test_morph",rank,domain,sDL,pML.grid,morph=morph_grid,morph2 = morph_grid2)


if __name__ == "__main__":
    my_function()
    MPI.Finalize()
