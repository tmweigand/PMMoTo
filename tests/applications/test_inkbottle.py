import numpy as np
from mpi4py import MPI
import pmmoto

def my_function():

    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()

    subdomains = [1,1,1] # Specifies how Domain is broken among procs
    nodes = [140,30,30] # Total Number of Nodes in Domain

    ## Ordering for Inlet/Outlet ( (-x,+x) , (-y,+y) , (-z,+z) )
    boundaries = [[0,0],[0,0],[0,0]] # 0: Nothing Assumed  1: Walls 2: Periodic
    inlet  = [[0,0],[0,0],[0,0]]
    outlet = [[0,0],[0,0],[0,0]]

    domain_size = np.array([[0.,14.],[-1.5,1.5],[-1.5,1.5]])
    sd = pmmoto.initialize(rank,size,subdomains,nodes,boundaries,inlet,outlet)
    pm = pmmoto.domain_generation.gen_pm_inkbottle(sd,domain_size,res_size = 0)

    # Multiphase parameters
    num_fluid_phases = 2

    w_inlet  = [[1,0],[0,0],[0,0]]
    nw_inlet = [[0,1],[0,0],[0,0]]
    mp_inlets = {1:w_inlet,
                 2:nw_inlet}

    w_outlet  = [[0,0],[0,0],[0,0]]
    nw_outlet = [[0,0],[0,0],[0,0]]
    mp_outlets = {1:w_outlet,
                  2:nw_outlet}

    # Initalize multiphase grid
    mp = pmmoto.initialize_mp(pm,num_fluid_phases,mp_inlets,mp_outlets,res_size = 10)
    mp = pmmoto.domain_generation.gen_mp_constant(mp,2)

    print(sd.index_own_nodes,mp.index_own_nodes)

    pc = [2.46250]
    gamma = 1.
    pmmoto.filters.multiPhase.calc_drainage(mp,pc,gamma)


    ### Save Grid Data where kwargs are used for saving other grid data (i.e. EDT, Medial Axis)
    pmmoto.io.save_grid_data("dataOut/test_inkbottle",sd,pm.grid,mp = mp.grid)

if __name__ == "__main__":
    my_function()
    MPI.Finalize()
