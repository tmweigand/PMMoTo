import numpy as np
from mpi4py import MPI
import pmmoto

def my_function():

    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()

    subdomains = [2,2,2] # Specifies how domain is broken among processes
    nodes = [200,200,200] # Total Number of Nodes in Domain

    ## Ordering for Inlet/Outlet ( (-xpip,+x) , (-y,+y) , (-z,+z) )
    boundaries = [[2,2],[2,2],[2,2]] # 0: Nothing Assumed  1: Walls 2: Periodic
    inlet  = [[0,0],[0,0],[0,0]]
    outlet = [[0,0],[0,0],[0,0]]

    file = './testDomains/50pack.out'
    #file = './testDomains/bcc.out'


    sd = pmmoto.initialize(rank,size,subdomains,nodes,boundaries,inlet,outlet)
    sphere_data,domain_data = pmmoto.io.read_sphere_pack_xyzr_domain(file)
    pm = pmmoto.domain_generation.gen_pm_spheres_domain(sd,sphere_data,domain_data)
    edt = pmmoto.filters.calc_edt(sd,pm.grid)

    ### Save Grid Data where kwargs are used for saving other grid data (i.e. EDT, Medial Axis)
    pmmoto.io.save_grid_data("dataOut/test_parallel_minkowski",sd,pm.grid,dist=edt)

    fun = pmmoto.analysis.minkowski.functionals(sd,np.logical_not(pm.grid))

    if rank == 0:
        print(fun[0])
        m_vol = fun[0]
        m_sa = 8*fun[1]
        m_curv = 4*np.pi*np.pi*fun[2]
        m_euler = 4*np.pi/3*fun[3]

        print(m_vol,m_sa,m_curv,m_euler)

        vol = 4/3*np.pi*0.25*0.25*0.25
        sa = 4*np.pi*0.25*0.25
        curv = sa*((1./0.25) + (1./0.25))
        euler = sa/(0.25*0.25)/(4*np.pi)

        print(vol,sa,curv,euler)


if __name__ == "__main__":
    my_function()
    MPI.Finalize()
