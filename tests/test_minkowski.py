import numpy as np
from mpi4py import MPI
import pmmoto

def my_function():

    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()

    subdomains = [1,1,1] # Specifies how domain is broken among processes
    nodes = [400,400,400] # Total Number of Nodes in Domain

    ## Ordering for Inlet/Outlet ( (-x,+x) , (-y,+y) , (-z,+z) )
    boundaries = [[0,0],[0,0],[0,0]] # 0: Nothing Assumed  1: Walls 2: Periodic
    inlet  = [[0,0],[0,0],[0,0]]
    outlet = [[0,0],[0,0],[0,0]]

    file = './testDomains/1pack.out'
    #file = './testDomains/bcc.out'


    sd = pmmoto.initialize(rank,size,subdomains,nodes,boundaries,inlet,outlet)
    sphere_data,domain_data = pmmoto.io.read_sphere_pack_xyzr_domain(file)
    pm = pmmoto.domain_generation.gen_pm_spheres_domain(sd,sphere_data,domain_data)
    edt = pmmoto.filters.calc_edt(sd,pm.grid)

    ### Save Grid Data where kwargs are used for saving other grid data (i.e. EDT, Medial Axis)
    pmmoto.io.save_grid_data("dataOut/test_minkowski",sd,pm.grid,dist=edt)


    fun = pmmoto.analysis.minkowski.functionals(sd,np.logical_not(pm.grid))
    print(fun)

    radius = 0.25

    vol = 4/3*np.pi*radius**3
    sa = 4*np.pi*radius*radius
    curv = sa*((1./radius) + (1./radius))
    euler = sa/(radius*radius)/(4*np.pi)

    print(vol,sa,curv,euler)


if __name__ == "__main__":
    my_function()
    MPI.Finalize()
