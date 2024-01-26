import numpy as np
from mpi4py import MPI
import pmmoto
import time
import cProfile

def profile(filename=None, comm=MPI.COMM_WORLD):
  def prof_decorator(f):
    def wrap_f(*args, **kwargs):
      pr = cProfile.Profile()
      pr.enable()
      result = f(*args, **kwargs)
      pr.disable()

      if filename is None:
        pr.print_stats()
      else:
        filename_r = filename + ".{}".format(comm.rank)
        pr.dump_stats(filename_r)

      return result
    return wrap_f
  return prof_decorator

@profile(filename="profile_out")
def my_function():

    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()

    subdomains = [1,1,1] # Specifies how Domain is broken among procs
    nodes = [100,100,100] # Total Number of Nodes in Domain

    ## Ordering for Inlet/Outlet ( (-x,+x) , (-y,+y) , (-z,+z) )
    boundaries = [[2,2],[2,2],[2,2]] # 0: Nothing Assumed  1: Walls 2: Periodic
    #boundaries = [[0,0],[0,0],[0,0]] # 0: Nothing Assumed  1: Walls 2: Periodic
    inlet  = [[0,0],[0,0],[0,0]]
    outlet = [[0,0],[0,0],[0,0]]

    file = './testDomains/50pack.out'

    save_data = True

    sd = pmmoto.initialize(rank,size,subdomains,nodes,boundaries,inlet,outlet)
    sphere_data,domain_data = pmmoto.io.read_sphere_pack_xyzr_domain(file)
    pm = pmmoto.domain_generation.gen_pm_spheres_domain(sd,sphere_data,domain_data)

    pm.grid[0:20,0:20,0:20] = 2
    pm.grid[-10:,-10:,-10:] = 2

    start = time.time()
    # pmmoto_sets = pmmoto.core.collect_sets(pm.grid,1,pm.inlet,pm.outlet,pm.loop_info,sd)
    # pmmoto_sets = pmmoto.core.collect_sets(pm.grid,0,pm.inlet,pm.outlet,pm.loop_info,sd)
    #print("PM Time:",time.time()-start)

    start = time.time()
    connected_sets = pmmoto.core.connect_all_phases(pm,pm.inlet,pm.outlet)
    #print("CC Time:",time.time()-start)

    if save_data:
        pmmoto.io.save_grid_data("dataOut/test_morph",sd,pm.grid)

        #pmmoto.io.save_set_data("dataOut/test_connect_sets",sd,pmmoto_sets)
        pmmoto.io.save_set_data("dataOut/test_new_connect_sets",sd,connected_sets)


if __name__ == "__main__":
    my_function()
    MPI.Finalize()
