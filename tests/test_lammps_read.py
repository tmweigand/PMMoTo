from mpi4py import MPI
import pmmoto

def my_function():

    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()

    subdomains = [1,1,1] # Specifies how Domain is broken among rrocs
    nodes = [50,50,50] # Total Number of Nodes in Domain

    ## Ordering for Inlet/Outlet ( (-x,+x) , (-y,+y) , (-z,+z) )
    boundaries = [[2,2],[2,2],[2,2]] # 0: Nothing Assumed  1: Walls 2: Periodic
    inlet  = [[0,0],[0,0],[0,0]]
    outlet = [[0,0],[0,0],[0,0]]


    r_lookup_file = './testDomains/molecular/single_atom_rlookup.txt'
    r_lookup = pmmoto.io.read_r_lookup_file(r_lookup_file)

    lammps_file = './testDomains/molecular/lammps_single_atom.out'
    atom_locations,atom_type,domain_data = pmmoto.io.read_lammps_atoms(lammps_file)

    print(atom_locations,atom_type,r_lookup,domain_data)

    sd = pmmoto.initialize(rank,size,subdomains,nodes,boundaries,inlet,outlet)
    sd.update_domain_size(domain_data)
    pm = pmmoto.domain_generation.gen_pm_atom_domain(sd,atom_locations,atom_type,r_lookup)
    atom_locations,atom_types = pmmoto.domain_generation.gen_periodic_atoms(sd,atom_locations,atom_type,r_lookup)
    pm_periodic = pmmoto.domain_generation.gen_pm_atom_domain(sd,atom_locations,atom_type,r_lookup)

    pmmoto.io.save_grid_data("dataOut/test_lammps_read_grid",sd,pm.grid,periodic=pm_periodic.grid)


if __name__ == "__main__":
    my_function()
    MPI.Finalize()
