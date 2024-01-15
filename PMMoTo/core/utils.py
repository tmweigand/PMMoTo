### Core Utility Functions ###
import sys
import numpy as np
from mpi4py import MPI
from . import Orientation

comm = MPI.COMM_WORLD

def raise_error():
    """Exit gracefuully.
    """
    MPI.Finalize()
    sys.exit()

def check_grid(subdomain,grid):
    """Esure solid voxel on each subprocess

    """
    if np.sum(grid) == np.prod(subdomain.nodes):
        print("This code requires at least 1 solid voxel in each subdomain. Please reorder processors!")
        raise_error()

def check_inputs(mpi_size,subdomains,nodes,boundaries,inlet,outlet):
    """
    Ensure Input Parameters are Valid
    """
    check_input_nodes(nodes)
    check_subdomain_size(mpi_size,subdomains)
    check_boundaries(boundaries)
    check_inlet_outlet(boundaries,inlet,outlet)
       
def check_input_nodes(nodes):
    """Check Nodes are Positive
    """
    error = False

    for n in nodes:
        if n <= 0:
            error = True
            print("Error: Nodes must be positive integer!")

    if error:
        raise_error()

def check_subdomain_size(mpi_size,subdomains):
    """
    Check subdomain size and ensure mpi size is equal to num_subdomains
    """
    error = False
    for n in subdomains:
        if n <= 0:
            error = True
            print("Error: Number of Subdomains must be positive integer!")

    num_subdomains = np.prod(subdomains)

    if mpi_size != num_subdomains:
        error = True
        print("Error: Number of MPI processes must equal number of subdomains!")

    if error:
        raise_error()

def check_boundaries(boundaries):
    """
    Check boundaries and boundary pairs
    """
    error = False
    for d in boundaries:
        for n in d:
            if n < 0 or n > 2:
                error = True
                print("Error: Allowable Boundary IDs are (0) None (1) Walls (2) Periodic")
    if error:
        raise_error()

def check_inlet_outlet(boundaries,inlet,outlet):
    """
    Check inlet and outlet conditions
    """
    error = False

    # Inlet
    n_sum = 0
    for d_in,d_bound in zip(inlet,boundaries):
        for n_in,n_bound in zip(d_in,d_bound):
            if n_in !=0:
                n_sum = n_sum + 1
                if n_bound != 0:
                    error = True
                    print("Error: Boundary must be type (0) None at Inlet")
    if n_sum > 1:
        error = True
        print("Error: Only 1 Inlet Allowed")

    # Outlet
    n_sum = 0
    for d_in,d_bound in zip(outlet,boundaries):
        for n_in,n_bound in zip(d_in,d_bound): 
            if n_in !=0:
                n_sum = n_sum + 1
                if n_bound != 0:
                    error = True
                    print("Error: Boundary must be type (0) None at Outlet")
    if n_sum > 1:
        error = True
        print("Error: Only 1 Outlet Allowed")

    if error:
        raise_error()

def unpad(grid,pad):
    """
    Unpad a padded array
    """
    _dim = grid.shape
    grid_out = grid[pad[0]:_dim[0]-pad[1],
                    pad[2]:_dim[1]-pad[3],
                    pad[4]:_dim[2]-pad[5]]
    return np.ascontiguousarray(grid_out)

def own_grid(grid,own):
    """
    Pass array with only nodes owned py that process
    """
    grid_out =  grid[own[0]:own[1],
                     own[2]:own[3],
                     own[4]:own[5]]
    
    return np.ascontiguousarray(grid_out)


def partition_boundary_solids(subdomain,solids,extend_factor = 0.7):
    """
    Trim solids to minimize communication and reduce KD Tree. Identify on Surfaces, Edges, and Corners
    Keep all face solids, and use extend factor to query which solids to include for edges and corners
    """
    face_solids = [[] for _ in range(len(Orientation.faces))]
    edge_solids = [[] for _ in range(len(Orientation.edges))]
    corner_solids = [[] for _ in range(len(Orientation.corners))]
    
    extend = [extend_factor*x for x in subdomain.size_subdomain]
    coords = subdomain.coords

    ### Faces ###
    for fIndex in Orientation.faces:
        pointsXYZ = []
        points = solids[np.where( (solids[:,0]>-1)
                                & (solids[:,1]>-1)
                                & (solids[:,2]>-1)
                                & (solids[:,3]==fIndex) )][:,0:3]
        for x,y,z in points:
            pointsXYZ.append([coords[0][x],coords[1][y],coords[2][z]] )
        face_solids[fIndex] = np.asarray(pointsXYZ)

    ### Edges ###
    for edge in subdomain.edges:
        edge.get_extension(extend,subdomain.bounds)
        for f,d in zip(edge.info['faceIndex'],reversed(edge.info['dir'])): # Flip dir for correct nodes
            f_solids = face_solids[f]
            values = (edge.extend[d][0] <= f_solids[:,d]) & (f_solids[:,d] <= edge.extend[d][1])
            if len(edge_solids[edge.ID]) == 0:
                edge_solids[edge.ID] = f_solids[np.where(values)]
            else:
                edge_solids[edge.ID] = np.append(edge_solids[edge.ID],f_solids[np.where(values)],axis=0)
        edge_solids[edge.ID] = np.unique(edge_solids[edge.ID],axis=0)

    ### Corners ###
    iterates = [[1,2],[0,2],[0,1]]
    for corner in subdomain.corners:
        corner.get_extension(extend,subdomain.bounds)
        values = [None,None]
        for it,f in zip(iterates,corner.info['faceIndex']):
            f_solids = face_solids[f]
            for n,i in enumerate(it):
                values[n] = (corner.extend[i][0] <= f_solids[:,i]) & (f_solids[:,i] <= corner.extend[i][1])
            if len(corner_solids[corner.ID]) == 0:
                corner_solids[corner.ID] = f_solids[np.where(values[0] & values[1])]
            else:
                corner_solids[corner.ID] = np.append(corner_solids[corner.ID],f_solids[np.where(values[0] & values[1])],axis=0)
        corner_solids[corner.ID] = np.unique(corner_solids[corner.ID],axis=0)

    return face_solids,edge_solids,corner_solids




def reconstruct_grid(subdomain,grid):
    """This function (re)constructs a grid from all proccesses to root
    """

    if subdomain.ID == 0:
        sd_all = np.empty((subdomain.domain.num_subdomains), dtype = object)
        grid_all = np.empty((subdomain.domain.num_subdomains), dtype = object)
        sd_all[0] = subdomain
        grid_all[0] = grid
        for neigh in range(1,subdomain.domain.num_subdomains):
            sd_all[neigh] = comm.recv(source=neigh)
            grid_all[neigh] = comm.recv(source=neigh)

    if subdomain.ID > 0:
        comm.send(subdomain,dest=0)
        comm.send(grid,dest=0)

    if subdomain.ID == 0:
        grid_out = np.zeros(subdomain.domain.nodes)
        for n in range(0,subdomain.domain.num_subdomains):
            _own_grid = own_grid(grid_all[n],sd_all[n].index_own_nodes)
            grid_out[sd_all[n].index_start[0] + sd_all[n].buffer[0]:
                     sd_all[n].index_start[0] + sd_all[n].nodes[0] - sd_all[n].buffer[1],
                     sd_all[n].index_start[1] + sd_all[n].buffer[2]:
                     sd_all[n].index_start[1] + sd_all[n].nodes[1] - sd_all[n].buffer[3],
                     sd_all[n].index_start[2] + sd_all[n].buffer[4]:
                     sd_all[n].index_start[2] + sd_all[n].nodes[2] - sd_all[n].buffer[5]] \
                   = _own_grid

        return grid_out

    return True
    