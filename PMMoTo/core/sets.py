import numpy as np
from mpi4py import MPI
import cc3d
from . import communication
from . import nodes
from . import _nodes
from . import set
from . import _sets

comm = MPI.COMM_WORLD

__all__ = [
        "collect_sets",
        "connect_all_phases"
    ]

def _get_phase_index(grid,index):
    """
    Get the phase of a set
    """
    return grid[index[0],index[1],index[2]]

def initialize_sets(subdomain,labels):
    """
    Initialize the Set and Sets Classes
    """
    _s = []
    for label in range(0,labels):
        _s.append(set.Set(subdomain = subdomain,
                          local_ID = label,
                          proc_ID = subdomain.ID))
        
    Sets = _sets.Sets(_s,labels,subdomain)

    return Sets

def connect_all_phases(img,inlet,outlet):
    """
    Create sets for all phases in grid
    """
    grid = img.grid
    loop_info = img.loop_info
    subdomain = img.subdomain
    
    label_grid,n_labels =  cc3d.connected_components(grid,return_N = True,out_dtype=np.uint64)
    n_labels += 1
    # Initialize Sets
    Sets = initialize_sets(subdomain,n_labels)
        
    # Grab Set info from nodes
    set_nodes,b_nodes,boundarys,features,inlets,outlets = _nodes.get_set_info(subdomain,label_grid,
                                   n_labels,
                                   loop_info,
                                   inlet,
                                   outlet)

    for label in range(0,n_labels):
        index = np.unravel_index(set_nodes[label][0],grid.shape)
        phase = _get_phase_index(grid,index)
        Sets.sets[label].set_phase(phase)
        Sets.sets[label].set_subdomain_info(boundarys[label],
                                       features[label],
                                       inlets[label],
                                       outlets[label])
        if boundarys[label]:
            Sets.sets[label].set_nodes(grid.shape,set_nodes[label],b_nodes[label])
        else:
            Sets.sets[label].set_nodes(grid.shape,set_nodes[label])


    Sets.get_boundary_sets()

    return Sets


def collect_sets(grid,phase,inlet,outlet,loopInfo,subdomain):
    """
    Create sets for specified phase (phase) in grid
    """
    rank = subdomain.ID
    size = subdomain.size

    Nodes  = nodes.get_node_info(grid,phase,inlet,outlet,subdomain.domain,loopInfo,subdomain)
    #nodes_new = _nodes._get_node_info(grid,phase,inlet,outlet,subdomain.domain,loopInfo,subdomain)
    Sets = nodes.get_connected_sets(subdomain,grid,phase,Nodes)


    # if size > 1:

    #     ### Grab boundary sets and send to neighboring procs
    #     Sets.get_boundary_sets()
    #     send_boundary_data = Sets.pack_boundary_data()
    #     recv_boundary_data = communication.set_COMM(subdomain,send_boundary_data)
    #     n_boundary_data = Sets.unpack_boundary_data(recv_boundary_data)

    #     ### Match boundary sets from neighboring procs and send to root for global ID generation
    #     all_matches = Sets.match_boundary_sets(n_boundary_data)
    #     Sets.get_num_global_nodes(all_matches,n_boundary_data)
    #     send_matched_set_data = Sets.pack_matched_sets(all_matches,n_boundary_data)
    #     recv_matched_set_data = comm.gather(send_matched_set_data, root=0)

    #     ### Connect sets that are not direct neighbors 
    #     if subdomain.ID == 0:
    #         all_matched_sets,index_convert = Sets.unpack_matched_sets(recv_matched_set_data)
    #         all_matched_sets,total_boundary_sets = Sets.organize_matched_sets(all_matched_sets,index_convert)
    #         all_matched_sets = Sets.repack_matched_sets(all_matched_sets)
    #     else:
    #         all_matched_sets = None
    #         global_matched_sets = comm.scatter(all_matched_sets, root=0)

    #     ### Generate and Update global ID information
    #     global_ID_data = comm.gather([Sets.setCount,Sets.boundarySetCount], root=0)
    #     if subdomain.ID == 0:
    #         local_set_ID_start = Sets.organize_global_ID(global_ID_data,total_boundary_sets)
    #     else:
    #         local_set_ID_start = None
    #     Sets.local_set_ID_start = comm.scatter(local_set_ID_start, root=0)

    #     ### Update IDs
    #     Sets.update_globalSetID(global_matched_sets)

    return Sets