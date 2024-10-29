# cython: profile=True
# cython: linetrace=True
import numpy as np
cimport numpy as cnp
cimport cython
import dataclasses
from libcpp.vector cimport vector
from numpy cimport npy_intp
from libcpp cimport bool

@dataclasses.dataclass
cdef class SetMatch:
    """
    SetMatch Data Class:
        ID: (set proc ID, set local ID)
        n_ID: (matched set proc ID, matched set local ID)
        inlet: determine if set or matched set are inlet sets
        outlet: determine if set or matched set are outlet sets
        global_ID: global Set ID
        visited: bool True, if merged
    """
    ID: tuple(npy_intp,npy_intp) 
    n_ID: list[(int,int)] = dataclasses.field(default_factory=list)
    inlet: bool = False
    outlet: bool = False
    global_ID: npy_intp = 0
    visited: bool = False

@dataclasses.dataclass
cdef class SetDataSend:
    local_ID: npy_intp
    proc_ID: npy_intp
    phase: npy_intp
    inlet: bool
    outlet: bool
    boundary_nodes: vector[npy_intp]


def _initialize_matches(set):
    """
    Initialize the SetMatch Class
    """
    match = SetMatch((set.proc_ID,set.local_ID))
    return match

def _match_boundary_sets(match,set,n_sets,face):
    """
    Match boundary sets based on boundary node global ID
    """
    pass
    # for n_set in n_sets[face.info['ID']]:
    #     if n_set.phase == set.phase:
    #         if match_boundary_nodes(n_set.boundary_nodes,set.boundary_nodes):
    #             n_ID = (n_set.proc_ID,n_set.local_ID)
    #             if n_ID not in match.n_ID:
    #                 match.n_ID.append((n_set.proc_ID,n_set.local_ID))
    #             if n_set.inlet or set.inlet:
    #                 match.inlet = True
    #             if n_set.outlet or set.outlet:
    #                 match.outlet = True
    # return match

def _get_num_global_nodes(set_boundary_nodes,n_set_boundary_nodes):
    """
    Count the number of shared boundary nodes between two sets
    """
    pass
    # return count_matched_nodes(set_boundary_nodes,n_set_boundary_nodes)