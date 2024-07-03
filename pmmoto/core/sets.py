import numpy as np
from mpi4py import MPI
import cc3d
import dataclasses
from pmmoto.core import communication
from pmmoto.core import Orientation
from pmmoto.core import _nodes
from pmmoto.core import set
from pmmoto.core import _sets
from pmmoto.core import utils

comm = MPI.COMM_WORLD

__all__ = [
        "connect_all_phases"
    ]

def connect_all_phases(img,inlet,outlet,return_grid = True, return_set = False,save_node_data = True):
    """
    Create sets for all phases in grid
    """
    grid = img.grid
    loop_info = img.loop_info
    subdomain = img.subdomain

    label_grid,label_count =  cc3d.connected_components(grid, return_N = True, out_dtype=np.uint64)
    label_count += 1

    boundary_node_data = _nodes.get_boundary_set_info(
                    subdomain,
                    label_grid,
                    label_count,
                    loop_info,
                    inlet,
                    outlet
                    )
            
    if save_node_data:
        internal_node_data = _nodes.get_internal_set_info(
                                    label_grid,
                                    label_count,
                                    loop_info
                                    )

    # Initialize Sets
    all_sets = Sets(label_count,subdomain)

    # Collect node info and put in sets
    all_sets.collect_boundary_set_info(boundary_node_data,grid,label_grid,label_count)

    # Match sets across process boundaries 
    n_sets = communication.pass_boundary_sets(subdomain,all_sets)
    all_sets.match_boundary_sets(n_sets)


    # Merge matched sets
    all_set_data = comm.gather(
                            {'matched_sets':all_sets.matched_sets,
                            'internal_set_count':all_sets.count.internal}, 
                            root=0)


    single_matched_sets = all_sets.single_merge_matched_sets(all_set_data)

    # Send matched sets
    matched_sets = comm.scatter(single_matched_sets, root=0)

    # Update Local/Global Label Mapping
    local_global_map = all_sets.gen_global_ID_mapping(matched_sets)
    
    # Update Matched Sets
    all_sets.update_boundary_sets(matched_sets)

    output = {}
    if return_grid:
        _nodes.renumber_grid(label_grid,local_global_map)
        output['grid'] = label_grid

    if return_set:

        # Update Set Info
        all_sets.collect_internal_set_info(internal_node_data,grid,label_count)
        all_sets.update_global_ID(local_global_map)
        output['sets'] = all_sets
    
    return output


@dataclasses.dataclass
class SetCount:
    """
    Set Count dataclass
    """
    all: int
    boundary: int = -1
    internal: int = -1

class Sets(object):
    """
    Class for containing all sets
    """
    def __init__(self,
                 set_count = 0,
                 subdomain = None):
        self.count = SetCount(set_count)
        self.subdomain = subdomain
        self.sets = {}
        self.matched_sets = {}
        self.boundary_sets = {}
        self.boundary_set_map = [[] for _ in Orientation.features]

    def collect_boundary_set_info(self,node_data,grid,label_grid,label_count):
        """
        Loop through faces of each subdomain and collect information to merge connected sets
        """

        for label in range(0,label_count):
            if node_data['boundary'][label]:
                boundary_set = set.BoundarySet(
                                    self.subdomain,
                                    label,
                                    self.subdomain.ID,
                                    node_data['boundary_nodes'][label])
                
                index = np.unravel_index(node_data['phase'][label],label_grid.shape)
                boundary_set.set_phase(grid,index)
                boundary_set.set_subdomain_info(
                                node_data['boundary'][label],
                                node_data['boundary_features'][label],
                                node_data['inlets'][label],
                                node_data['outlets'][label]
                                )

                # Ensure boundary set has valid neighboring process
                boundary_set.get_set_neighbors()
                boundary_set.set_nodes(node_data['nodes'][label],grid.shape)
                if boundary_set.subdomain_data.boundary:
                    boundary_set.set_boundary_data()
                    self.boundary_sets[label] = boundary_set
                self.sets[label] = boundary_set

        self.count.boundary = len(self.boundary_sets)
        self.count.internal = self.count.all - self.count.boundary
        self.get_boundary_set_map()

    def collect_internal_set_info(self,node_data,grid,label_count):
        """
        Initilizae Set class for internal sets
        """
        for label in range(0,label_count):
            if node_data['phase'][label] > -1:
                if label in self.sets:
                    self.sets[label].add_internal_nodes(node_data['nodes'][label])
                else:
                    sset = set.Set(
                            self.subdomain,
                            label,
                            self.subdomain.ID
                            )
                    index = np.unravel_index(node_data['phase'][label],grid.shape)
                    sset.set_phase(grid,index)
                    sset.set_nodes(node_data['nodes'][label],grid.shape)
                    self.sets[label] = sset
                
    def get_boundary_set_map(self):
        """
        Populate the list that contains the subdomain features and the set on that feature 
        """
        for local_ID,sset in self.boundary_sets.items():
            for n_feature,feature in enumerate(sset.subdomain_data.index):
                if feature:
                    self.boundary_set_map[n_feature].append(local_ID)

    def match_boundary_sets(self,n_sets):
        """
        Loop through own boundary and neighbor boundary procs and match by boundary nodes
        """
        for local_ID,sset in self.boundary_sets.items():
            sset.match_boundary_sets(n_sets)
            self.matched_sets[local_ID] = sset.match_data.matches

    def get_num_global_nodes(self,n_sets):
        """
        Update the number of global nodes due to double counting the buffer nodes
        """
        for sset in self.boundary_sets:
            sset.get_num_global_nodes(n_sets)

    def single_get_global_ID(self,all_set_data,merged_sets):
        """
        Determine the global IDs for all sets
        """
        internal_set_counts = {}
        for n_proc,proc_match in enumerate(all_set_data):
            internal_set_counts[n_proc] = proc_match['internal_set_count']

        count_ID = np.zeros(self.subdomain.size,dtype=np.int64)
        count_ID[0] = merged_sets
        for n in range(1,self.subdomain.size):
            count_ID[n] = count_ID[n-1] + internal_set_counts[n-1]

        return count_ID

    def single_merge_matched_sets(self,all_set_data):
        """
        Single Process!!
        Take list of matched sets from all processes and merge
        """

        if self.subdomain.ID == 0:
            matches,merged_sets = _sets._single_merge_matched_sets(all_set_data)
            count_ID = self.single_get_global_ID(all_set_data,merged_sets)
           
            # Prep data for scatter
            all_matches = [{'matches':{},'count_ID':0} for _ in range(0,self.subdomain.size)]
            for match in matches:
                all_matches[match.ID[0]]['matches'][match.ID] = match
                all_matches[match.ID[0]]['count_ID'] = count_ID[match.ID[0]]
        else:
            all_matches = None

        return all_matches

    def gen_global_ID_mapping(self,matched_sets):
        """
        Generate the local to global ID mapping for all sets
        """
        local_global_map = -np.ones(self.count.all,dtype=np.int64)
        ID_start = matched_sets['count_ID']
        for local_ID,sset in self.boundary_sets.items():
            key = (self.subdomain.ID,local_ID)
            local_global_map[local_ID] = matched_sets['matches'][key].global_ID
        
        for n in range(0,self.count.all):
            if local_global_map[n] == -1:
                local_global_map[n] = ID_start
                ID_start += 1

        return local_global_map

    def update_global_ID(self,local_global_map):
        """
        Update the global_ID of every set
        """
        for local_ID,sset in self.sets.items():
            sset.set_global_ID(local_global_map[local_ID])


    def update_boundary_sets(self,boundary_data):
        """
        Update the infor on the matched boundary sets
        """
        for key,value in boundary_data['matches'].items():
            self.sets[key[1]].update_boundary_set(value)
            
