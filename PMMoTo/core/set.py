import numpy as np
import dataclasses
from pmmoto.core import _set

@dataclasses.dataclass
class SetSubdomain:
    """
    Subdomain info for set class
    """
    boundary: bool
    index: np.array
    inlet: bool
    outlet: bool
    n_procs: list[int] = dataclasses.field(default_factory=list)

@dataclasses.dataclass
class SetNodes:
    """
    Node info for set class
    """
    nodes: np.array
    boundary_nodes: np.array
    num_nodes: int
    num_boundary_nodes: int
    index_map: tuple


# @dataclasses.dataclass
# class SetDataSend:
#     """
#     Data to send to neighboring procs
#     """
#     local_ID: int
#     proc_ID: int
#     phase: int
#     num_nodes: int
#     inlet: bool
#     outlet: bool
#     boundary_nodes: np.array

class Set:
    """
    Set class
    """
    def __init__(self,
                 subdomain,
                 local_ID = 0,
                 proc_ID = 0):
        self.subdomain = subdomain
        self.local_ID = local_ID
        self.proc_ID = proc_ID
        self.global_ID = -1
        self.phase = None
        self.subdomain_data = None
        self.node_data = None
        self.boundary_data = None

    def set_phase(self,phase):
        """
        Assign the grid phase to the set
        """
        self.phase = phase

    def set_nodes(self,shape,nodes,boundary_nodes = None):
        """
        Assign the nodes to set
        """
        if boundary_nodes is not None:
            self.node_data = SetNodes(nodes,boundary_nodes,len(nodes),len(boundary_nodes),shape)
        else:
            self.node_data = SetNodes(nodes,None,len(nodes),0,shape)

    def set_subdomain_info(self,boundary,boundary_index,inlet = False,outlet = False):
        """
        Set subdomain information
        """
        self.subdomain_data = SetSubdomain(boundary,np.asarray(boundary_index),inlet,outlet)

    def collect_boundary_data(self):
        """
        Collect all the data to send to neighboring procs
        """
        self.node_data.boundary_nodes = np.sort(self.node_data.boundary_nodes)
        self.boundary_data = _set.SetDataSend(self.local_ID,
                                         self.proc_ID,
                                         self.phase,
                                         self.node_data.num_nodes,
                                         self.subdomain_data.inlet,
                                         self.subdomain_data.outlet,
                                         self.node_data.boundary_nodes)

    def get_set_neighbors(self):
        """
        Loop through all features and collect neighbor proccess
        This also cleans up subdomain_data.index if boundaries are external/not periodic
         """

        n_procs = self.subdomain_data.n_procs

        # Faces
        for face in self.subdomain.faces:
            if self.subdomain_data.index[face.feature_ID]:
                if face.n_proc < 0:
                    self.subdomain_data.index[face.feature_ID] = False
                elif face.n_proc != self.subdomain.ID and face.n_proc not in n_procs:
                    n_procs.append(face.n_proc)

        # Edges
        for edge in self.subdomain.edges:
            if self.subdomain_data.index[edge.feature_ID]:
                if edge.n_proc < 0:
                    self.subdomain_data.index[edge.feature_ID] = False
                elif edge.n_proc != self.subdomain.ID and edge.n_proc not in n_procs:
                    n_procs.append(edge.n_proc)

        # Corners
        for corner in self.subdomain.corners:
            if self.subdomain_data.index[corner.feature_ID]:
                if corner.n_proc < 0:
                    self.subdomain_data.index[corner.feature_ID] = False
                elif corner.n_proc != self.subdomain.ID and corner.n_proc not in n_procs:
                    n_procs.append(corner.n_proc)

        self.set_boundary_flag()

    def set_boundary_flag(self):
        """
        If the boundary_index is zero, the set is not on the boundary.
        """
        if np.sum(self.subdomain_data.index) == 0:
            self.subdomain_data.boundary = False

    def match_boundary_sets(self,n_sets):
        """
        Determine which neigboring sets match by comnpairing boundary nodes global ID
        n_sets are based on feature
        """
        all_matches = []
        for face in self.subdomain.faces:
            if self.subdomain_data.index[face.feature_ID]:
                matches = _set._match_boundary_sets(self.boundary_data,n_sets,face)
                for m in matches:
                    if m not in all_matches:
                        all_matches.append(m)
        
        
        print(all_matches)