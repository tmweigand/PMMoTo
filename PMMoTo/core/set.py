import numpy as np
import dataclasses
from . import _Orientation
from . import Orientation


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

    def get_set_neighbors(self):
        """
        Loop through all features and collect neighbor proccess
        This also cleans up subdomain_data.index if boundaries are external/not periodic
         """
        
        # Faces
        for face in self.subdomain.faces:
            if self.subdomain_data.index[face.feature_ID]:
                if face.n_proc < 0:
                    self.subdomain_data.index[face.feature_ID] = False
                elif face.n_proc != self.subdomain.ID:
                    if face.n_proc not in self.subdomain_data.n_procs:
                        self.subdomain_data.n_procs.append(face.n_proc)

        # Edges
        for edge in self.subdomain.edges:
            if self.subdomain_data.index[edge.feature_ID]:
                if edge.n_proc < 0:
                    self.subdomain_data.index[edge.feature_ID] = False
                elif edge.n_proc != self.subdomain.ID:
                    if edge.n_proc not in self.subdomain_data.n_procs:
                        self.subdomain_data.n_procs.append(edge.n_proc)
        
        # Corners
        for corner in self.subdomain.corners:
            if self.subdomain_data.index[corner.feature_ID]:
                if corner.n_proc < 0:
                    self.subdomain_data.index[corner.feature_ID] = False
                elif corner.n_proc != self.subdomain.ID:
                    if corner.n_proc not in self.subdomain_data.n_procs:
                        self.subdomain_data.n_procs.append(corner.n_proc)