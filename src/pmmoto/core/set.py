import numpy as np
import dataclasses
from . import _set


@dataclasses.dataclass
class SetSubdomain:
    """
    Subdomain info for set class
    """

    boundary: bool = False
    index: np.array = -1
    inlet: bool = False
    outlet: bool = False
    n_procs: list[int] = dataclasses.field(default_factory=list)


@dataclasses.dataclass
class SetNodes:
    """
    Node info for set class
    """

    nodes: np.array
    index_map: tuple
    num_nodes: int = -1


@dataclasses.dataclass
class SetMatches:
    """
    Subdomain info for set class
    """

    match_lookup: dict
    matches: list[tuple] = dataclasses.field(default_factory=list)


class Set:
    """
    Set class
    """

    def __init__(self, subdomain, local_ID, phase, proc_ID):
        self.subdomain = subdomain
        self.local_ID = local_ID
        self.proc_ID = proc_ID
        self.phase = phase
        self.global_ID = -1
        self.subdomain_data = SetSubdomain()
        self.node_data = None
        self.boundary_data = None

    def set_nodes(self, nodes, grid_shape):
        """
        Assign the nodes to set
        """
        self.node_data = SetNodes(nodes, grid_shape)

    def set_subdomain_info(self, boundary, boundary_index, inlet=False, outlet=False):
        """
        Set subdomain information
        """
        self.subdomain_data = SetSubdomain(
            boundary, np.asarray(boundary_index), inlet, outlet
        )
        self.get_set_neighbors()

    def get_set_neighbors(self):
        """
        Loop through all features and collect neighbor ranks
        This also cleans up subdomain_data.index if boundaries are external/not periodic
        """

        n_procs = self.subdomain_data.n_procs

        # Faces
        for face in self.subdomain.features["faces"]:
            if self.subdomain_data.index[face.feature_id]:
                if face.n_proc < 0:
                    self.subdomain_data.index[face.feature_id] = False
                elif face.n_proc != self.subdomain.rank and face.n_proc not in n_procs:
                    n_procs.append(face.n_proc)

        # Edges
        for edge in self.subdomain.features["edges"]:
            if self.subdomain_data.index[edge.feature_id]:
                if edge.n_proc < 0:
                    self.subdomain_data.index[edge.feature_id] = False
                elif edge.n_proc != self.subdomain.rank and edge.n_proc not in n_procs:
                    n_procs.append(edge.n_proc)

        # Corners
        for corner in self.subdomain.features["corners"]:
            if self.subdomain_data.index[corner.feature_id]:
                if corner.n_proc < 0:
                    self.subdomain_data.index[corner.feature_id] = False
                elif (
                    corner.n_proc != self.subdomain.rank
                    and corner.n_proc not in n_procs
                ):
                    n_procs.append(corner.n_proc)
        self.set_boundary_flag()

    def set_boundary_flag(self):
        """
        If the boundary_index is zero, the set is not on the boundary.
        """
        if np.sum(self.subdomain_data.index) == 0:
            self.subdomain_data.boundary = False

    def set_global_ID(self, global_ID):
        """
        Set the global_ID
        """
        self.global_ID = global_ID


class BoundarySet(Set):
    """
    Boundary Set Class
    """

    def __init__(self, subdomain, local_ID=0, phase=-1, proc_ID=0, boundary_nodes=None):
        super().__init__(subdomain, local_ID, phase, proc_ID)
        self.boundary_data = None
        self.match_data = None
        self.boundary_nodes = boundary_nodes
        self.sort_boundary_nodes()

    def set_boundary_data(self):
        """
        Set the boundary data that is to be sent to neighboring processes
        """
        self.boundary_data = _set.SetDataSend(
            self.local_ID,
            self.proc_ID,
            self.phase,
            self.subdomain_data.inlet,
            self.subdomain_data.outlet,
            self.boundary_nodes,
        )

    def add_internal_nodes(self, nodes):
        """
        Add intenral nodes to SetNodes data class
        """
        self.node_data.nodes = np.append(self.node_data.nodes, nodes)

    def sort_boundary_nodes(self):
        """
        Sort boundary nodes to searing is more efficient
        """
        self.boundary_nodes = np.sort(self.boundary_nodes)

    def match_boundary_sets(self, n_sets):
        """
        Determine which neighboring sets match by comparing boundary nodes global ID
        n_sets are based on feature
        """
        match_lookup = {}
        for feature_index in n_sets:
            for num_set, sset in enumerate(n_sets[feature_index]):
                if not sset:
                    continue
                match_lookup[(sset.proc_ID, sset.local_ID)] = (feature_index, num_set)

        matches = _set._initialize_matches(self.boundary_data)

        for face in self.subdomain.features["faces"]:
            if self.subdomain_data.index[face.feature_id]:
                _set._match_boundary_sets(matches, self.boundary_data, n_sets, face)

        # Check if no matches!

        if not matches.n_ID:
            raise AttributeError(
                "No matching boundary set was found. If using periodic boundary conditions, your grid may not be actually be periodic!"
            )

        self.match_data = SetMatches(match_lookup, matches)

    def get_num_global_nodes(self):
        """
        Determine the total number of nodes for a set. Subdomains are padded so need to remove.
        """

    def update_boundary_set(self, boundary_data):
        """
        Update boundary set information after matching from neighboring procs
        """
        self.boundary_data = boundary_data
        self.global_ID = boundary_data.global_ID
        self.subdomain_data.inlet = boundary_data.inlet
        self.subdomain_data.outlet = boundary_data.outlet
