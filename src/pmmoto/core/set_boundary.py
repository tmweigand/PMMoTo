import numpy as np
import dataclasses
from . import set
from . import _set


@dataclasses.dataclass
class Matches:
    """
    Subdomain info for set class
    """

    match_lookup: dict
    matches: list[tuple] = dataclasses.field(default_factory=list)


class BoundarySet(set.Set):
    """
    Boundary Set Class
    """

    def __init__(
        self,
        subdomain,
        local_ID,
        phase,
        boundary_nodes,
        boundary_features,
        inlet=False,
        outlet=False,
    ):
        super().__init__(subdomain, local_ID, phase)
        self.boundary_data = None
        self.match_data = None
        self.boundary_nodes = boundary_nodes
        self.subdomain_data = set.Subdomain(True, boundary_features, inlet, outlet)
        self.sort_boundary_nodes()
        self.get_set_neighbors()

    def set_boundary_data(self):
        """
        Set the boundary data that is to be sent to neighboring processes
        """
        self.boundary_data = _set.SetDataSend(
            self.local_ID,
            self.subdomain.rank,
            self.phase,
            self.subdomain_data.inlet,
            self.subdomain_data.outlet,
            self.boundary_nodes,
        )
        self.sort_boundary_nodes()

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
