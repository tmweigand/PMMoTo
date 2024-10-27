import numpy as np
import dataclasses
from pmmoto.core import orientation
from pmmoto.core import set
from pmmoto.core import set_boundary

from pmmoto.core import _sets
from pmmoto.core import communication
from pmmoto.core import voxels
from mpi4py import MPI

comm = MPI.COMM_WORLD

__all_ = ["create_sets_and_merge"]


@dataclasses.dataclass
class Count:
    """
    Set count dataclass
    """

    all: int
    boundary: int = -1
    internal: int = -1


class Sets(object):
    """
    Class for containing sets
    """

    def __init__(self, subdomain, set_count=0):
        self.count = Count(set_count)
        self.subdomain = subdomain
        self.sets = {}
        self.matched_sets = {}
        self.boundary_sets = {}
        self.boundary_set_map = [[] for _ in orientation.features]

    def collect_boundary_set_info(self, data, grid, label):
        """
        Loop through each set and determine if a boundary set.
        If so:
            create a boundary set instance

        """
        if data["boundary"][label]:

            boundary_set = set_boundary.BoundarySet(
                subdomain=self.subdomain,
                local_ID=label,
                phase=0,
                boundary_nodes=data["boundary_nodes"][label],
                boundary_features=data["boundary_features"][label],
                inlet=data["inlets"][label],
                outlet=data["outlets"][label],
            )

            boundary_set.set_voxels(data["nodes"][label], grid.shape)
            assert boundary_set.subdomain_data.boundary
            boundary_set.set_boundary_data()
            self.boundary_sets[label] = boundary_set
            self.sets[label] = boundary_set

        self.count.boundary = len(self.boundary_sets)
        self.count.internal = self.count.all - self.count.boundary
        self.get_boundary_set_map()

    def collect_internal_set_info(self, node_data, grid, label_count):
        """
        Initialize Set class for internal sets
        """
        for label in range(0, label_count):
            if node_data["phase"][label] > -1:
                if label in self.sets:
                    self.sets[label].add_internal_nodes(node_data["nodes"][label])
                else:
                    sset = set.Set(self.subdomain, label, self.subdomain.rank)
                    index = np.unravel_index(node_data["phase"][label], grid.shape)
                    sset.set_phase(grid, index)
                    sset.set_voxels(node_data["nodes"][label], grid.shape)
                    self.sets[label] = sset

    def get_boundary_set_map(self):
        """
        Populate the list that contains the subdomain features and the set on that feature
        """
        for local_ID, sset in self.boundary_sets.items():
            for n_feature, feature in enumerate(sset.subdomain_data.index):
                if feature:
                    self.boundary_set_map[n_feature].append(local_ID)

    def match_boundary_sets(self, n_sets):
        """
        Loop through own boundary and neighbor boundary procs and match by boundary nodes
        """
        for local_ID, sset in self.boundary_sets.items():
            sset.match_boundary_sets(n_sets)
            self.matched_sets[local_ID] = sset.match_data.matches

    def get_num_global_nodes(self, n_sets):
        """
        Update the number of global nodes due to double counting the buffer nodes
        """
        for sset in self.boundary_sets:
            sset.get_num_global_nodes(n_sets)

    def single_get_global_ID(self, all_set_data, merged_sets):
        """
        Determine the global IDs for all sets
        """
        internal_set_counts = {}
        for n_proc, proc_match in enumerate(all_set_data):
            internal_set_counts[n_proc] = proc_match["internal_set_count"]

        count_ID = np.zeros(self.subdomain.num_subdomains, dtype=np.int64)
        count_ID[0] = merged_sets
        for n in range(1, self.subdomain.num_subdomains):
            count_ID[n] = count_ID[n - 1] + internal_set_counts[n - 1]

        return count_ID

    def single_merge_matched_sets(self, all_set_data):
        """
        Single Process!!
        Take list of matched sets from all processes and merge
        """

        if self.subdomain.rank == 0:
            matches, merged_sets = _sets._single_merge_matched_sets(all_set_data)
            count_ID = self.single_get_global_ID(all_set_data, merged_sets)

            # Prep data for scatter
            all_matches = [
                {"matches": {}, "count_ID": 0}
                for _ in range(0, self.subdomain.num_subdomains)
            ]
            for match in matches:
                all_matches[match.ID[0]]["matches"][match.ID] = match
                all_matches[match.ID[0]]["count_ID"] = count_ID[match.ID[0]]
        else:
            all_matches = None

        return all_matches

    def gen_global_ID_mapping(self, matched_sets):
        """
        Generate the local to global ID mapping for all sets
        """
        # local_global_map = -np.ones(self.count.all,dtype=np.int64)
        local_global_map = {}
        ID_start = matched_sets["count_ID"]
        for local_ID, sset in self.boundary_sets.items():
            key = (self.subdomain.rank, local_ID)
            local_global_map[local_ID] = matched_sets["matches"][key].global_ID

        for n in range(0, self.count.all):
            if n not in local_global_map:
                local_global_map[n] = ID_start
                ID_start += 1

        return local_global_map

    def update_global_ID(self, local_global_map):
        """
        Update the global_ID of every set
        """
        for local_ID, sset in self.sets.items():
            sset.set_global_ID(local_global_map[local_ID])

    def update_boundary_sets(self, boundary_data):
        """
        Update the infor on the matched boundary sets
        """
        for key, value in boundary_data["matches"].items():
            self.sets[key[1]].update_boundary_set(value)


def create_sets_and_merge(
    img,
    subdomain,
    label_count,
):
    """
    Find the sets on the boundaries of all subdomains and merge them to form one set
    """
    boundary_node_data = voxels.get_boundary_voxels(
        subdomain,
        img,
        label_count,
    )

    # Initialize Sets
    all_sets = Sets(subdomain, label_count)

    # Collect node info and put in sets
    print(boundary_node_data)
    for label in range(label_count):
        all_sets.collect_boundary_set_info(boundary_node_data, img, label)

    # Match sets across process boundaries
    # n_sets = communication.pass_boundary_sets(img.subdomain, all_sets)

    # all_sets.match_boundary_sets(n_sets)

    # # Merge matched sets
    # all_set_data = comm.gather(
    #     {
    #         "matched_sets": all_sets.matched_sets,
    #         "internal_set_count": all_sets.count.internal,
    #     },
    #     root=0,
    # )

    # single_matched_sets = all_sets.single_merge_matched_sets(all_set_data)

    # # Send matched sets
    # matched_sets = comm.scatter(single_matched_sets, root=0)

    # # Update Local/Global Label Mapping
    # local_global_map = all_sets.gen_global_ID_mapping(matched_sets)

    # # Update Matched Sets
    # all_sets.update_boundary_sets(matched_sets)  # ERRROR HERE

    # return all_sets, local_global_map
