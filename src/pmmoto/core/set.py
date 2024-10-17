import numpy as np
import dataclasses
from . import _set


@dataclasses.dataclass
class Subdomain:
    """
    Subdomain info for set class
    """

    boundary: bool = False
    index: np.array = -1
    inlet: bool = False
    outlet: bool = False
    n_procs: list[int] = dataclasses.field(default_factory=list)


@dataclasses.dataclass
class Voxels:
    """
    Node info for set class
    """

    nodes: np.array
    index_map: tuple
    num_nodes: int = -1


class Set:
    """
    A group is a set of voxels. Initilization requires the subdomain information, the local id of the set, and the phase id or the unifer of the voxels.
    """

    def __init__(self, subdomain, local_ID, phase):
        self.subdomain = subdomain
        self.local_ID = local_ID
        self.phase = phase
        self.global_ID = -1
        self.subdomain_data = Subdomain()
        self.data = None
        self.boundary_data = None

    def set_voxels(self, voxels, grid_shape):
        """
        Collect the voxel data
        """
        self.data = Voxels(voxels, grid_shape)

    def set_subdomain_info(self, boundary, boundary_index, inlet=False, outlet=False):
        """
        Set subdomain information
        """
        self.subdomain_data = Subdomain(
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
        for face in self.subdomain.features["faces"].values():
            if self.subdomain_data.index[face.feature_id]:
                if face.n_proc < 0:
                    self.subdomain_data.index[face.feature_id] = False
                elif face.n_proc != self.subdomain.rank and face.n_proc not in n_procs:
                    n_procs.append(face.n_proc)

        # Edges
        for edge in self.subdomain.features["edges"].values():
            if self.subdomain_data.index[edge.feature_id]:
                if edge.n_proc < 0:
                    self.subdomain_data.index[edge.feature_id] = False
                elif edge.n_proc != self.subdomain.rank and edge.n_proc not in n_procs:
                    n_procs.append(edge.n_proc)

        # Corners
        for corner in self.subdomain.features["corners"].values():
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
        self.subdomain_data.boundary = False

    def set_global_ID(self, global_ID):
        """
        Set the global_ID
        """
        self.global_ID = global_ID
