"""subdomain_features.py"""

import numpy as np

from . import orientation

__all__ = [
    "collect_features",
    "get_feature_voxels",
    "collect_periodic_features",
]


class Feature(object):
    """Base class for holding features: {face, edge, corner} information for a subdomain.
    This is the main abstraction for handling boundary conditions and parallel communication
    """

    def __init__(self, feature_id, neighbor_rank, boundary_type, global_boundary=None):
        self.feature_id = feature_id
        self.neighbor_rank = neighbor_rank
        self.boundary_type = boundary_type
        self.global_boundary = global_boundary
        self.periodic = False
        self.periodic_correction = (0, 0, 0)
        self.info = None
        self.opp_info = None
        self.extend = [[0, 0], [0, 0], [0, 0]]
        self.loop = None

    def convert_feature_id(self, index=None):
        """Convert the feature if of type (int,int,int) -> int

        Returns:
            int: feature id

        """
        if index is None:
            index = self.feature_id
        return orientation.get_boundary_id(index)


class Face(Feature):
    """Face information for a subdomain
    """

    def __init__(
        self,
        feature_id,
        neighbor_rank,
        boundary_type,
        global_boundary=None,
        inlet=None,
        outlet=None,
    ):
        assert feature_id in orientation.faces
        super().__init__(feature_id, neighbor_rank, boundary_type, global_boundary)
        self.info = orientation.faces[feature_id]
        self.opp_info = orientation.faces[feature_id]["opp"]
        self.global_boundary = self.get_global_boundary(global_boundary)
        self.periodic = self.is_periodic(boundary_type)
        self.inlet = self.is_inlet(inlet)
        self.outlet = self.is_outlet(outlet)
        self.periodic_correction = self.get_periodic_correction()
        self.forward = self.get_direction()
        self.slice = self.get_slice()

    def is_inlet(self, inlet) -> bool:
        """Determine if the face is on the inlet

        Returns:
            bool: True if on inlet

        """
        return inlet

    def is_outlet(self, outlet) -> bool:
        """Determine if the face is on the outlet

        Returns:
            bool: True if on outlet

        """
        return outlet

    def is_periodic(self, boundary_type) -> bool:
        """Determine if the face is a periodic face

        Returns:
            bool: True if periodic

        """
        return boundary_type == "periodic"

    def get_periodic_correction(self) -> tuple[int, ...]:
        """Determine spatial correction factor if periodic
        """
        _period_correction = [0, 0, 0]
        if self.periodic:
            _period_correction[self.info["argOrder"][0]] = self.info["dir"]

        return tuple(_period_correction)

    def get_global_boundary(self, global_boundary) -> bool:
        """Determine if the face is an external boundary
        """
        return global_boundary

    def get_strides(self, strides):
        """_summary_

        Args:
            strides (tuple): The output from a NumPy array .strides

        Returns:
            _type_: _description_

        """
        for feature_id, stride in zip(self.feature_id, strides):
            if feature_id != 0:
                return stride

    def get_direction(self):
        """Determine if the face is point in the forward direction:
            -1 for the non-negative feature id

        Returns:
            _type_: _description_

        """
        forward = True
        for feature_id in self.feature_id:
            if feature_id == 1:
                forward = False
        return forward

    def index_to_entry(self, entry, value):
        """Use the feature id to set a value of a list:
        ((n,n),(n,n),(n,n))
        """
        for dim, f_id in enumerate(self.feature_id):
            if f_id > 0:
                entry[dim][1] = value
            elif f_id < 0:
                entry[dim][0] = value

        return entry

    def get_slice(self):
        """For some situations, like with wall boundary types, it is convenient to operate
        on the entire slice of the img as opposed to being broken down by edges and corner.

        Returns:
            list[slices]

        """
        face_slice = [slice(None) for _ in range(3)]
        for dim, f_id in enumerate(self.feature_id):
            if f_id == -1:
                face_slice[dim] = 0
            elif f_id == 1:
                face_slice[dim] = -1

        return tuple(face_slice)


class Edge(Feature):
    """Edge information for a subdomain
    Need to distinguish between internal and external edges.
    There are 12 external corners. All others are termed internal
    """

    def __init__(
        self,
        feature_id,
        neighbor_rank,
        boundary_type,
        global_boundary=None,
    ):
        assert feature_id in orientation.edges
        super().__init__(feature_id, neighbor_rank, boundary_type, global_boundary)
        self.info = orientation.edges[feature_id]
        self.periodic = self.is_periodic(boundary_type)
        self.opp_info = orientation.edges[feature_id]["opp"]
        self.global_boundary = self.get_global_boundary(global_boundary)
        self.periodic_correction = self.get_periodic_correction()

    def is_periodic(self, boundary_type) -> bool:
        """Determine if an edge is a periodic edge

        Returns:
            bool: True if periodic

        """
        return boundary_type == "periodic"

    def get_periodic_correction(self) -> tuple[int, ...]:
        """Determine spatial correction factor if periodic
        """
        _period_correction = [0, 0, 0]
        for face in self.info["faces"]:
            if self.boundary_type == "periodic":
                _period_correction[orientation.faces[face]["argOrder"][0]] = (
                    orientation.faces[face]["dir"]
                )

        return tuple(_period_correction)

    def get_global_boundary(self, global_boundary) -> bool:
        """Determine if the edge is an external boundary
        """
        return global_boundary


class Corner(Feature):
    """Corner information for a subdomain.
    Need to distinguish between internal and external corners.
    There are 8 external corners. All others are termed internal
    """

    def __init__(
        self,
        feature_id,
        neighbor_rank,
        boundary_type,
        global_boundary=None,
    ):
        super().__init__(feature_id, neighbor_rank, boundary_type, global_boundary)
        self.info = orientation.corners[feature_id]
        self.periodic = self.is_periodic(boundary_type)
        self.opp_info = orientation.corners[feature_id]["opp"]
        self.periodic_correction = self.get_periodic_correction()
        self.global_boundary = self.get_global_boundary(global_boundary)

    def is_periodic(self, boundary_type) -> bool:
        """Determine if a corner is a periodic corner

        Returns:
            bool: True if periodic

        """
        return boundary_type == "periodic"

    def get_periodic_correction(self) -> tuple[int, ...]:
        """Determine spatial correction factor (shift) if periodic
        """
        _period_correction = [0, 0, 0]
        for n_face in self.info["faces"]:
            if self.boundary_type == "periodic":
                _period_correction[orientation.faces[n_face]["argOrder"][0]] = (
                    orientation.faces[n_face]["dir"]
                )

        return tuple(_period_correction)

    def get_global_boundary(self, global_boundary) -> bool:
        """Determine if the corner is a global boundary
        """
        return global_boundary


def collect_features(
    neighbor_ranks,
    global_boundary,
    boundary_types,
    voxels,
    inlet=None,
    outlet=None,
    pad=None,
    reservoir_pad=None,
):
    """Collect information for faces, edges, and corners
    """
    faces = {}
    edges = {}
    corners = {}

    if inlet is None:
        inlet = (False, False, False, False, False, False)
    if outlet is None:
        outlet = (False, False, False, False, False, False)

    ### Faces
    for feature in orientation.faces.keys():
        face_dim = np.nonzero(feature)[0][0]
        index = face_dim * 2 if feature[face_dim] < 0 else face_dim * 2 + 1
        faces[feature] = Face(
            feature_id=feature,
            neighbor_rank=neighbor_ranks[feature],
            boundary_type=boundary_types[feature],
            global_boundary=global_boundary[feature],
            inlet=inlet[index],
            outlet=outlet[index],
        )
        faces[feature].loop = get_feature_voxels(
            feature, voxels, boundary_type=boundary_types[feature], pad=pad
        )

    ### Edges
    for feature in orientation.edges.keys():
        edges[feature] = Edge(
            feature_id=feature,
            neighbor_rank=neighbor_ranks[feature],
            boundary_type=boundary_types[feature],
            global_boundary=global_boundary[feature],
        )
        edges[feature].loop = get_feature_voxels(
            feature, voxels, boundary_type=boundary_types[feature], pad=pad
        )

    ### Corners
    for feature in orientation.corners.keys():
        corners[feature] = Corner(
            feature_id=feature,
            neighbor_rank=neighbor_ranks[feature],
            boundary_type=boundary_types[feature],
            global_boundary=global_boundary[feature],
        )
        corners[feature].loop = get_feature_voxels(
            feature, voxels, boundary_type=boundary_types[feature], pad=pad
        )

    data_out = {"faces": faces, "edges": edges, "corners": corners}

    return data_out


def get_feature_voxels(feature_id, voxels, boundary_type=None, pad=None):
    """Compute feature-specific (i.e face, edge, corner) voxel ranges with optional padding.

    This function generates voxel ranges for a given feature based on the `feature_id`,
    `voxels`, and an optional `pad` parameter. It returns a dictionary with ranges for
    "own" voxels and, if applicable, "neighbor" voxels.

    Args:
        feature_id (list[int]): A list of feature identifiers, where each value
                                specifies the direction or category (-1, 0, or 1).
        voxels (list[int]): A list of integers representing the lengths of the voxel
                            dimensions along each axis.
        pad (list[list[int]], optional): A 2D list specifying the padding for each axis.
                                         Each inner list contains two integers:
                                         [padding_start, padding_end]. Defaults to None.

    Returns:
        dict: A dictionary with keys:
              - "own": A NumPy array of shape (3, 2) containing voxel ranges for the
                feature's own voxels along each axis.
              - "neighbor" (optional): A NumPy array of shape (3, 2) containing voxel ranges
                for neighboring voxels along each axis if padding is specified.

    Notes:
        - If `pad` is not provided or contains only zeros, the function assumes no padding.
        - The `feature_id` determines how voxel ranges are computed for each axis:
          - -1: Specifies the start of the axis.
          - 0: Includes the entire range (or adjusted range if padding exists).
          - 1: Specifies the end of the axis.
        - For wall boundary conditions, the walls are stores as neighbor.

    """
    # Determine if padding is active
    padded = pad is not None and np.any(pad)

    # Default pad values
    if not padded:
        pad = np.zeros((len(voxels), 2), dtype=int)

    # Initialize loop dictionary
    loop = {"own": np.zeros((len(voxels), 2), dtype=np.uint64)}
    if padded:
        loop["neighbor"] = np.zeros((len(voxels), 2), dtype=np.uint64)

    for n, length in enumerate(voxels):

        if feature_id[n] == -1:
            lower_pad = pad[n][0]
            if lower_pad != 0:
                if boundary_type == "wall":
                    loop["own"][n] = [1, 2]
                else:
                    loop["own"][n] = [lower_pad, lower_pad * 2]
            else:
                loop["own"][n] = [0, 1]
            if "neighbor" in loop:
                if boundary_type == "wall":
                    loop["neighbor"][n] = [0, 1]
                else:
                    loop["neighbor"][n] = [0, lower_pad]

        elif feature_id[n] == 1:
            upper_pad = pad[n][1]
            if upper_pad != 0:
                if boundary_type == "wall":
                    loop["own"][n] = [length - 2, length - 1]
                else:
                    loop["own"][n] = [length - upper_pad * 2, length - upper_pad]
            else:
                loop["own"][n] = [length - 1, length]
            if "neighbor" in loop:
                if boundary_type == "wall":
                    loop["neighbor"][n] = [length - 1, length]
                else:
                    loop["neighbor"][n] = [length - upper_pad, length]

        else:
            if padded:
                loop["own"][n] = [pad[n][0], length - pad[n][1]]
            else:
                loop["own"][n] = [0, length]
            if "neighbor" in loop:
                loop["neighbor"][n] = [pad[n][0], length - pad[n][1]]

    return loop


def collect_periodic_features(features):
    """Loop through features and collect periodic ones

    Args:
        features (_type_): _description_

    """
    periodic_features = []
    feature_types = ["faces", "edges", "corners"]
    for feature_type in feature_types:
        for feature_id, feature in features[feature_type].items():
            if feature.periodic:
                periodic_features.append(feature_id)

    return periodic_features


def collect_periodic_corrections(features):
    """Loop through features and collect periodic ones

    Args:
        features (_type_): _description_

    """
    periodic_corrections = {}
    feature_types = ["faces", "edges", "corners"]
    for feature_type in feature_types:
        for feature_id, feature in features[feature_type].items():
            if feature.periodic:
                periodic_corrections[feature_id] = feature.periodic_correction

    return periodic_corrections
