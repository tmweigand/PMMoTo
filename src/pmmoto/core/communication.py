"""communication.py"""

# from . import orientation

import numpy as np
from mpi4py import MPI

__all__ = [
    "all_gather",
    "gather",
    "update_buffer",
    # "generate_halo",
    # "pass_external_data",
    # "pass_boundary_sets",
    "communicate_features",
]


comm = MPI.COMM_WORLD


def all_gather(data):
    """_summary_

    Args:
        data (_type_): _description_
    """
    all_data = comm.allgather(data)

    return all_data


def gather(data):
    """_summary_

    Args:
        data (_type_): _description_
    """
    all_data = comm.gather(data, root=0)

    return all_data


def update_buffer(subdomain, grid):
    """
    Organize the communication to update the padding/buffer on subdomains and account for periodic boundary conditions.

    Args:
        subdomain (object): The subdomain object containing features and neighbor information.
        grid (numpy.ndarray): The grid data to be updated.

    Returns:
        numpy.ndarray: The updated grid with the buffer data.
    """
    send_data = buffer_pack(subdomain, grid)
    recv_data = communicate_features(subdomain, send_data)
    grid = buffer_unpack(subdomain, grid, recv_data)

    return grid


# def generate_halo(subdomain, grid, halo):
#     """ """
#     halo_data, halo_out = halo_pack(subdomain, grid, halo)
#     f, e, c = communicate(subdomain, halo_data)
#     halo_grid = halo_unpack(subdomain, grid, halo_out, f, e, c)
#     return halo_grid, halo_out


# def pass_external_data(domain, subdomain, face_solids, edge_solids, corner_solids):
#     """ """
#     external_data = external_nodes_pack(
#         subdomain, face_solids, edge_solids, corner_solids
#     )
#     f, e, c = communicate(subdomain, external_data)
#     external_solids = external_nodes_unpack(
#         domain, subdomain, f, e, c, face_solids, edge_solids, corner_solids
#     )
#     return external_solids


# def pass_boundary_sets(subdomain, sets):
#     """
#     Huh?
#     """
#     send_boundary_sets = boundary_set_pack(subdomain, sets)
#     f, e, c = communicate(subdomain, send_boundary_sets)
#     recv_boundary_sets = boundary_set_unpack(subdomain, sets, f, e, c)
#     return recv_boundary_sets


def buffer_pack(subdomain, grid):
    """
    Packs the buffer data for communication.

    Args:
        subdomain (object): The subdomain object containing features.
        grid (numpy.ndarray): The grid data to be packed.

    Returns:
        dict: A dictionary containing the packed buffer data.
    """

    buffer_data = {}
    feature_types = ["faces", "edges", "corners"]
    for feature_type in feature_types:
        for feature_id, feature in subdomain.features[feature_type].items():
            if feature.neighbor_rank > -1:
                buffer_data[feature_id] = grid[
                    feature.loop["own"][0][0] : feature.loop["own"][0][1],
                    feature.loop["own"][1][0] : feature.loop["own"][1][1],
                    feature.loop["own"][2][0] : feature.loop["own"][2][1],
                ]

    return buffer_data


def buffer_unpack(subdomain, grid, features_recv):
    """
    Unpack buffer information and account for serial periodic boundary conditions
    """

    feature_types = ["faces", "edges", "corners"]
    for feature_type in feature_types:
        for feature_id, feature in subdomain.features[feature_type].items():
            if feature_id in features_recv:
                grid[
                    feature.loop["neighbor"][0][0] : feature.loop["neighbor"][0][1],
                    feature.loop["neighbor"][1][0] : feature.loop["neighbor"][1][1],
                    feature.loop["neighbor"][2][0] : feature.loop["neighbor"][2][1],
                ] = features_recv[feature_id]

    return grid


# def halo_pack(subdomain, grid, halo):
#     """
#     Grab The Slices (based on Size) to pack and send to Neighbors
#     for faces, edges and corners
#     """

#     pad = np.zeros([subdomain.dims * 2], dtype=int)
#     for n in range(subdomain.dims):
#         pad[n * 2] = subdomain.pad[n][0]
#         pad[n * 2 + 1] = subdomain.pad[n][1]

#     send_faces, send_edges, send_corners = orientation.get_send_halo(
#         halo, pad, grid.shape
#     )

#     halo_out = np.zeros(orientation.num_faces, dtype=int)
#     for face in subdomain.features["faces"]:
#         if face.n_proc > -1:
#             halo_out[face.ID] = halo[face.ID]

#     halo_data = {}
#     for feature, n_proc in subdomain.neighbor_ranks.items():
#         if n_proc > -1 and n_proc != subdomain.rank:
#             halo_data[n_proc] = {"ID": {}}
#             halo_data[n_proc]["ID"][feature] = None

#     slices = send_faces
#     for face in subdomain.features["faces"]:
#         if face.n_proc > -1 and face.n_proc != subdomain.rank:
#             s = slices[face.ID, :]
#             halo_data[face.n_proc]["ID"][face.info["ID"]] = grid[s[0], s[1], s[2]]

#     slices = send_edges
#     for edge in subdomain.features["edges"]:
#         if edge.n_proc > -1 and edge.n_proc != subdomain.rank:
#             s = slices[edge.ID, :]
#             halo_data[edge.n_proc]["ID"][edge.info["ID"]] = grid[s[0], s[1], s[2]]

#     slices = send_corners
#     for corner in subdomain.features["corners"]:
#         if corner.n_proc > -1 and corner.n_proc != subdomain.rank:
#             s = slices[corner.ID, :]
#             halo_data[corner.n_proc]["ID"][corner.info["ID"]] = grid[s[0], s[1], s[2]]

#     return halo_data, halo_out


# def halo_unpack(subdomain, grid, halo, face_recv, edge_recv, corner_recv):
#     """ """
#     if all(halo == 0):
#         halo_grid = grid
#     else:
#         halo_grid = np.pad(
#             grid,
#             ((halo[0], halo[1]), (halo[2], halo[3]), (halo[4], halo[5])),
#             "constant",
#             constant_values=255,
#         )

#         pad = np.zeros([subdomain.dims * 2], dtype=int)
#         for n in range(subdomain.dims):
#             pad[n * 2] = subdomain.pad[n][0]
#             pad[n * 2 + 1] = subdomain.pad[n][1]

#         recv_faces, recv_edges, recv_corners = orientation.get_recv_halo(
#             halo, pad, halo_grid.shape
#         )
#         send_faces, send_edges, send_corners = orientation.get_send_halo(
#             halo, pad, grid.shape
#         )

#         #### Faces ####
#         r_slices = recv_faces
#         s_slices = send_faces
#         for face in subdomain.features["faces"]:
#             if face.n_proc > -1 and face.n_proc != subdomain.rank:
#                 r_s = r_slices[face.ID, :]
#                 halo_grid[r_s[0], r_s[1], r_s[2]] = face_recv[face.ID]["ID"][
#                     face.opp_info["ID"]
#                 ]
#             elif face.n_proc == subdomain.rank:
#                 r_s = r_slices[face.ID, :]
#                 s_s = s_slices[face.info["oppIndex"], :]
#                 halo_grid[r_s[0], r_s[1], r_s[2]] = grid[s_s[0], s_s[1], s_s[2]]

#         #### Edges ####
#         r_slices = recv_edges
#         s_slices = send_edges
#         for edge in subdomain.features["edges"]:
#             if edge.n_proc > -1 and edge.n_proc != subdomain.rank:
#                 r_s = r_slices[edge.ID, :]
#                 halo_grid[r_s[0], r_s[1], r_s[2]] = edge_recv[edge.ID]["ID"][
#                     edge.opp_info["ID"]
#                 ]
#             elif edge.n_proc == subdomain.rank:
#                 r_s = r_slices[edge.ID, :]
#                 s_s = s_slices[edge.info["oppIndex"], :]
#                 halo_grid[r_s[0], r_s[1], r_s[2]] = grid[s_s[0], s_s[1], s_s[2]]

#         #### Corners ####
#         r_slices = recv_corners
#         s_slices = send_corners
#         for corner in subdomain.features["corners"]:
#             if corner.n_proc > -1 and corner.n_proc != subdomain.rank:
#                 r_s = r_slices[corner.ID, :]
#                 halo_grid[r_s[0], r_s[1], r_s[2]] = corner_recv[corner.ID]["ID"][
#                     corner.opp_info["ID"]
#                 ]
#             elif corner.n_proc == subdomain.rank:
#                 r_s = r_slices[corner.ID, :]
#                 s_s = s_slices[corner.info["oppIndex"], :]
#                 halo_grid[r_s[0], r_s[1], r_s[2]] = grid[s_s[0], s_s[1], s_s[2]]

#     return halo_grid


def external_nodes_pack(subdomain, face_solids, edge_solids, corner_solids):
    """
    Send boundary solids data.
    external_data[neighbor_proc_ID]['ID'][(0,0,0)] = boundary_nodes[feature.ID]
    """
    external_data = {}
    for feature, n_proc in subdomain.neighbor_ranks.items():
        if n_proc != subdomain.rank:
            external_data[n_proc] = {"ID": {}}
            external_data[n_proc]["ID"][feature] = None

    for face in subdomain.features["faces"]:
        if face.n_proc != subdomain.rank:
            external_data[face.n_proc]["ID"][face.info["ID"]] = face_solids[face.ID]

    for edge in subdomain.features["edges"]:
        if edge.n_proc != subdomain.rank:
            external_data[edge.n_proc]["ID"][edge.info["ID"]] = edge_solids[edge.ID]

    for corner in subdomain.features["corners"]:
        if corner.n_proc != subdomain.rank:
            external_data[corner.n_proc]["ID"][corner.info["ID"]] = corner_solids[
                corner.ID
            ]

    return external_data


# def external_nodes_unpack(
#     domain,
#     subdomain,
#     face_recv,
#     edge_recv,
#     corner_recv,
#     face_solids,
#     edge_solids,
#     corner_solids,
# ):
#     """
#     Recieve boundary solids data.
#     """

#     external_solids = {key: None for key in orientation.features}

#     #### FACE ####
#     for face in subdomain.features["faces"]:

#         period_correction = []
#         for n in range(subdomain.dims):
#             period_correction.append(face.periodic_correction[n] * domain.length[n])

#         if face.n_proc > -1 and face.n_proc != subdomain.rank:
#             opp_ID = face.opp_info["ID"]
#             external_solids[face.info["ID"]] = (
#                 face_recv[face.ID]["ID"][opp_ID] - period_correction
#             )
#         elif face.n_proc == subdomain.rank:
#             external_solids[face.info["ID"]] = (
#                 face_solids[face.info["oppIndex"]] - period_correction
#             )

#     #### EDGE ####
#     for edge in subdomain.features["edges"]:

#         period_correction = []
#         for n in range(subdomain.dims):
#             period_correction.append(edge.periodic_correction[n] * domain.length[n])

#         if edge.n_proc > -1 and edge.n_proc != subdomain.rank:
#             opp_ID = edge.opp_info["ID"]
#             external_solids[edge.info["ID"]] = (
#                 edge_recv[edge.ID]["ID"][opp_ID] - period_correction
#             )
#         elif edge.n_proc == subdomain.rank:
#             external_solids[edge.info["ID"]] = (
#                 edge_solids[edge.info["oppIndex"]] - period_correction
#             )

#     #### Corner ####
#     for corner in subdomain.features["corners"]:

#         period_correction = []
#         for n in range(subdomain.dims):
#             period_correction.append(corner.periodic_correction[n] * domain.length[n])

#         if corner.n_proc > -1 and corner.n_proc != subdomain.rank:
#             opp_ID = corner.opp_info["ID"]
#             external_solids[corner.info["ID"]] = (
#                 corner_recv[corner.ID]["ID"][opp_ID] - period_correction
#             )
#         elif corner.n_proc == subdomain.rank:
#             external_solids[corner.info["ID"]] = (
#                 corner_solids[corner.info["oppIndex"]] - period_correction
#             )

#     return external_solids


def boundary_set_pack(subdomain, sets):
    """
    TO write...
    """

    # Initialize send_boundary_data
    send_boundary_data = {}
    for feature, n_proc in subdomain.neighbor_ranks.items():
        if n_proc != subdomain.rank:
            if n_proc not in send_boundary_data:
                send_boundary_data[n_proc] = {"ID": {}}
            send_boundary_data[n_proc]["ID"][feature] = []

    # Loop through faces

    for face in subdomain.features["faces"]:
        if face.n_proc > -1 and face.n_proc != subdomain.rank:
            for s in sets.boundary_set_map[face.feature_id]:
                send_boundary_data[face.n_proc]["ID"][face.info["ID"]].append(
                    sets.boundary_sets[s].boundary_data
                )

    # Loop through faces
    for edge in subdomain.features["edges"]:
        if edge.n_proc > -1 and edge.n_proc != subdomain.rank:
            for s in sets.boundary_set_map[edge.feature_id]:
                send_boundary_data[edge.n_proc]["ID"][edge.info["ID"]].append(
                    sets.boundary_sets[s].boundary_data
                )

    # Loop through corners
    for corner in subdomain.features["corners"]:
        if corner.n_proc > -1 and corner.n_proc != subdomain.rank:
            for s in sets.boundary_set_map[corner.feature_id]:
                send_boundary_data[corner.n_proc]["ID"][corner.info["ID"]].append(
                    sets.boundary_sets[s].boundary_data
                )

    return send_boundary_data


# def boundary_set_unpack(subdomain, sets, face_recv, edge_recv, corner_recv):
#     """
#     Unpack the neighboring boundary set data
#     """

#     external_sets = {key: [] for key in orientation.features}

#     # Faces
#     for face in subdomain.features["faces"]:
#         if face.n_proc > -1 and face.n_proc != subdomain.rank:
#             external_sets[face.info["ID"]].extend(
#                 face_recv[face.ID]["ID"][face.opp_info["ID"]]
#             )
#         elif face.n_proc == subdomain.rank:
#             for s in sets.boundary_set_map[
#                 orientation.get_boundary_id(face.opp_info["ID"])
#             ]:
#                 # print(s)
#                 # for ss in s:
#                 external_sets[face.info["ID"]].append(
#                     sets.boundary_sets[s].boundary_data
#                 )

#     # Edges
#     for edge in subdomain.features["edges"]:
#         if edge.n_proc > -1 and edge.n_proc != subdomain.rank:
#             external_sets[edge.info["ID"]].extend(
#                 edge_recv[edge.ID]["ID"][edge.opp_info["ID"]]
#             )
#         elif edge.n_proc == subdomain.rank:
#             for s in sets.boundary_set_map[
#                 orientation.get_boundary_id(edge.opp_info["ID"])
#             ]:
#                 # for ss in s:
#                 external_sets[edge.info["ID"]].append(
#                     sets.boundary_sets[s].boundary_data
#                 )

#     # Corners
#     for corner in subdomain.features["corners"]:
#         if corner.n_proc > -1 and corner.n_proc != subdomain.rank:
#             external_sets[corner.info["ID"]].extend(
#                 corner_recv[corner.ID]["ID"][corner.opp_info["ID"]]
#             )
#         elif corner.n_proc == subdomain.rank:
#             for s in sets.boundary_set_map[
#                 orientation.get_boundary_id(corner.opp_info["ID"])
#             ]:
#                 # for ss in s:
#                 external_sets[corner.info["ID"]].append(
#                     sets.boundary_sets[s].boundary_data
#                 )

#     return external_sets


def communicate_features(subdomain, send_data, unpack=True, feature_types=None):
    """
    Send data between processes for faces, edges, and corners.
    This also swaps the feature ids!!

    Args:
        subdomain (object): The subdomain object containing rank and features information.
        send_data (dict): The data to be sent to neighboring processes.
        unpack (bool, optional): If True, unpack the received data. Defaults to False.
        feature_types (list, optional): List of feature types to communicate. Defaults to ["faces", "edges", "corners"].

    Returns:
        dict: Received data from neighboring processes. If unpack is True, returns unpacked received data.
    """
    recv_data = {}
    data_per_process = {}

    if feature_types is None:
        feature_types = ["faces", "edges", "corners"]

    for feature_type in feature_types:
        for feature_id in subdomain.features[feature_type]:
            feature = subdomain.features[feature_type][feature_id]

            # Neighbor process
            if (
                feature.neighbor_rank > -1
                and feature.neighbor_rank != subdomain.rank
                and feature_id in send_data
            ):

                if feature.neighbor_rank not in data_per_process:
                    data_per_process[feature.neighbor_rank] = {}

                data_per_process[feature.neighbor_rank][feature_id] = send_data[
                    feature_id
                ]

    # if subdomain.rank == 2:
    #     print("data per process", data_per_process)

    recv_data = send_recv(
        subdomain.rank,
        data_per_process,
    )

    # if subdomain.rank == 0:
    #     if 2 in recv_data.keys():
    #         print("chawsivhweiogvhfnewriovhwiov", recv_data[2][(0, -1, 0)])

    if unpack:
        _recv_data = {}
        for feature_type in feature_types:
            for feature_id in subdomain.features[feature_type]:
                feature = subdomain.features[feature_type][feature_id]
                if (
                    feature.neighbor_rank > -1
                    and feature.neighbor_rank != subdomain.rank
                    and feature_id in send_data
                ):
                    # Swap the feature ids here
                    _recv_data[feature_id] = recv_data[feature.neighbor_rank][
                        feature.opp_info
                    ]

                # Periodic boundary conditions where process is own neighbor
                elif (
                    feature.neighbor_rank > -1
                    and feature.neighbor_rank == subdomain.rank
                    and feature.opp_info in send_data
                ):
                    # Swap the feature ids here
                    _recv_data[feature_id] = send_data[feature.opp_info]

        recv_data = _recv_data

    return recv_data


def send_recv(rank, data_per_process):
    """
    Performs non-blocking sends and blocking receives for inter-process communication.

    This function uses MPI to send data to and receive data from other processes.
    Each process sends data to all other processes except itself, using non-blocking
    sends (`isend`). It receives data from all other processes using blocking receives
    (`recv`). After initiating all non-blocking sends, it waits for all sends to complete.

    @param rank The rank of the current process.
    @param data_per_process A dictionary where keys represent process ranks, and values
                            are the data to be sent to the respective processes.
                            The rank key's value is ignored as a process does not send
                            data to itself.

    @return A dictionary where keys represent the ranks of processes that sent data to
            the current process, and values are the received data.
    """

    send_request = {}
    receive_data = {}

    for n_proc in data_per_process:
        if n_proc != rank:
            try:
                send_request[n_proc] = comm.isend(
                    data_per_process[n_proc],
                    dest=n_proc,
                )

                receive_data[n_proc] = comm.recv(
                    source=n_proc,
                )
            except Exception as e:
                print(f"Error initiating communication with process {n_proc}: {e}")

    # Wait for all sends to complete
    try:
        MPI.Request.waitall(list(send_request.values()))
    except Exception as e:
        print(f"Error completing send requests: {e}")

    return receive_data
