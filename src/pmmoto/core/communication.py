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
    "communicate",
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
    Organize the communication to update the padding/buffer on subdomains account for periodic boundary conditions here
    """
    send_data, own_data = buffer_pack(subdomain, grid)

    if send_data:
        recv_data = communicate(subdomain, send_data)
        own_data.update(recv_data)

    buffer_grid = buffer_unpack(subdomain, grid, own_data)

    return buffer_grid


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
    Grab The Slices (based on Buffer Size [1,1,1]) to pack and send to Neighbors
    for faces, edges and corners.
    """

    buffer_data = {}
    periodic_data = {}
    # for feature, n_proc in subdomain.neighbor_ranks.items():
    #     if n_proc > -1 and n_proc != subdomain.rank:
    #         buffer_data[n_proc] = {"ID": {}}
    #         buffer_data[n_proc]["ID"][feature] = None

    feature_types = ["faces", "edges", "corners"]
    for feature_type in feature_types:
        for feature_id, feature in subdomain.features[feature_type].items():
            if feature.neighbor_rank > -1 and feature.neighbor_rank != subdomain.rank:
                buffer_data[feature_id] = grid[
                    feature.loop["own"][0][0] : feature.loop["own"][0][1],
                    feature.loop["own"][1][0] : feature.loop["own"][1][1],
                    feature.loop["own"][2][0] : feature.loop["own"][2][1],
                ]
            elif feature.neighbor_rank == subdomain.rank:
                periodic_data[feature_id] = grid[
                    feature.loop["own"][0][0] : feature.loop["own"][0][1],
                    feature.loop["own"][1][0] : feature.loop["own"][1][1],
                    feature.loop["own"][2][0] : feature.loop["own"][2][1],
                ]
    return buffer_data, periodic_data


def buffer_unpack(subdomain, grid, features_recv):
    """
    Unpack buffer information and account for serial periodic boundary conditions
    """

    feature_types = ["faces", "edges", "corners"]
    for feature_type in feature_types:
        for feature in subdomain.features[feature_type].values():
            if feature.opp_info in features_recv:
                grid[
                    feature.loop["neighbor"][0][0] : feature.loop["neighbor"][0][1],
                    feature.loop["neighbor"][1][0] : feature.loop["neighbor"][1][1],
                    feature.loop["neighbor"][2][0] : feature.loop["neighbor"][2][1],
                ] = features_recv[feature.opp_info]

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


def communicate(subdomain, send_data, unpack=False):
    """
    Send data between processes for faces, edges, and corners.
    """
    recv_data = {}
    feature_types = ["faces", "edges", "corners"]
    for feature in feature_types:
        recv_data[feature] = send_recv(
            subdomain.rank,
            subdomain.features[feature],
            send_data,
        )

    if unpack:
        unpack_recv_data = {}
        for feature in feature_types:
            for feature_id, feature in recv_data[feature].items():
                unpack_recv_data[feature_id] = feature

        return unpack_recv_data

    return recv_data


def send_recv(rank, features, send_data):
    """_summary_

    Args:
        features (_type_): _description_
        num_features (_type_): _description_
        send_data (_type_): _description_

    Returns:
        _type_: _description_
    """

    reqs = {}
    reqr = {}
    recv_data = {}

    for feature_id, feature in features.items():
        if (
            feature.neighbor_rank > -1
            and feature.neighbor_rank != rank
            and feature_id in send_data
        ):
            reqs[feature_id] = comm.isend(
                send_data[feature_id], dest=feature.neighbor_rank
            )

        if (
            feature.neighbor_rank > -1
            and feature.neighbor_rank != rank
            and feature_id in send_data
        ):
            reqr[feature_id] = comm.recv(source=feature.neighbor_rank)

    # Wait for all sends to complete
    MPI.Request.waitall(list(reqs.values()))

    return reqr
