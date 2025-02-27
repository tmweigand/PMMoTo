"""communication.py"""

# from . import orientation
from . import subdomain_features
from . import utils
import copy

import numpy as np
from mpi4py import MPI

__all__ = [
    "all_gather",
    "gather",
    "update_buffer",
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


def update_buffer(subdomain, img, buffer=None):
    """
    Organize the communication to update the padding/buffer on subdomains and account for periodic boundary conditions.

    Args:
        subdomain (object): The subdomain object containing features and neighbor information.
        img (numpy.ndarray): The grid data to be updated.
        buffer list[int,...]: The pad to add to img in each dimension

    Returns:
        numpy.ndarray: The updated grid with the buffer data.
    """

    if buffer is not None:
        pad, extended_loop = subdomain.extend_padding(buffer)
        _img = utils.constant_pad_img(img.copy(), pad, 255)
    else:
        extended_loop = None
        _img = img

    send_data = buffer_pack(subdomain, _img, extended_loop)
    recv_data = communicate_features(subdomain, send_data)
    updated_img = buffer_unpack(subdomain, _img, recv_data, extended_loop)

    if buffer is not None:
        return updated_img, pad

    return updated_img


def buffer_pack(subdomain, img, extended_loop=None):
    """
    Packs the buffer data for communication.

    Args:
        subdomain (object): The subdomain object containing features.
        grid (numpy.ndarray): The grid data to be packed.

    Returns:
        dict: A dictionary containing the packed buffer data.
    """

    # Check if img can be extended based on boundary conditions
    # if pad_extend is not None:
    #     _pad_extend = subdomain.extend_padding(pad_extend)
    #     summed_pad = [
    #         (s[0] + p[0], s[1] + p[1]) for s, p in zip(subdomain.pad, _pad_extend)
    #     ]
    #     voxels = [v + p[0] + p[1] for v, p in zip(subdomain.voxels, _pad_extend)]

    #     img = utils.constant_pad_img(img.copy(), _pad_extend, 255)

    buffer_data = {}
    feature_types = ["faces", "edges", "corners"]
    for feature_type in feature_types:
        for feature_id, feature in subdomain.features[feature_type].items():
            if feature.neighbor_rank > -1:
                if extended_loop is not None:
                    loop = extended_loop[feature_id]
                else:
                    loop = feature.loop

                buffer_data[feature_id] = img[
                    loop["own"][0][0] : loop["own"][0][1],
                    loop["own"][1][0] : loop["own"][1][1],
                    loop["own"][2][0] : loop["own"][2][1],
                ]

    return buffer_data


def buffer_unpack(subdomain, img, features_recv, extended_loop=None):
    """
    This function updates the padding of a subdomain.
    Alternatively, if pad_extend is a tuple of pad size like: ((1, 1), (1, 1), (1, 1)),
    a new image is returned. Boundary conditions are applied to the extended image.
    """
    buffered_img = img.copy()

    # Check if img can be extended based on boundary conditions
    # if pad_extend is not None:
    #     _pad_extend = subdomain.extend_padding(pad_extend)

    #     summed_pad = [
    #         (s[0] + p[0], s[1] + p[1]) for s, p in zip(subdomain.pad, _pad_extend)
    #     ]
    #     voxels = [v + p[0] + p[1] for v, p in zip(subdomain.voxels, _pad_extend)]

    #     buffered_img = utils.constant_pad_img(buffered_img, _pad_extend, 255)

    feature_types = ["faces", "edges", "corners"]
    for feature_type in feature_types:
        for feature_id, feature in subdomain.features[feature_type].items():
            if feature_id in features_recv:

                if extended_loop is not None:
                    loop = extended_loop[feature_id]
                else:
                    loop = feature.loop

                buffered_img[
                    loop["neighbor"][0][0] : loop["neighbor"][0][1],
                    loop["neighbor"][1][0] : loop["neighbor"][1][1],
                    loop["neighbor"][2][0] : loop["neighbor"][2][1],
                ] = features_recv[feature_id]

    buffered_img = subdomain.set_wall_bcs(buffered_img)
    assert not np.any(buffered_img == 255)

    return buffered_img


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

    recv_data = send_recv(
        subdomain.rank,
        data_per_process,
    )

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
