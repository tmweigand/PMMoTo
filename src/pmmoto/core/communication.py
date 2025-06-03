"""communication.py"""

from typing import TypeVar, List, Optional, Union
import numpy as np
from mpi4py import MPI
from . import utils
from .logging import get_logger


__all__ = [
    "all_gather",
    "all_reduce",
    "gather",
    "update_buffer",
    "communicate_features",
]


comm = MPI.COMM_WORLD

_OP_MAP: dict[str, MPI.Op] = {
    "sum": MPI.SUM,
    "prod": MPI.PROD,
    "max": MPI.MAX,
    "min": MPI.MIN,
    "land": MPI.LAND,
    "band": MPI.BAND,
    "lor": MPI.LOR,
    "bor": MPI.BOR,
    "lxor": MPI.LXOR,
    "bxor": MPI.BXOR,
    "minloc": MPI.MINLOC,
    "maxloc": MPI.MAXLOC,
}

T = TypeVar("T")

logger = get_logger()


def all_gather(data: T) -> List[T]:
    """Gather data from all processes and return the combined result.

    Args:
        data: Data to be gathered from each process.

    Returns:
        List containing data from all processes.

    """
    return comm.allgather(data)


def gather(data: T, root: int = 0) -> Optional[List[T]]:
    """Gather data from all processes to the root process.

    Args:
        data: Data to be gathered from each process.
        root: Rank of the root process that receives the data.

    Returns:
        List of data on root process, None on others.

    """
    return comm.gather(data, root=root)


def all_reduce(data: T, op: Union[str, MPI.Op] = "sum") -> T:
    """Reduce data from all processes using the given operation and distribute the result.

    Args:
        data: Local data to reduce (e.g., int, float, NumPy array).
        op: MPI operation or string alias (e.g., 'sum', 'max').

    Returns:
        The reduced result, available to all processes.

    Raises:
        ValueError: If the provided operation is not supported.

    """
    if isinstance(op, str):
        try:
            op = _OP_MAP[op.lower()]
        except KeyError:
            raise ValueError(f"Unsupported reduction operation string: {op!r}")

    elif not isinstance(op, MPI.Op):
        raise TypeError("Operation must be an MPI.Op or string alias")

    return comm.allreduce(data, op=op)


def update_buffer(subdomain, img, buffer=None):
    """Update the buffer on subdomains and account for periodic boundary conditions.

    Args:
        subdomain (object): The subdomain object containing features and neighbor
        information.
        img (numpy.ndarray): The grid data to be updated.
        buffer (list[int], optional): The pad to add to img in each dimension.

    Returns:
        numpy.ndarray or tuple: The updated grid with the buffer data,
                                and pad if buffer is not None.

    """
    if buffer is not None:
        # Ensure buffer does not span more than 1 subdomain
        # If does, exit as horribly inefficient and a
        # different subdomain topology is needed.
        utils.check_subdomain_condition(
            subdomain=subdomain,
            condition_fn=lambda s, b: np.any(s.own_voxels < b),
            args=(buffer,),
            error_message=(
                "The buffer (%s) exceeds at least one dimension of the subdomain (%s). "
                "Simulation stopping."
            ),
            error_args=(buffer, subdomain.own_voxels),
        )
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
    """Pack the buffer data for communication.

    Args:
        subdomain (object): The subdomain object containing features.
        img (numpy.ndarray): The grid data to be packed.
        extended_loop (dict, optional): Extended loop indices for padding.

    Returns:
        dict: A dictionary containing the packed buffer data.

    """
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
    """Update the padding of a subdomain with received buffer data.

    Args:
        subdomain (object): The subdomain object containing features.
        img (numpy.ndarray): The grid data to be updated.
        features_recv (dict): Received buffer data from neighbors.
        extended_loop (dict, optional): Extended loop indices for padding.

    Returns:
        numpy.ndarray: The updated grid with buffer data applied.

    """
    buffered_img = img.copy()

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
    """Send data between processes for faces, edges, and corners.

    This also swaps the feature ids.

    Args:
        subdomain (object): Subdomain object containing rank and features information.
        send_data (dict): The data to be sent to neighboring processes.
        unpack (bool, optional): If True, unpack the received data. Defaults to True.
        feature_types (list, optional): List of feature types to communicate.
                                        Defaults to ["faces", "edges", "corners"].

    Returns:
        dict: Received data from neighboring processes.
              If unpack is True, returns unpacked received data.

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
    """Perform non-blocking sends and receives for inter-process communication.

    Uses a two-phase communication pattern to avoid deadlocks.

    Args:
        rank (int): The rank of the current process.
        data_per_process (dict): Data to send to each process.

    Returns:
        dict: Data received from each process.

    """
    send_requests = {}
    receive_data = {}

    # First phase: Post all sends
    for n_proc in data_per_process:
        if n_proc != rank:
            try:
                send_requests[n_proc] = comm.isend(
                    data_per_process[n_proc], dest=n_proc
                )
            except Exception as e:
                print(f"Error sending to process {n_proc}: {e}")
                raise

    # Second phase: Receive from all processes
    for n_proc in data_per_process:
        if n_proc != rank:
            try:
                receive_data[n_proc] = comm.recv(source=n_proc)
            except Exception as e:
                print(f"Error receiving from process {n_proc}: {e}")
                raise

    # Wait for all sends to complete
    try:
        MPI.Request.waitall(list(send_requests.values()))
    except Exception as e:
        print(f"Error completing send requests: {e}")
        raise

    return receive_data
