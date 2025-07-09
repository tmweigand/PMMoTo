"""communication.py"""

from __future__ import annotations
from typing import Any, TypeVar

import numpy as np
from numpy.typing import NDArray
from mpi4py import MPI
from .logging import get_logger, USE_LOGGING

from . import utils
from .subdomain import Subdomain
from .subdomain_padded import PaddedSubdomain
from .subdomain_verlet import VerletSubdomain
from .subdomain_features import SubdomainFeatures

T = TypeVar("T", bound=np.generic)

__all__ = [
    "all_gather",
    "all_reduce",
    "gather",
    "update_buffer",
    "communicate_features",
]


comm = MPI.COMM_WORLD
logger = get_logger()

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

if USE_LOGGING:
    logger = get_logger()


def all_gather(data: Any) -> list[Any]:
    """Gather data from all processes and return the combined result.

    Args:
        data: Data to be gathered from each process.

    Returns:
        List containing data from all processes.

    """
    return comm.allgather(data)


def gather(data: T, root: int = 0) -> list[T] | None:
    """Gather data from all processes to the root process.

    Args:
        data: Data to be gathered from each process.
        root: Rank of the root process that receives the data.

    Returns:
        List of data on root process, None on others.

    """
    return comm.gather(data, root=root)


def all_reduce(data: Any, op: str | MPI.Op = "sum") -> Any:
    """Reduce data from all processes using given operation and distribute result.

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

    result = comm.allreduce(data, op=op)
    return result


def update_buffer(
    subdomain: Subdomain | PaddedSubdomain | VerletSubdomain,
    img: NDArray[T],
) -> NDArray[T]:
    """Update the buffer on subdomains based on their current feature info.

    Args:
        subdomain: The subdomain object.
        img: The grid data to be updated.

    Returns:
        The updated grid.

    """
    send_data = buffer_pack(subdomain, img)
    recv_data = communicate_features(subdomain, send_data)
    updated_img = buffer_unpack(subdomain, img, recv_data)
    return updated_img


def update_extended_buffer(
    subdomain: PaddedSubdomain | VerletSubdomain,
    img: NDArray[T],
    buffer: tuple[int, ...],
) -> tuple[NDArray[T], tuple[tuple[int, int], ...]]:
    """Update the buffer on subdomains with additional buffer padding.

    Args:
        subdomain: The subdomain object.
        img: The grid data to be updated.
        buffer: The pad to add to img in each dimension.

    Returns:
        A tuple of:
            - The updated grid with the buffer data.
            - The padding used.

    """
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

    pad, extended_features = subdomain.extend_padding(buffer)
    _img: NDArray[T] = utils.constant_pad_img(img.copy(), pad, pad_value=255)

    send_data = buffer_pack(subdomain, _img, extended_features)
    recv_data = communicate_features(subdomain, send_data)
    updated_img = buffer_unpack(subdomain, _img, recv_data, extended_features)

    return updated_img, pad


def buffer_pack(
    subdomain: Subdomain | PaddedSubdomain | VerletSubdomain,
    img: NDArray[T],
    extended_loop: None | SubdomainFeatures = None,
) -> dict[tuple[int, ...], NDArray[T]]:
    """Pack the buffer data for communication.

    Args:
        subdomain (object): The subdomain object containing features.
        img (numpy.ndarray): The grid data to be packed.
        extended_loop (dict, optional): Extended loop indices for padding.

    Returns:
        dict: A dictionary containing the packed buffer data.

    """
    buffer_data: dict[tuple[int, ...], NDArray[T]] = {}
    for feature_id, feature in subdomain.features.all_features:
        if feature.neighbor_rank > -1:
            if extended_loop is not None:
                own_voxels = extended_loop.get_feature_member(feature_id, "own")
            else:
                own_voxels = feature.own

            buffer_data[feature_id] = img[
                own_voxels[0][0] : own_voxels[0][1],
                own_voxels[1][0] : own_voxels[1][1],
                own_voxels[2][0] : own_voxels[2][1],
            ]

    return buffer_data


def buffer_unpack(
    subdomain: Subdomain | PaddedSubdomain | VerletSubdomain,
    img: NDArray[T],
    features_recv: dict[tuple[int, ...], Any],
    extended_features: None | SubdomainFeatures = None,
) -> NDArray[T]:
    """Update the padding of a subdomain with received buffer data.

    Args:
        subdomain (object): The subdomain object containing features.
        img (numpy.ndarray): The grid data to be updated.
        features_recv (dict): Received buffer data from neighbors.
        extended_features (dict, optional): Extended SubdomainFeatures object.

    Returns:
        numpy.ndarray: The updated grid with buffer data applied.

    """
    buffered_img = img.copy()
    for feature_id, feature in subdomain.features.all_features:
        if feature_id in features_recv:
            if extended_features is not None:
                neighbor = extended_features.get_feature_member(feature_id, "neighbor")
            else:
                neighbor = feature.neighbor

            buffered_img[
                neighbor[0][0] : neighbor[0][1],
                neighbor[1][0] : neighbor[1][1],
                neighbor[2][0] : neighbor[2][1],
            ] = features_recv[feature_id]

    buffered_img = subdomain.set_wall_bcs(buffered_img)
    assert not np.any(buffered_img == 255)

    return buffered_img


def communicate_features(
    subdomain: Subdomain | PaddedSubdomain | VerletSubdomain,
    send_data: dict[tuple[int, ...], Any],
) -> dict[tuple[int, ...], Any]:
    """Send data between processes for faces, edges, and corners.

    This also swaps the feature ids.

    Args:
        subdomain (object): Subdomain object containing rank and features information.
        send_data (dict): The data to be sent to neighboring processes.

    Returns:
        dict: Received data from neighboring processes.
              If unpack is True, returns unpacked received data.

    """
    data_per_process: dict[int, dict[tuple[int, ...], Any]] = {}
    for feature_id, feature in subdomain.features.all_features:
        # Neighbor process
        if (
            feature.neighbor_rank > -1
            and feature.neighbor_rank != subdomain.rank
            and feature_id in send_data
        ):
            if feature.neighbor_rank not in data_per_process:
                data_per_process[feature.neighbor_rank] = {}

            data_per_process[feature.neighbor_rank][feature_id] = send_data[feature_id]

    recv_data_process = send_recv(
        subdomain.rank,
        data_per_process,
    )

    recv_data: dict[tuple[int, ...], Any] = {}
    for feature_id, feature in subdomain.features.all_features:
        if (
            feature.neighbor_rank > -1
            and feature.neighbor_rank != subdomain.rank
            and feature_id in send_data
        ):
            # Swap the feature ids here
            recv_data[feature_id] = recv_data_process[feature.neighbor_rank][
                feature.info.opp
            ]

        # Periodic boundary conditions where process is own neighbor
        elif (
            feature.neighbor_rank > -1
            and feature.neighbor_rank == subdomain.rank
            and feature.info.opp in send_data
        ):
            # Swap the feature ids here
            recv_data[feature_id] = send_data[feature.info.opp]

    return recv_data


def send_recv(rank: int, data_per_process: dict[int, Any]) -> dict[int, Any]:
    """Perform non-blocking sends and receives for inter-process communication.

    Uses a two-phase communication pattern to avoid deadlocks.

    Args:
        rank (int): The rank of the current process.
        data_per_process (dict): Data to send to each process.

    Returns:
        dict: Data received from each process.

    """
    send_requests: dict[int, Any] = {}
    receive_data: dict[int, Any] = {}

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
