"""Unit tests for PMMoTo core communication routines.

Tests include buffer updates, feature communication, and buffer extension
with MPI parallelism.
"""

import numpy as np
from mpi4py import MPI
import pytest
import pmmoto


def test_update_buffer() -> None:
    """Ensure that features and buffer are being communicated to neighbor processes"""
    solution = np.array(
        [
            [
                [26, 24, 25, 26, 24],
                [20, 18, 19, 20, 18],
                [23, 21, 22, 23, 21],
                [26, 24, 25, 26, 24],
                [20, 18, 19, 20, 18],
            ],
            [
                [8, 6, 7, 8, 6],
                [2, 0, 1, 2, 0],
                [5, 3, 4, 5, 3],
                [8, 6, 7, 8, 6],
                [2, 0, 1, 2, 0],
            ],
            [
                [17, 15, 16, 17, 15],
                [11, 9, 10, 11, 9],
                [14, 12, 13, 14, 12],
                [17, 15, 16, 17, 15],
                [11, 9, 10, 11, 9],
            ],
            [
                [26, 24, 25, 26, 24],
                [20, 18, 19, 20, 18],
                [23, 21, 22, 23, 21],
                [26, 24, 25, 26, 24],
                [20, 18, 19, 20, 18],
            ],
            [
                [8, 6, 7, 8, 6],
                [2, 0, 1, 2, 0],
                [5, 3, 4, 5, 3],
                [8, 6, 7, 8, 6],
                [2, 0, 1, 2, 0],
            ],
        ],
        dtype=int,
    )

    subdomains = (1, 1, 1)
    voxels = (3, 3, 3)
    box = ((0, 1), (0, 1), (0, 1))
    boundary_types = (
        (pmmoto.BoundaryType.PERIODIC, pmmoto.BoundaryType.PERIODIC),
        (pmmoto.BoundaryType.PERIODIC, pmmoto.BoundaryType.PERIODIC),
        (pmmoto.BoundaryType.PERIODIC, pmmoto.BoundaryType.PERIODIC),
    )

    sd = pmmoto.initialize(
        box=box,
        subdomains=subdomains,
        voxels=voxels,
        boundary_types=boundary_types,
        rank=0,
    )

    img = np.zeros(sd.voxels)
    own_nodes = [sd.voxels[0] - 2, sd.voxels[1] - 2, sd.voxels[2] - 2]
    img[1:-1, 1:-1, 1:-1] = np.arange(
        own_nodes[0] * own_nodes[1] * own_nodes[2]
    ).reshape(own_nodes)

    updated_grid = pmmoto.core.communication.update_buffer(sd, img)

    np.testing.assert_array_almost_equal(updated_grid, solution)


@pytest.mark.mpi(min_size=8)
def test_communicate_features() -> None:
    """Ensure that features are being communicated to neighbor processes"""
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    sd = pmmoto.initialize(
        box=((0, 1), (0, 1), (0, 1)),
        subdomains=(2, 2, 2),
        voxels=(10, 10, 10),
        boundary_types=(
            (pmmoto.BoundaryType.PERIODIC, pmmoto.BoundaryType.PERIODIC),
            (pmmoto.BoundaryType.PERIODIC, pmmoto.BoundaryType.PERIODIC),
            (pmmoto.BoundaryType.PERIODIC, pmmoto.BoundaryType.PERIODIC),
        ),
        rank=rank,
    )

    feature_data = {}
    for feature_id, feature in sd.features.all_features:
        feature_data[feature_id] = rank

    recv_data = pmmoto.core.communication.communicate_features(
        subdomain=sd,
        send_data=feature_data,
    )

    for feature_id, feature in sd.features.all_features:
        if feature_id in recv_data.keys():
            assert recv_data[feature_id] == feature.neighbor_rank


@pytest.mark.mpi(min_size=8)
def test_update_buffer_with_buffer() -> None:
    """Ensure that features are being communicated to neighbor processes"""
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    sd = pmmoto.initialize(
        box=((0, 1), (0, 1), (0, 1)),
        subdomains=(2, 2, 2),
        voxels=(10, 10, 10),
        boundary_types=(
            (pmmoto.BoundaryType.END, pmmoto.BoundaryType.END),
            (pmmoto.BoundaryType.END, pmmoto.BoundaryType.END),
            (pmmoto.BoundaryType.END, pmmoto.BoundaryType.END),
        ),
        rank=rank,
        pad=(1, 1, 1),
    )

    img = (rank + 1) * np.ones(sd.voxels)
    img = sd.set_wall_bcs(img)

    buffer = (2, 2, 2)

    update_img, halo = pmmoto.core.communication.update_extended_buffer(
        subdomain=sd,
        img=img,
        buffer=buffer,
    )

    # pmmoto.io.output.save_img(
    #     "data_out/test_comm_buffer", sd, img, additional_img={"og": img}
    # )

    # pmmoto.io.output.save_extended_img_data_parallel(
    #     "data_out/test_comm_buffer_extended", sd, update_img, halo
    # )


@pytest.mark.mpi(min_size=8)
def test_update_buffer_with_buffer() -> None:
    """Ensure that features are being communicated to neighbor processes"""
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    box = (
        (0.0, 176),
        (0.0, 176),
        (-100, 100),
    )

    sd = pmmoto.initialize(
        box=box,
        subdomains=(2, 2, 2),
        voxels=(100, 100, 100),
        boundary_types=(
            (pmmoto.BoundaryType.PERIODIC, pmmoto.BoundaryType.PERIODIC),
            (pmmoto.BoundaryType.PERIODIC, pmmoto.BoundaryType.PERIODIC),
            (pmmoto.BoundaryType.END, pmmoto.BoundaryType.END),
        ),
        rank=rank,
        pad=(1, 1, 1),
    )

    img = (rank + 1) * np.ones(sd.voxels)
    img = sd.set_wall_bcs(img)

    buffer = (10, 10, 9)

    update_img, halo = pmmoto.core.communication.update_extended_buffer(
        subdomain=sd,
        img=img,
        buffer=buffer,
    )


@pytest.mark.mpi(min_size=1)
def test_gather():
    """Test MPI gather utility."""
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    data = rank  # simple, deterministic per-rank data
    root = 0

    result = pmmoto.core.communication.gather(data, root=root)

    if rank == root:
        assert isinstance(result, list)
        assert len(result) == size
        assert result == list(range(size))
    else:
        assert result is None


def test_all_reduce_error():
    """Test MPI gather utility."""
    with pytest.raises(ValueError):
        _ = pmmoto.core.communication.all_reduce(1, op="error")


@pytest.mark.mpi(min_size=2)
def test_send_recv():
    """Test non-blocking send/recv communication."""
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    # Each rank sends a message to every other rank
    data_per_process = {r: f"msg_from_{rank}_to_{r}" for r in range(size) if r != rank}

    received = pmmoto.core.communication.send_recv(
        rank,
        data_per_process,
    )

    # We should receive exactly one message from every other rank
    assert set(received.keys()) == {r for r in range(size) if r != rank}

    for src_rank, msg in received.items():
        assert msg == f"msg_from_{src_rank}_to_{rank}"


class DummyRequest:
    """Dummy MPI request for testing waitall."""

    def wait(self):
        pass


class MockComm:
    """Mock communicator to simulate MPI exceptions."""

    def __init__(self, fail_send=False, fail_recv=False, fail_waitall=False):
        self.fail_send = fail_send
        self.fail_recv = fail_recv
        self.fail_waitall = fail_waitall
        self.send_calls = []

    def isend(self, data, dest):
        self.send_calls.append((data, dest))
        if self.fail_send:
            raise RuntimeError(f"mock isend failure for {dest}")
        return DummyRequest()

    def recv(self, source):
        if self.fail_recv:
            raise RuntimeError(f"mock recv failure from {source}")
        return f"data_from_{source}"


def test_send_recv_isend_exception():
    comm = MockComm(fail_send=True)
    with pytest.raises(RuntimeError, match="MPI isend failed for destination rank 1"):
        pmmoto.core.communication.send_recv(
            rank=0, data_per_process={1: "x"}, comm=comm
        )


def test_send_recv_recv_exception():
    comm = MockComm(fail_recv=True)
    with pytest.raises(RuntimeError, match="MPI recv failed for destination rank 1"):
        pmmoto.core.communication.send_recv(
            rank=0, data_per_process={1: "x"}, comm=comm
        )


def test_waitall_exception():
    comm = MockComm()  # normal communicator

    # Define a dummy waitall_func that always fails
    def failing_waitall(requests):
        raise RuntimeError("mock waitall failure")

    with pytest.raises(
        RuntimeError, match="Error completing send requests: mock waitall failure"
    ):
        pmmoto.core.communication.send_recv(
            rank=0, data_per_process={1: "x"}, comm=comm, waitall_func=failing_waitall
        )
