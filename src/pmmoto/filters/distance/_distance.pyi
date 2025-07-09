"""interface to _distance.pyx"""

import numpy as np
from numpy.typing import NDArray

class Hull:
    vertex: int
    height: float
    range: float

def get_initial_envelope_correctors(
    img: NDArray[np.uint8], dimension: int
) -> tuple[NDArray[np.float32], NDArray[np.float32]]: ...
def get_initial_envelope(
    img: NDArray[np.uint8],
    img_out: NDArray[np.float32],
    dimension: int,
    pad: int = 0,
    resolution: float = 1,
    lower_boundary: None | NDArray[np.float32] = None,
    upper_boundary: None | NDArray[np.float32] = None,
) -> NDArray[np.float32]: ...
def get_parabolic_envelope(
    img: NDArray[np.float32],
    dimension: int,
    resolution: float = 1,
    lower_hull: None | list[list[Hull]] = None,
    upper_hull: None | list[list[Hull]] = None,
) -> None: ...
def get_initial_envelope_correctors_2d(
    img: NDArray[np.uint8], dimension: int
) -> tuple[NDArray[np.float32], NDArray[np.float32]]: ...
def get_initial_envelope_2d(
    img: NDArray[np.uint8],
    img_out: NDArray[np.float32],
    dimension: int,
    pad: int = 0,
    resolution: float = 1,
    lower_boundary: None | NDArray[np.float32] = None,
    upper_boundary: None | NDArray[np.float32] = None,
) -> NDArray[np.float32]: ...
def get_boundary_hull_2d(
    img: NDArray[np.float32],
    bound: NDArray[np.int64],
    dimension: int,
    resolution: float,
    num_hull: int,
    forward: bool = True,
    lower_skip: int = 0,
    upper_skip: int = 0,
) -> list[list[Hull]]: ...
def get_parabolic_envelope_2d(
    img: NDArray[np.float32],
    dimension: int,
    lower_hull: None | list[list[Hull]] = None,
    upper_hull: None | list[list[Hull]] = None,
    resolution: float = 1,
    pad: int = 0,
) -> None: ...
def get_boundary_hull(
    img: NDArray[np.float32],
    bound: NDArray[np.int64],
    dimension: int,
    resolution: float,
    num_hull: int,
    forward: bool = True,
    lower_skip: int = 0,
    upper_skip: int = 0,
) -> list[list[Hull]]: ...
