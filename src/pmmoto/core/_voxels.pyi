"""interface to _voxelss.pyx"""

import numpy as np
from numpy.typing import NDArray
from typing import Any, TypeVar

T = TypeVar("T", bound=np.generic)
FEATURE_TYPE = TypeVar("FEATURE_TYPE", bound=tuple[int, ...])
INT = TypeVar("INT", np.integer[Any], np.unsignedinteger[Any])
INT2 = TypeVar("INT2", np.integer[Any], np.unsignedinteger[Any])

def renumber_img(
    img: NDArray[INT2], conversion_map: dict[INT2, INT]
) -> NDArray[INT]: ...
def get_id(ind: tuple[int, ...], total_voxels: tuple[int, ...]) -> np.uint64: ...
def gen_img_to_label_map(
    img: NDArray[INT], labels: NDArray[INT2]
) -> dict[INT2, INT]: ...
def count_label_voxels(img: NDArray[Any], map: dict[int, int]) -> dict[int, int]: ...
def find_unique_pairs(to_match: NDArray[INT]) -> NDArray[INT]: ...
def process_matches_by_feature(
    matches: NDArray[INT],
    unique_matches: dict[tuple[int, INT], dict[str, tuple[int, INT]]],
    rank: int,
    neighbor_rank: int,
) -> dict[tuple[int, INT], dict[str, tuple[int, INT]]]: ...
def get_nearest_boundary_index_face(
    img: NDArray[T],
    dimension: int,
    forward: bool,
    label: int,
    lower_skip: int = 0,
    upper_skip: int = 0,
) -> NDArray[T]: ...
def merge_matched_voxels(
    all_match_data: list[dict[tuple[int, INT], dict[str, tuple[int, INT]]]],
) -> tuple[dict[tuple[int, INT], dict[str, INT]], int]: ...
def get_nearest_boundary_index_face_2d(
    img: NDArray[T],
    dimension: int,
    forward: bool,
    label: int,
    lower_skip: int = 0,
    upper_skip: int = 0,
) -> NDArray[np.int64]: ...
