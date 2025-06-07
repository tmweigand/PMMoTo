"""interface to _voxelss.pyx"""

import numpy as np
from numpy.typing import NDArray
from typing import Any, TypeVar

T = TypeVar("T", bound=np.generic)
FEATURE_TYPE = TypeVar("FEATURE_TYPE", bound=tuple[int, ...])

def renumber_img(img: NDArray[T], conversion_map: dict[T, T]) -> NDArray[T]: ...
def get_id(ind: tuple[int, ...], total_voxels: tuple[int, ...]) -> np.uint64: ...
def gen_img_to_label_map(img: NDArray[T], labels: NDArray[T]) -> NDArray[Any]: ...
def count_label_voxels(img: NDArray[Any], map: dict[int, int]) -> dict[int, int]: ...
def find_unique_pairs(to_match: NDArray[Any]) -> NDArray[Any]: ...
def process_matches_by_feature(
    matches: NDArray[Any],
    unique_matches: dict[tuple[int, int], dict[str, tuple[int, int]]],
    rank: int,
    neighbor_rank: int,
) -> dict[tuple[int, int], dict[str, tuple[int, int]]]: ...
def get_nearest_boundary_index_face(
    img: NDArray[T],
    dimension: int,
    forward: bool,
    label: int,
    lower_skip: int = 0,
    upper_skip: int = 0,
) -> NDArray[T]: ...
def merge_matched_voxels(
    all_match_data: list[dict[tuple[int, int], dict[str, tuple[int, int]]]],
) -> tuple[dict[tuple[int, int], dict[str, int]], int]: ...
