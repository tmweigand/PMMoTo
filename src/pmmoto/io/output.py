"""Output utilities for saving PMMoTo simulation data.

This module provides functions to save particle and image data in VTK formats,
supporting both serial and parallel output.
"""

from __future__ import annotations
from typing import Any, TypeVar
import numpy as np
from numpy.typing import NDArray
from mpi4py import MPI
import xml.etree.ElementTree as ET

from ..core.subdomain import Subdomain
from ..core.subdomain_padded import PaddedSubdomain
from ..core.subdomain_verlet import VerletSubdomain
from . import io_utils
from . import evtk


T = TypeVar("T", bound=np.generic)

comm = MPI.COMM_WORLD

__all__ = [
    "save_particle_data",
    "save_img",
    "save_extended_img_data_parallel",
]


def save_particle_data(
    file_name: str,
    subdomain: Subdomain | PaddedSubdomain | VerletSubdomain,
    particles: NDArray[np.floating[Any]],
    **kwargs: Any,
) -> None:
    """Save particle data as VTK PolyData.

    Args:
        file_name (str): Output file base name.
        subdomain: Subdomain object with rank attribute.
        particles (np.ndarray): Particle data array.
        **kwargs: Additional keyword arguments.

    """
    if subdomain.rank == 0:
        io_utils.check_file_path(file_name)
    comm.barrier()

    data_length = particles.shape[1]
    data = {}
    if data_length > 3:
        radius = np.ascontiguousarray(particles[:, 3])
        data["radius"] = radius
    if data_length > 4:
        own = np.ascontiguousarray(particles[:, 4])
        data["own"] = own
    if data_length > 5:
        label = np.ascontiguousarray(particles[:, 5])
        data["label"] = label

    file_proc = (
        file_name + "/" + file_name.split("/")[-1] + "Proc." + str(subdomain.rank)
    )

    evtk.pointsToVTK(
        file_proc,
        np.ascontiguousarray(particles[:, 0]),
        np.ascontiguousarray(particles[:, 1]),
        np.ascontiguousarray(particles[:, 2]),
        data=data,
    )

    # Rank 0 creates the .pvtp file
    comm.barrier()
    if subdomain.rank == 0:
        create_pvtu_file(file_name, subdomain)


def create_pvtu_file(
    file_name: str, subdomain: Subdomain | PaddedSubdomain | VerletSubdomain
) -> None:
    """Create a .pvtu file that groups all .vtu files.

    Args:
        file_name (str): Output file base name.
        subdomain: Subdomain object.

    """
    base_name = file_name.split("/")[-1]
    pvtu_file = f"{file_name}/{base_name}.pvtu"

    # Create the root XML element
    root = ET.Element(
        "VTKFile",
        type="PUnstructuredGrid",
        version="1.0",
        byte_order="LittleEndian",
        header_type="UInt64",  # match your vtu file
    )
    collection = ET.SubElement(root, "PUnstructuredGrid")

    # Add point data structure (match your radius field!)
    point_data = ET.SubElement(collection, "PPointData", Scalars="radius")
    ET.SubElement(
        point_data,
        "PDataArray",
        type="Float64",  # MATCH type
        Name="radius",  # MATCH name
        NumberOfComponents="1",
        format="appended",  # MATCH format
    )

    # Add cell data (empty for now)
    ET.SubElement(collection, "PCellData")

    # Add points
    points = ET.SubElement(collection, "PPoints")
    ET.SubElement(
        points,
        "PDataArray",
        type="Float64",  # MATCH type
        Name="points",  # MATCH name (lowercase!)
        NumberOfComponents="3",
        format="appended",  # MATCH format
    )

    # Write the .pvtu file
    tree = ET.ElementTree(root)
    tree.write(pvtu_file, encoding="utf-8", xml_declaration=True)


def save_img(
    file_name: str,
    subdomain: (
        dict[int, Subdomain | PaddedSubdomain | VerletSubdomain]
        | Subdomain
        | PaddedSubdomain
        | VerletSubdomain
    ),
    img: dict[int, NDArray[T]] | NDArray[T],
    **kwargs: Any,
) -> None:
    """Save image(s).

    Args:
        file_name (str): Output file base name.
        subdomain (dict): Dictionary of subdomain objects.
        img (dict): Dictionary of image arrays.
        **kwargs: Additional keyword arguments.

    """
    # Mainly for debugging but when decomposing multiple subdomains
    # when size != num_subdomains
    if isinstance(subdomain, dict) and isinstance(img, dict):
        _save_img_multiple_serial(file_name, subdomain, img, **kwargs)
    elif not isinstance(subdomain, dict) and not isinstance(img, dict):
        if subdomain.domain.num_subdomains > 1:
            _save_img_parallel(file_name, subdomain, img, **kwargs)
        else:
            _save_img(file_name, subdomain, img, **kwargs)


def _save_img_multiple_serial(
    file_name: str,
    subdomains: dict[int, Subdomain | PaddedSubdomain | VerletSubdomain],
    img: dict[int, NDArray[T]],
    **kwargs: Any,
) -> None:
    """Save a decomposed image that is on a single process.

    Args:
        file_name (str): Output file base name.
        subdomains (dict): Dictionary of subdomain objects.
        img (dict): Dictionary of image arrays.
        **kwargs: Additional keyword arguments.

    """
    io_utils.check_file_path(file_name, extra_info="_proc")

    for local_subdomain, local_img in zip(subdomains.values(), img.values()):
        _save_img_proc(file_name, local_subdomain, local_img, **kwargs)

    write_parallel_VTK_img(file_name, subdomains[0], img[0], **kwargs)


def _save_img_parallel(
    file_name: str,
    subdomain: Subdomain | PaddedSubdomain | VerletSubdomain,
    img: NDArray[T],
    additional_img: None | dict[str, NDArray[T]] = None,
) -> None:
    """Save image data in parallel, one file per process.

    Args:
        file_name (str): Output file base name.
        subdomain: Subdomain object with rank attribute.
        img (np.ndarray): Image array.
        additional_img (dict, optional): Additional images to save.

    """
    if subdomain.rank == 0:
        io_utils.check_file_path(file_name, extra_info="_proc")
    comm.barrier()

    _save_img_proc(file_name, subdomain, img, additional_img)

    if subdomain.rank == 0:
        write_parallel_VTK_img(file_name, subdomain, img, additional_img)


def _save_img(
    file_name: str,
    subdomain: Subdomain | PaddedSubdomain | VerletSubdomain,
    img: NDArray[T],
    additional_img: None | dict[str, NDArray[T]] = None,
) -> None:
    """Save an image as a VTK file.

    Args:
        file_name (str): Output file base name.
        subdomain: Subdomain object with rank attribute.
        img (np.ndarray): Image array.
        additional_img: (dict[str, NDArray[T]], optional): {Name: img}.

    """
    io_utils.check_file_path(file_name, create_folder=False)

    _data = {"img": img}
    _data_info = {"img": (img.dtype, 1)}

    origin = [
        subdomain.domain.box[0][0],
        subdomain.domain.box[1][0],
        subdomain.domain.box[2][0],
    ]

    # grab additional images
    if additional_img is not None:
        for key, value in additional_img.items():

            if not isinstance(value, np.ndarray):
                raise TypeError(
                    f"Expected numpy array for {key}, but got {type(value)}"
                )

            if value.shape != img.shape:
                raise ValueError(
                    f"Invalid img size! {key} has shape {value.shape}. "
                    f"Required shape is {img.shape} "
                )
            _data[key] = value
            _data_info[key] = (value.dtype, 1)

    evtk.imageToVTK(
        path=file_name,
        spacing=subdomain.domain.resolution,
        cellData=_data,
        origin=origin,
        start=subdomain.start,
        end=[sum(s) for s in zip(subdomain.start, subdomain.voxels)],
    )


def save_extended_img_data_parallel(
    file_name: str,
    subdomain: Subdomain | PaddedSubdomain | VerletSubdomain,
    img: NDArray[T],
    extension: tuple[tuple[int, int], ...] = ((0, 0), (0, 0), (0, 0)),
    additional_img: None | dict[str, NDArray[T]] = None,
) -> None:
    """Save an image where img.shape > subdomain.voxels.

    Args:
        file_name (str): Output file base name.
        subdomain: Subdomain object.
        img (np.ndarray): Image array.
        extension (tuple, optional): Extension for each dimension.
        additional_img (dict, optional): Additional images to save.

    """
    if img.shape == subdomain.voxels:
        raise ValueError(
            f"Invalid img size! img has same shape as subdomain.voxels {img.shape}."
            "Use save_img_data_proc"
        )

    if subdomain.rank == 0:
        io_utils.check_file_path(file_name)
    comm.barrier()

    save_extended_img_data_proc(file_name, subdomain, img, extension)
    if subdomain.rank == 0:
        write_parallel_VTK_img(file_name, subdomain, img, additional_img, extension)


def _save_img_proc(
    file_name: str,
    subdomain: Subdomain | PaddedSubdomain | VerletSubdomain,
    img: NDArray[T],
    additional_img: None | dict[str, NDArray[T]] = None,
) -> None:
    """Save image data for a single process/subdomain.

    Args:
        file_name (str): Output file base name.
        subdomain: Subdomain object.
        img (np.ndarray): Image array.
        additional_img (dict, optional): Additional images to save.

    """
    if img.shape != subdomain.voxels:
        raise ValueError(
            f"Invalid img size! img has shape {img.shape}."
            f"This subdomain only has {subdomain.voxels} voxels "
        )

    file_name += "_proc"

    io_utils.check_file_path(file_name)
    file_proc = file_name + "/" + file_name.split("/")[-1] + "_" + str(subdomain.rank)

    _save_img(file_proc, subdomain, img, additional_img)


def save_extended_img_data_proc(
    file_name: str,
    subdomain: Subdomain | PaddedSubdomain | VerletSubdomain,
    img: NDArray[T],
    extension: tuple[tuple[int, int], ...],
) -> None:
    """Save extended image data for a single process/subdomain.

    Args:
        file_name (str): Output file base name.
        subdomain: Subdomain object.
        img (np.ndarray): Image array.
        extension (tuple): Extension for each dimension.

    """
    io_utils.check_file_path(file_name)
    file_proc = file_name + "/" + file_name.split("/")[-1] + "_proc_"

    point_data = {"img": img}

    origin = (0, 0, 0)
    # origin = (-extension[0][0], -extension[1][0], -extension[2][0])

    start = [s - e[0] for s, e in zip(subdomain.start, extension)]
    end = [
        s + v + e[1] for s, v, e in zip(subdomain.start, subdomain.voxels, extension)
    ]

    evtk.imageToVTK(
        path=file_proc + str(subdomain.rank),
        origin=origin,
        start=start,
        end=end,
        spacing=subdomain.domain.resolution,
        cellData=point_data,
    )


def write_parallel_VTK_img(
    file_name: str,
    subdomain: Subdomain | PaddedSubdomain | VerletSubdomain,
    img: NDArray[T],
    additional_img: None | dict[str, NDArray[T]] = None,
    extension: tuple[tuple[int, int], ...] = ((0, 0), (0, 0), (0, 0)),
) -> None:
    """Write a parallel VTK image using evtk.writeParallelVTKGrid.

    Args:
        file_name (str): Output file base name.
        subdomain: Subdomain object.
        img (np.ndarray): Image array.
        additional_img (dict, optional): Additional images to save.
        extension (tuple, optional): Extension for each dimension.

    """
    local_file_proc = (
        file_name.split("/")[-1] + "_proc/" + file_name.split("/")[-1] + "_proc_"
    )

    data_info = {"img": (img.dtype, 1)}
    if additional_img is not None:
        for key, value in additional_img.items():
            data_info[key] = (value.dtype, 1)

    lower_extent = [-e[0] for e in extension]
    name = [local_file_proc] * subdomain.domain.num_subdomains
    starts = [[0, 0, 0] for _ in range(subdomain.domain.num_subdomains)]
    ends = [[0, 0, 0] for _ in range(subdomain.domain.num_subdomains)]
    origin = [
        subdomain.domain.box[0][0],
        subdomain.domain.box[1][0],
        subdomain.domain.box[2][0],
    ]

    for n in range(0, subdomain.domain.num_subdomains):
        _sd = PaddedSubdomain(n, subdomain.domain)
        name[n] = name[n] + str(n) + ".vti"
        _index = _sd.get_index(rank=n, subdomains=subdomain.domain.subdomains)
        # starts[n] = _sd.get_start()
        starts[n] = [
            s - e[0]
            for s, e in zip(
                _sd.get_start(
                    index=_index,
                    domain_voxels=subdomain.domain.voxels,
                    subdomains=subdomain.domain.subdomains,
                ),
                extension,
            )
        ]
        ends[n] = [
            s + v + e[1]
            for s, v, e in zip(
                _sd.get_start(
                    index=_index,
                    domain_voxels=subdomain.domain.voxels,
                    subdomains=subdomain.domain.subdomains,
                ),
                _sd.get_voxels(
                    index=_index,
                    domain_voxels=subdomain.domain.voxels,
                    subdomains=subdomain.domain.subdomains,
                ),
                extension,
            )
        ]
        # ends[n] = [sum(x) for x in zip(starts[n], _sd.get_voxels())]

    voxels = [v + e[0] + e[1] for v, e in zip(subdomain.domain.voxels, extension)]

    evtk.writeParallelVTKGrid(
        file_name,
        coordsData=(
            voxels,
            subdomain.coords[0].dtype,
        ),
        origin=origin,
        starts=starts,
        ends=ends,
        sources=name,
        lower_extent=lower_extent,
        spacing=subdomain.domain.resolution,
        cellData=data_info,
    )
