import numpy as np
from mpi4py import MPI
import xml.etree.ElementTree as ET


from ..core import subdomain_padded
from . import io_utils
from . import evtk


comm = MPI.COMM_WORLD

__all__ = [
    "save_particle_data",
    "save_img_data_serial",
    "save_img_data_parallel",
    "save_extended_img_data_parallel",
    "save_img",
]


def save_particle_data(file_name: str, subdomain, particles, **kwargs):
    """Save particle data as VTK PolyData
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


def create_pvtu_file(file_name: str, subdomain):
    """Create a .pvtu file that groups all .vtu files.
    """
    comm = MPI.COMM_WORLD
    num_procs = comm.Get_size()
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

    # Add references to individual .vtu files
    for rank in range(num_procs):
        piece = ET.SubElement(collection, "Piece", Source=f"{base_name}Proc.{rank}.vtu")

    # Write the .pvtu file
    tree = ET.ElementTree(root)
    tree.write(pvtu_file, encoding="utf-8", xml_declaration=True)


def save_img_data_serial(file_name: str, subdomains: dict, img: dict, **kwargs):
    """This functions saves a decomposed image that is one a single process
    """
    if type(subdomains) is not dict:
        subdomains = {0: subdomains}

    if type(img) is not dict:
        img = {0: img}

    io_utils.check_file_path(file_name)

    for local_subdomain, local_img in zip(subdomains.values(), img.values()):
        save_img_data_proc(file_name, local_subdomain, local_img, **kwargs)

    write_parallel_VTK_img(file_name, subdomains[0], img[0], **kwargs)


def save_img_data_parallel(file_name, subdomain, img, additional_img=None):
    """_summary_

    Args:
        file_name (_type_): _description_
        subdomain (_type_): _description_
        img (_type_): _description_

    """
    if subdomain.rank == 0:
        io_utils.check_file_path(file_name)
    comm.barrier()

    save_img_data_proc(file_name, subdomain, img, additional_img)

    if subdomain.rank == 0:
        write_parallel_VTK_img(file_name, subdomain, img, additional_img)


def save_img(file_name, img, resolution=None, **kwargs):
    """Save an image as is.
    """
    io_utils.check_file_path(file_name)

    if resolution is None:
        resolution = (1, 1, 1)

    data = {"img": img}
    for key, value in kwargs.items():
        data[key] = value

    evtk.imageToVTK(
        path=file_name,
        origin=(0, 0, 0),
        start=(0, 0, 0),
        end=img.shape,
        spacing=resolution,
        cellData=data,
    )


def save_extended_img_data_parallel(
    file_name, subdomain, img, extension=((0, 0), (0, 0), (0, 0)), additional_img=None
):
    """Save an image where img.shape > subdomain.voxels
    """
    if img.shape == subdomain.voxels:
        raise ValueError(
            f"Invalid img size! img has same shape as subdomain.voxels {img.shape}. Use save_img_data_proc"
        )

    if subdomain.rank == 0:
        io_utils.check_file_path(file_name)
    comm.barrier()

    save_extended_img_data_proc(file_name, subdomain, img, extension)
    if subdomain.rank == 0:
        write_parallel_VTK_img(file_name, subdomain, img, additional_img, extension)


def save_img_data_proc(file_name, subdomain, img, additional_img=None):
    """_summary_

    Args:
        file_name (_type_): _description_
        subdomain (_type_): _description_
        img (_type_): _description_


        additional_img is a dictionary like
            additional_img[name] = another_img
        extend only needed if another_img.shape != img.shape

    """
    if img.shape != subdomain.voxels:
        raise ValueError(
            f"Invalid img size! img has shape {img.shape} but your subdomain only has {subdomain.voxels} voxels "
        )

    io_utils.check_file_path(file_name)
    file_proc = (
        file_name + "/" + file_name.split("/")[-1] + "Proc." + str(subdomain.rank)
    )

    point_data = {"img": img}
    point_data_info = {"img": (img.dtype, 1)}
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
                    f"Invalid img size! {key} has shape {value.shape}. Required shape is {img.shape} "
                )

            point_data[key] = value
            point_data_info[key] = (value.dtype, 1)

    evtk.imageToVTK(
        path=file_proc,
        spacing=subdomain.domain.resolution,
        cellData=point_data,
        origin=origin,
        start=subdomain.start,
        end=[sum(s) for s in zip(subdomain.start, subdomain.voxels)],
    )


def save_extended_img_data_proc(file_name, subdomain, img, extension):
    """_summary_"""
    io_utils.check_file_path(file_name)
    file_proc = file_name + "/" + file_name.split("/")[-1] + "Proc."

    point_data = {"img": img}
    point_data_info = {"img": (img.dtype, 1)}

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
    file_name, subdomain, img, additional_img=None, extension=((0, 0), (0, 0), (0, 0))
):
    """Wrapper to evtk.writeParallelVTKGrid
    """
    local_file_proc = (
        file_name.split("/")[-1] + "/" + file_name.split("/")[-1] + "Proc."
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
        _sd = subdomain_padded.PaddedSubdomain(n, subdomain.domain)
        name[n] = name[n] + str(n) + ".vti"
        _index = _sd.get_index(rank=n, subdomains=subdomain.domain.subdomains)
        # starts[n] = _sd.get_start()
        starts[n] = [s - e[0] for s, e in zip(_sd.get_start(), extension)]
        ends[n] = [
            s + v + e[1]
            for s, v, e in zip(
                _sd.get_start(),
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
