import numpy as np
from mpi4py import MPI

# from evtk import (
#     pointsToVTK,
#     gridToVTK,
#     writeParallelVTKGrid,
#     _addDataToParallelFile,
# )
# from pyevtk import vtk

from ..core import subdomain_padded
from . import io_utils
from . import evtk


comm = MPI.COMM_WORLD

__all__ = [
    "save_grid_data_serial",
    "save_grid_data_parallel",
    "save_grid_data_proc",
    "save_grid",
    # "save_grid_data_deconstructed",
    # "save_grid_data_csv",
    # "save_set_data",
]


def save_grid_data_serial(file_name, subdomains, grid, **kwargs):
    """_summary_

    Args:
        file_name (_type_): _description_
        subdomains (_type_): _description_
        grid (_type_): _description_
    """

    io_utils.check_file_path(file_name)

    if type(subdomains) is not list:
        subdomains = [subdomains]

    for subdomain in subdomains:
        save_grid_data_proc(file_name, subdomain, grid, **kwargs)

    write_parallel_VTK_grid(file_name, subdomains[0], grid, **kwargs)


def save_grid_data_parallel(file_name, subdomain, domain, grid, **kwargs):
    """_summary_

    Args:
        file_name (_type_): _description_
        subdomain (_type_): _description_
        grid (_type_): _description_
    """

    if subdomain.rank == 0:
        io_utils.check_file_path(file_name)
    comm.barrier()

    save_grid_data_proc(file_name, subdomain, grid, **kwargs)

    if subdomain.rank == 0:
        write_parallel_VTK_grid(file_name, subdomain, domain, grid, **kwargs)


def save_grid_data_proc(file_name, subdomain, grid, **kwargs):
    """_summary_

    Args:
        file_name (_type_): _description_
        subdomain (_type_): _description_
        grid (_type_): _description_
    """

    io_utils.check_file_path(file_name)
    file_proc = file_name + "/" + file_name.split("/")[-1] + "Proc."

    point_data = {"grid": grid}
    point_data_info = {"grid": (grid.dtype, 1)}
    for key, value in kwargs.items():
        point_data[key] = value
        point_data_info[key] = (value.dtype, 1)

    evtk.imageToVTK(
        path=file_proc + str(subdomain.rank),
        origin=(0, 0, 0),
        start=subdomain.start,
        end=[sum(x) for x in zip(subdomain.start, subdomain.voxels)],
        spacing=subdomain.domain.resolution,
        cellData=point_data,
    )


def save_grid(file_name, grid, **kwargs):
    """_summary_

    Args:
        file_name (_type_): _description_
        grid (_type_): _description_
    """

    io_utils.check_file_path(file_name)

    data = {"grid": grid}
    for key, value in kwargs.items():
        data[key] = value

    evtk.imageToVTK(
        path=file_name,
        origin=(0, 0, 0),
        start=(0, 0, 0),
        end=grid.shape,
        spacing=(1, 1, 1),
        cellData=data,
    )


def write_parallel_VTK_grid(file_name, subdomain, grid, **kwargs):
    """
    Wrapper to evtk.writeParallelVTKGrid
    """
    local_file_proc = (
        file_name.split("/")[-1] + "/" + file_name.split("/")[-1] + "Proc."
    )

    data_info = {"grid": (grid.dtype, 1)}
    for key, value in kwargs.items():
        data_info[key] = (value.dtype, 1)

    name = [local_file_proc] * subdomain.domain.num_subdomains
    starts = [[0, 0, 0] for _ in range(subdomain.domain.num_subdomains)]
    ends = [[0, 0, 0] for _ in range(subdomain.domain.num_subdomains)]

    for n in range(0, subdomain.domain.num_subdomains):
        _sd = subdomain_padded.PaddedSubdomain(n, subdomain.domain)
        name[n] = name[n] + str(n) + ".vti"
        _index = _sd.get_index()
        starts[n] = _sd.get_start()
        ends[n] = [sum(x) for x in zip(starts[n], _sd.get_voxels())]

    evtk.writeParallelVTKGrid(
        file_name,
        coordsData=(
            subdomain.domain.voxels,
            subdomain.coords[0].dtype,
        ),
        starts=starts,
        ends=ends,
        sources=name,
        spacing=subdomain.domain.resolution,
        cellData=data_info,
    )


# def save_set_data(file_name, subdomain, set_list, **kwargs):
#     """_summary_

#     Args:
#         file_name (_type_): _description_
#         subdomain (_type_): _description_
#         set_list (_type_): _description_
#     """

#     rank = subdomain.rank

#     if rank == 0:
#         io_utils.check_file_path(file_name)
#     comm.barrier()

#     proc_set_counts = comm.allgather(set_list.count.all)
#     nonzero_proc = np.where(np.asarray(proc_set_counts) > 0)[0][0]

#     ### Place Set Values in Arrays
#     if set_list.count.all > 0:
#         dim = 0
#         for local_ID, ss in set_list.sets.items():
#             dim = dim + len(ss.node_data.nodes)
#         x = np.zeros(dim)
#         y = np.zeros(dim)
#         z = np.zeros(dim)
#         set_rank = rank * np.ones(dim, dtype=np.uint8)
#         global_ID = np.zeros(dim, dtype=np.uint64)
#         local_ID = np.zeros(dim, dtype=np.uint64)
#         phase = np.ones(dim, dtype=np.uint8)
#         point_data = {
#             "set": set_rank,
#             "globalID": global_ID,
#             "localID": local_ID,
#             "phase": phase,
#         }
#         point_data_info = {
#             "set": (set_rank.dtype, 1),
#             "globalID": (global_ID.dtype, 1),
#             "localID": (local_ID.dtype, 1),
#             "phase": (phase.dtype, 1),
#         }

#         ### Handle kwargs - need to fix for values that are subclasses!
#         for key, value in kwargs.items():

#             sub_classes = value.count(".")
#             val_split = value.split(".")

#             valid_data = True
#             next = list(set_list.sets.values())[0]
#             for n, child in enumerate(val_split):
#                 if hasattr(next, child):
#                     next = getattr(next, child)
#                 else:
#                     valid_data = False

#             if not valid_data:
#                 if rank == 0:
#                     print(
#                         f"Error: Cannot save set data as kwarg {value} is not an attribute in Set"
#                     )
#                 communication.raiseError()

#             dataType = type(next)
#             if dataType == bool:  ### pyectk does not support bool?
#                 dataType = np.uint8
#             point_data[key] = np.zeros(dim, dtype=dataType)
#             point_data_info[key] = (point_data[key].dtype, 1)

#         c = 0
#         for ss in set_list.sets.values():
#             indexs = np.unravel_index(ss.node_data.nodes, ss.node_data.index_map)
#             for index in zip(indexs[0], indexs[1], indexs[2]):
#                 x[c] = subdomain.coords[0][index[0]]
#                 y[c] = subdomain.coords[1][index[1]]
#                 z[c] = subdomain.coords[2][index[2]]

#                 global_ID[c] = ss.global_ID
#                 local_ID[c] = ss.local_ID
#                 phase[c] = ss.phase
#                 for key, value in kwargs.items():

#                     val_split = value.split(".")
#                     next = ss
#                     for n, child in enumerate(val_split):
#                         next = getattr(next, child)
#                     point_data[key][c] = next
#                 c = c + 1

#         file_proc = file_name + "/" + file_name.split("/")[-1] + "Proc."
#         local_file_proc = (
#             file_name.split("/")[-1] + "/" + file_name.split("/")[-1] + "Proc."
#         )
#         pointsToVTK(file_proc + str(rank), x, y, z, data=point_data)

#     if rank == nonzero_proc:
#         w = vtk.VtkParallelFile(file_name, vtk.VtkPUnstructuredGrid)
#         w.openGrid()
#         point_data = point_data_info
#         _addDataToParallelFile(w, cellData=None, pointData=point_data)
#         w.openElement("PPoints")
#         w.addHeader("points", dtype=x.dtype, ncomp=3)
#         w.closeElement("PPoints")

#         procs_with_sets = np.count_nonzero(np.array(proc_set_counts) > 0)
#         name = [local_file_proc] * procs_with_sets
#         for n in range(subdomain.num_subdomains):
#             if proc_set_counts[n] > 0:
#                 name[n] = name[n] + str(n) + ".vtu"

#         for s in name:
#             w.addPiece(start=None, end=None, source=s)
#         w.closeGrid()
#         w.save()


# def save_grid_data_deconstructed(file_name, coords, grid):
#     """Save grid data for a decomposed grid"""

#     io_utils.check_file_path(file_name)
#     point_data = {"Grid": grid}

#     gridToVTK(
#         file_name,
#         coords[0],
#         coords[1],
#         coords[2],
#         start=[0, 0, 0],
#         pointData=point_data,
#     )


# def save_grid_data_csv(file_name, subdomain, x, y, z, grid, remove_halo=False):
#     """_summary_

#     Args:
#         file_name (_type_): _description_
#         subdomain (_type_): _description_
#         x (_type_): _description_
#         y (_type_): _description_
#         z (_type_): _description_
#         grid (_type_): _description_
#         remove_halo (bool, optional): _description_. Defaults to False.
#     """
#     rank = subdomain.rank

#     if rank == 0:
#         io_utils.check_file_path(file_name)
#     comm.barrier()

#     if remove_halo:
#         own = subdomain.index_own_Nodes
#         size = (own[1] - own[0]) * (own[3] - own[2]) * (own[5] - own[4])
#         grid_out = np.zeros([size, 4])
#     else:
#         own = np.zeros([6], dtype=np.int64)
#         own[1] = grid.shape[0]
#         own[3] = grid.shape[1]
#         own[5] = grid.shape[2]
#         grid_out = np.zeros([grid.size, 4])

#     file_proc = file_name + "/" + file_name.split("/")[-1] + "Proc."

#     c = 0
#     for i in range(own[0], own[1]):
#         for j in range(own[2], own[3]):
#             for k in range(own[4], own[5]):
#                 grid_out[c, 0] = x[i]
#                 grid_out[c, 1] = y[j]
#                 grid_out[c, 2] = z[k]
#                 grid_out[c, 3] = grid[i, j, k]
#                 c = c + 1

#     header = "x,y,z,Grid"
#     np.savetxt(file_proc + str(rank) + ".csv", grid_out, delimiter=",", header=header)
