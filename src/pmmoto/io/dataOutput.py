import os
import numpy as np
from mpi4py import MPI
from pyevtk.hl import (
    pointsToVTK,
    gridToVTK,
    writeParallelVTKGrid,
    _addDataToParallelFile,
)
from pyevtk import vtk
from pmmoto.core import communication
from . import io_utils

comm = MPI.COMM_WORLD

__all__ = [
    "save_grid_data",
    "save_grid_data_proc",
    "save_grid_data_deconstructed",
    "save_grid_data_csv",
    "save_set_data",
]


def save_grid_data(file_name, subdomain, grid, **kwargs):
    """Save grid data as vtk"""

    if subdomain.rank == 0:
        io_utils.check_file_path(file_name)
    comm.barrier()

    all_info = comm.gather([subdomain.start, grid.shape], root=0)

    file_proc = file_name + "/" + file_name.split("/")[-1] + "Proc."
    local_file_proc = (
        file_name.split("/")[-1] + "/" + file_name.split("/")[-1] + "Proc."
    )
    point_data = {"grid": grid}
    point_data_info = {"grid": (grid.dtype, 1)}
    for key, value in kwargs.items():
        point_data[key] = value
        point_data_info[key] = (value.dtype, 1)

    gridToVTK(
        file_proc + str(subdomain.rank),
        subdomain.coords[0],
        subdomain.coords[1],
        subdomain.coords[2],
        start=[
            subdomain.start[0],
            subdomain.start[1],
            subdomain.start[2],
        ],
        pointData=point_data,
    )

    if subdomain.rank == 0:
        name = [local_file_proc] * subdomain.num_subdomains
        starts = [[0, 0, 0] for _ in range(subdomain.num_subdomains)]
        ends = [[0, 0, 0] for _ in range(subdomain.num_subdomains)]
        domain_voxels = [[np.inf, np.inf, np.inf], [-np.inf, -np.inf, -np.inf]]
        for n in range(0, subdomain.num_subdomains):
            name[n] = name[n] + str(n) + ".vtr"
            for dim in range(subdomain.dims):
                starts[n][dim] = all_info[n][0][dim]
                ends[n][dim] = starts[n][dim] + all_info[n][1][dim] - 1

                if starts[n][dim] < domain_voxels[0][dim]:
                    domain_voxels[0][dim] = starts[n][dim]

                if ends[n][dim] > domain_voxels[1][dim]:
                    domain_voxels[1][dim] = ends[n][dim]

        writeParallelVTKGrid(
            file_name,
            coordsData=(
                (
                    domain_voxels[1][0] - domain_voxels[0][0],
                    domain_voxels[1][1] - domain_voxels[0][1],
                    domain_voxels[1][2] - domain_voxels[0][2],
                ),
                subdomain.coords[0].dtype,
            ),
            starts=starts,
            ends=ends,
            sources=name,
            pointData=point_data_info,
        )


def save_grid_data_proc(file_name, subdomains, grids):
    """Save grid data for a single process"""

    io_utils.check_file_path(file_name)
    file_proc = file_name + "/" + file_name.split("/")[-1] + "Proc."
    num_procs = len(subdomains)
    for n in range(0, num_procs):
        point_data = {"Grid": grids[n]}
        gridToVTK(
            file_proc + str(n),
            subdomains[n].coords[0],
            subdomains[n].coords[1],
            subdomains[n].coords[2],
            start=[0, 0, 0],
            pointData=point_data,
        )


def save_grid_data_deconstructed(file_name, coords, grid):
    """Save grid data for a decomposed grid"""

    io_utils.check_file_path(file_name)
    point_data = {"Grid": grid}

    gridToVTK(
        file_name,
        coords[0],
        coords[1],
        coords[2],
        start=[0, 0, 0],
        pointData=point_data,
    )


def save_grid_data_csv(file_name, subdomain, x, y, z, grid, remove_halo=False):
    """Save grid as csv. Warning this is not lightweight."""
    rank = subdomain.rank

    if rank == 0:
        io_utils.check_file_path(file_name)
    comm.barrier()

    if remove_halo:
        own = subdomain.index_own_Nodes
        size = (own[1] - own[0]) * (own[3] - own[2]) * (own[5] - own[4])
        grid_out = np.zeros([size, 4])
    else:
        own = np.zeros([6], dtype=np.int64)
        own[1] = grid.shape[0]
        own[3] = grid.shape[1]
        own[5] = grid.shape[2]
        grid_out = np.zeros([grid.size, 4])

    file_proc = file_name + "/" + file_name.split("/")[-1] + "Proc."

    c = 0
    for i in range(own[0], own[1]):
        for j in range(own[2], own[3]):
            for k in range(own[4], own[5]):
                grid_out[c, 0] = x[i]
                grid_out[c, 1] = y[j]
                grid_out[c, 2] = z[k]
                grid_out[c, 3] = grid[i, j, k]
                c = c + 1

    header = "x,y,z,Grid"
    np.savetxt(file_proc + str(rank) + ".csv", grid_out, delimiter=",", header=header)


def save_set_data(file_name, subdomain, set_list, **kwargs):
    """
    Save the set data as vtk.
    """

    rank = subdomain.rank

    if rank == 0:
        io_utils.check_file_path(file_name)
    comm.barrier()

    proc_set_counts = comm.allgather(set_list.count.all)
    nonzero_proc = np.where(np.asarray(proc_set_counts) > 0)[0][0]

    ### Place Set Values in Arrays
    if set_list.count.all > 0:
        dim = 0
        for local_ID, ss in set_list.sets.items():
            dim = dim + len(ss.node_data.nodes)
        x = np.zeros(dim)
        y = np.zeros(dim)
        z = np.zeros(dim)
        set_rank = rank * np.ones(dim, dtype=np.uint8)
        global_ID = np.zeros(dim, dtype=np.uint64)
        local_ID = np.zeros(dim, dtype=np.uint64)
        phase = np.ones(dim, dtype=np.uint8)
        point_data = {
            "set": set_rank,
            "globalID": global_ID,
            "localID": local_ID,
            "phase": phase,
        }
        point_data_info = {
            "set": (set_rank.dtype, 1),
            "globalID": (global_ID.dtype, 1),
            "localID": (local_ID.dtype, 1),
            "phase": (phase.dtype, 1),
        }

        ### Handle kwargs - need to fix for values that are subclasses!
        for key, value in kwargs.items():

            sub_classes = value.count(".")
            val_split = value.split(".")

            valid_data = True
            next = list(set_list.sets.values())[0]
            for n, child in enumerate(val_split):
                if hasattr(next, child):
                    next = getattr(next, child)
                else:
                    valid_data = False

            if not valid_data:
                if rank == 0:
                    print(
                        f"Error: Cannot save set data as kwarg {value} is not an attribute in Set"
                    )
                communication.raiseError()

            dataType = type(next)
            if dataType == bool:  ### pyectk does not support bool?
                dataType = np.uint8
            point_data[key] = np.zeros(dim, dtype=dataType)
            point_data_info[key] = (point_data[key].dtype, 1)

        c = 0
        for ss in set_list.sets.values():
            indexs = np.unravel_index(ss.node_data.nodes, ss.node_data.index_map)
            for index in zip(indexs[0], indexs[1], indexs[2]):
                x[c] = subdomain.coords[0][index[0]]
                y[c] = subdomain.coords[1][index[1]]
                z[c] = subdomain.coords[2][index[2]]

                global_ID[c] = ss.global_ID
                local_ID[c] = ss.local_ID
                phase[c] = ss.phase
                for key, value in kwargs.items():

                    val_split = value.split(".")
                    next = ss
                    for n, child in enumerate(val_split):
                        next = getattr(next, child)
                    point_data[key][c] = next
                c = c + 1

        file_proc = file_name + "/" + file_name.split("/")[-1] + "Proc."
        local_file_proc = (
            file_name.split("/")[-1] + "/" + file_name.split("/")[-1] + "Proc."
        )
        pointsToVTK(file_proc + str(rank), x, y, z, data=point_data)

    if rank == nonzero_proc:
        w = vtk.VtkParallelFile(file_name, vtk.VtkPUnstructuredGrid)
        w.openGrid()
        point_data = point_data_info
        _addDataToParallelFile(w, cellData=None, pointData=point_data)
        w.openElement("PPoints")
        w.addHeader("points", dtype=x.dtype, ncomp=3)
        w.closeElement("PPoints")

        procs_with_sets = np.count_nonzero(np.array(proc_set_counts) > 0)
        name = [local_file_proc] * procs_with_sets
        for n in range(subdomain.num_subdomains):
            if proc_set_counts[n] > 0:
                name[n] = name[n] + str(n) + ".vtu"

        for s in name:
            w.addPiece(start=None, end=None, source=s)
        w.closeGrid()
        w.save()
