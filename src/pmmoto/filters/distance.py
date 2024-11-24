import numpy as np
from mpi4py import MPI

# from pykdtree.kdtree import KDTree

### if using WSL, uncomment line below.
from scipy.spatial import KDTree

from edt import edt3d
from pmmoto.core import communication
from pmmoto.core import nodes
from pmmoto.core import utils


comm = MPI.COMM_WORLD

__all__ = ["calc_edt"]


# def _fix_interface(edt, subdomain, solids, external_solids):
#     """
#     Loop through faces and correct distance with external_solids
#     """
#     visited = np.zeros_like(edt, dtype=np.uint8)
#     for face in subdomain.features["faces"]:
#         if face.n_proc > -1:
#             arg = face.info["argOrder"]
#             data = np.copy(external_solids[face.info["ID"]])

#             for edge in subdomain.features["edges"]:
#                 for f in edge.info["faceIndex"]:
#                     if f == face.ID and edge.n_proc > -1:
#                         data = np.append(data, external_solids[edge.info["ID"]], axis=0)

#             for corner in subdomain.features["corners"]:
#                 for f in corner.info["faceIndex"]:
#                     if f == face.ID and corner.n_proc > -1:
#                         data = np.append(
#                             data, external_solids[corner.info["ID"]], axis=0
#                         )

#             tree = KDTree(data)
#             face_solids = solids[np.where(solids[:, 3] == face.ID)][:, 0:3]
#             edt, visited = nodes.fixInterfaceCalc(
#                 tree,
#                 edt.shape[arg[0]],
#                 face.info["dir"],
#                 face_solids,
#                 edt,
#                 visited,
#                 min(subdomain.resolution),
#                 subdomain.coords,
#                 arg,
#             )
#     return edt


def calc_edt(domain, subdomain, grid):
    """Calculate the Euclidean Distance Transform from 0 voxels."""
    size = subdomain.num_subdomains
    edt = edt3d(grid, anisotropy=subdomain.resolution)
    # if size > 1 or (size == 1 and any(subdomain.boundaries == 2)):
    #     solids = nodes.get_phase_boundary_nodes(grid, 0)
    #     face_solids, edge_solids, corner_solids = utils.partition_boundary_solids(
    #         subdomain, solids
    #     )
    #     external_solids = communication.pass_external_data(
    #         domain, subdomain, face_solids, edge_solids, corner_solids
    #     )
    #     edt = _fix_interface(edt, subdomain, solids, external_solids)
    #     edt = communication.update_buffer(subdomain, edt)

    return edt
