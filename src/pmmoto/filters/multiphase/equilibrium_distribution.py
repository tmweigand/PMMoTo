import numpy as np
from mpi4py import MPI

from pmmoto.filters import distance
from pmmoto.filters import morphology
from pmmoto.filters import connect_all_phases
from pmmoto.core import multiphase
from pmmoto.io import dataOutput
from pmmoto.core import utils
from pmmoto.core import _nodes

comm = MPI.COMM_WORLD

__all__ = ["calcOpenSW"]


class EquilibriumDistribution(object):
    def __init__(self, multiphase):
        self.multiphase = multiphase
        self.subdomain = multiphase.subdomain
        self.porousmedia = multiphase.porousmedia
        self.gamma = 1
        self.d_probe = 0
        self.r_probe = 0
        self.pc = 0

    def get_inlet_connected_nodes(self, Sets, flag):
        """
        Grab from Sets that are on the Inlet Reservoir and create binary grid
        """
        nodes = []
        _grid_out = np.zeros_like(self.multiphase.mp_grid)

        for s in Sets.sets:
            if s.inlet:
                for node in s.nodes:
                    nodes.append(node)

        for n in nodes:
            _grid_out[n[0], n[1], n[2]] = flag

        return _grid_out

    def get_disconnected_nodes(self, sets, flag):
        """
        Grab from Sets that are on the Inlet Reservoir and create binary grid
        """
        nodes = []
        _grid_out = np.zeros_like(self.multiphase.mp_grid)

        for s in sets:
            if not s.inlet:
                for node in s.nodes:
                    nodes.append(node)

        for n in nodes:
            _grid_out[n[0], n[1], n[2]] = flag

        return _grid_out

    def remove_small_sets(self, sets, grid_in, nw_ID, min_set_size):
        """
        Remove sets smaller than target size
        """
        nodes = []
        _grid_out = np.copy(grid_in)

        for s in sets:
            # print(s.numNodes)
            if s.numGlobalNodes < min_set_size:
                for node in s.nodes:
                    nodes.append(node)

        for n in nodes:
            _grid_out[n[0], n[1], n[2]] = nw_ID

        return _grid_out


def calcOpenSW(
    domain, subdomain, _multiphase, saturation, interval, minimum_set_size, save=False
):

    nw_id = 1
    w_id = 2

    if subdomain.rank == 0:
        print("Running opening without CA control and with saturation targets")

    ### Get Distance from Solid to Pore Space (Ignore Fluid Phases)
    pore_dist = distance.calc_edt(domain, subdomain, _multiphase.pm_grid)

    ## Make sure sw targets are ordered largest to smallest
    saturation.sort(reverse=True)

    ## Find initial radius target (largest EDT value)
    rad_temp = np.amax(pore_dist)
    rad = np.array([rad_temp])
    comm.Allreduce(MPI.IN_PLACE, rad, op=MPI.MAX)

    min_rad = np.min(subdomain.resolution) / 2.0
    saturation_new = 1.0

    for sat in saturation:

        while saturation_new > sat and rad[0] > min_rad:

            if rad[0] > 0:
                p_c = multiphase.get_pc(rad[0], 1)
            else:
                p_c = 0

            ### Get Sphere Radius from Pressure
            probe_radius = multiphase.get_probe_radius(p_c, gamma=1)

            # Step 2 - Dilate Solid Phase and Flag Allowable Fluid Voxes as 1
            ind = np.where(
                (pore_dist >= probe_radius) & (_multiphase.pm_grid == 1), 1, 0
            ).astype(np.uint8)

            # Step 3 - Check if Points were Marked
            # continueFlag = eqDist.checkPoints(ind, 1, True)
            if utils.phase_exists(ind, 1):

                # Step 3g
                morph = morphology.dilate(subdomain, ind, probe_radius)
                # morph = morphology.morph(ind, mP.subDomain, eqDist.probeR)

                print(np.sum(morph))

                dataOutput.save_grid_data("dataOut/test_open_morph", subdomain, morph)

                _multiphase.grid = np.where(
                    (morph == 1), nw_id, _multiphase.grid
                ).astype(np.uint8)

                if minimum_set_size > 0:
                    if utils.phase_exists(_multiphase.grid, w_id):

                        saturation_new = _multiphase.get_saturation(nw_id)

                        dataOutput.save_grid_data(
                            "dataOut/test_open_connect", subdomain, _multiphase.grid
                        )

                        connected = connect_all_phases(
                            _multiphase,
                            return_grid=True,
                            return_set=False,
                            return_voxel_count=True,
                        )

                        labeled_grid = connected["grid"]
                        voxel_count = connected["voxel_count"]
                        label_to_phase_map = connected["phase_map"]

                        dataOutput.save_grid_data(
                            "dataOut/test_open_label", subdomain, labeled_grid
                        )

                        print(label_to_phase_map)

                        remove_small_sets(
                            minimum_set_size, voxel_count, label_to_phase_map, 1, 2
                        )

                        print(label_to_phase_map)

                        _nodes.renumber_grid(labeled_grid, label_to_phase_map)
                        _multiphase.grid = labeled_grid

                        dataOutput.save_grid_data(
                            "dataOut/test_open_new_mp_grid", subdomain, _multiphase.grid
                        )

                        print(np.sum(np.where(_multiphase.grid == 1)))

                        raise ValueError

            # Step 4
            saturation_new = _multiphase.get_saturation(w_id)
            # sW_new = eqDist.calcSaturation(mP.mpGrid, mP.nwID)

            if subdomain.rank == 0:
                if saturation_new <= sat:
                    print(
                        "SAVE Capillary pressure: %e Radius: %e Wetting Phase Saturation: %e Target Saturation: %e"
                        % (p_c, rad[0], saturation_new, sat)
                    )

                else:
                    print(
                        "SKIP Capillary pressure: %e Radius: %e Wetting Phase Saturation: %e Target Saturation: %e"
                        % (p_c, rad[0], saturation_new, sat)
                    )

            rad[0] *= interval

        if save:
            fileName = "dataOut/Open/twoPhase_open_sw_" + str(s)
            # dataOutput.saveGrid(fileName, mP.subDomain, mP.mpGrid)

            fileName = "dataOut/OpenRAW/twoPhase_open_sw_" + str(s)
            # dataOutput.saveGridraw(fileName, mP.subDomain, mP.mpGrid)

    return _multiphase


def remove_small_sets(
    minimum_set_size, voxel_count, label_to_phase_map, phase_in, phase_out
):
    """
    Check the voxel counts per label along with label to phase map to remove small sets

    Args:
        grid (_type_): _description_
        in_phase (_type_): _description_
        out_phase (_type_): _description_
        minimum_set_size (_type_): _description_
    """
    for label, count in voxel_count.items():
        if label_to_phase_map[label] == phase_in and count < minimum_set_size:
            label_to_phase_map[label] = phase_out
