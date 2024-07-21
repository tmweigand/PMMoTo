from typing import Literal
import numpy as np


class Domain:
    """
    Information for domain including:
        size_domain: Size of the domain in physical units
        boundaries:  0: No assumption made
                     1: Wall boundary condition
                     2: Periodic boundary condition
                        Opposing boundary must also be 2
        inlet: True/False boundary must be 0
        outlet: True/False boundary must be 0

    """

    def __init__(
        self,
        box: np.ndarray[Literal[2], np.dtype[np.float64]],
        boundaries: tuple[tuple[int, int], ...] = ((0, 0), (0, 0), (0, 0)),
        inlet: tuple[tuple[int, int], ...] = ((0, 0), (0, 0), (0, 0)),
        outlet: tuple[tuple[int, int], ...] = ((0, 0), (0, 0), (0, 0)),
    ):
        self.box = box
        self.boundaries = boundaries
        self.inlet = inlet
        self.outlet = outlet
        self.dims = 3
        self.periodic = self.periodic_check()
        self.length = self.get_length()

    def get_length(self) -> tuple[float, ...]:
        """
        Calculate the length of the domain
        """
        length = np.zeros([self.dims], dtype=np.float64)
        for n in range(0, self.dims):
            length[n] = self.box[n, 1] - self.box[n, 0]

        return tuple(length)

    def periodic_check(self) -> bool:
        """
        Check if any external boundary is periodic boundary
        """
        periodic = False
        for d_bound in self.boundaries:
            for n_bound in d_bound:
                if n_bound == 2:
                    periodic = True
        return periodic

    def update_domain_size(self):
        """
        Use data from io to set domain size and determine voxel size and coordinates
        """
        pass

    # def generate_global_map(self):
    #     """
    #     Generate Domain lookup map.
    #     -2: Wall Boundary Condition
    #     -1: No Assumption Boundary Condition
    #     >=0: proc_ID
    #     """

    #     self.global_map[1:-1,1:-1,1:-1] = np.arange(self.num_subdomains).reshape(self.subdomains)

    #     ### Set Boundaries of global SubDomain Map
    #     if self.boundaries[0][0] == 1:
    #         self.global_map[0,:,:] = -2
    #     if self.boundaries[0][1] == 1:
    #         self.global_map[-1,:,:] = -2
    #     if self.boundaries[1][0] == 1:
    #         self.global_map[:,0,:] = -2
    #     if self.boundaries[1][1] == 1:
    #         self.global_map[:,-1,:] = -2
    #     if self.boundaries[2][0] == 1:
    #         self.global_map[:,:,0] = -2
    #     if self.boundaries[2][1] == 1:
    #         self.global_map[:,:,-1] = -2

    #     if self.boundaries[0][0] == 2:
    #         self.global_map[0,:,:]  = self.global_map[-2,:,:]
    #         self.global_map[-1,:,:] = self.global_map[1,:,:]

    #     if self.boundaries[1][0] == 2:
    #         self.global_map[:,0,:]  = self.global_map[:,-2,:]
    #         self.global_map[:,-1,:] = self.global_map[:,1,:]

    #     if self.boundaries[2][0] == 2:
    #         self.global_map[:,:,0]  = self.global_map[:,:,-2]
    #         self.global_map[:,:,-1] = self.global_map[:,:,1]

    # def get_coords(self):
    #     """
    #     Calculate the Physical coordinaties of the voxels for entire domain
    #     """
    #     for n in range(self.dims):
    #         self.coords[n] = np.linspace(
    #             self.size_domain[n][0]+self.voxel[n]/2.,
    #             self.size_domain[n][1]-self.voxel[n]/2.,
    #             self.nodes[n] )
