"""Module for PMMoTO  subDomains."""
import numpy as np
from mpi4py import MPI
from . import Orientation

comm = MPI.COMM_WORLD

class Subdomain(object):
    """Base class for PMMoTo
    """
    def __init__(self,ID,subdomains,domain):
        self.ID          = ID
        self.size        = np.prod(subdomains)
        self.subdomains  = subdomains
        self.domain      = domain
        self.boundary    = False
        self.boundary_ID  = -np.ones([6],dtype = np.int8)
        self.buffer      = np.ones([6],dtype = np.int8)
        self.nodes       = np.zeros([3],dtype = np.int64)
        self.own_nodes    = np.zeros([3],dtype = np.int64)
        self.index_own_nodes = np.zeros([6],dtype = np.int64)
        self.index_global = np.zeros([6],dtype = np.int64)
        self.index_start  = np.zeros([3],dtype = np.int64)
        self.size_subdomain = np.zeros([3])
        self.bounds = np.zeros([3,2])
        self.n_procs = {}
        self.sub_ID  = np.zeros([3],dtype = np.int64)
        self.faces = [None]*Orientation.num_faces
        self.edges = [None]*Orientation.num_edges
        self.corners = [None]*Orientation.num_corners
        self.coords = [None]*3


    def get_info(self):
        """
        Gather information for each subDomain including:
        ID, boundary information, number of nodes, global index start, buffer
        """
        n = 0
        for i in range(0,self.subdomains[0]):
            for j in range(0,self.subdomains[1]):
                for k in range(0,self.subdomains[2]):
                    if n == self.ID:
                        for nn,d in enumerate([i,j,k]):
                            self.sub_ID[nn] = d
                            self.nodes[nn] = self.domain.sub_nodes[nn]
                            self.own_nodes[nn] = self.domain.sub_nodes[nn]
                            self.index_start[nn] = d * self.domain.sub_nodes[nn]

                            if d == 0:
                                self.boundary = True
                                self.boundary_ID[nn*2] = self.domain.boundaries[nn][0]

                            if d == self.subdomains[nn] - 1:
                                self.boundary = True
                                self.boundary_ID[nn*2+1] = self.domain.boundaries[nn][1]
                                self.nodes[nn] += self.domain.rem_sub_nodes[d]
                                self.own_nodes[nn] += self.domain.rem_sub_nodes[d]

                    n = n + 1

        # If boundary_ID == 0, buffer is not added
        for f in range(0,Orientation.num_faces):
            if self.boundary_ID[f] == 0:
                self.buffer[f] = 0

    def get_coordinates(self, pad = None, get_coords = True, multiphase = False):
        """
        Determine actual coordinate information (x,y,z)
        If boundaryID and Domain.boundary == 0, buffer is not added
        Everywhere else a buffer is added
        Pad is also Reservoir Size for mulitPhase 
        """

        if pad is None:
            pad = self.buffer

        sd_size = [None,None]
        for n in range(self.domain.dims):
    
            self.nodes[n] += pad[n*2] + pad[n*2+1]
            self.index_start[n] -= pad[n*2]

            self.index_own_nodes[n*2] += pad[n*2]
            self.index_own_nodes[n*2+1] = self.index_own_nodes[n*2] +self.own_nodes[n]

            self.index_global[n*2] = self.index_start[n] + pad[n*2]
            self.index_global[n*2+1] = self.index_start[n] + self.nodes[n] - pad[n*2+1]

            if get_coords:
                vox = self.domain.voxel[n]
                d_size = self.domain.size_domain[n]
                self.coords[n] = np.zeros(self.nodes[n],dtype = np.double)
                sd_size[0] = vox/2 + d_size[0] + vox*(self.index_start[n])
                sd_size[1] = vox/2 + d_size[0] + vox*(self.index_start[n] + self.nodes[n] - 1)
                self.coords[n] = np.linspace(sd_size[0], sd_size[1], self.nodes[n] )
                
                self.size_subdomain[n] = sd_size[1] - sd_size[0]
                self.bounds[n] = [sd_size[0],sd_size[1]]

            ### Not Sure why I have this. Commenting out in case useful
            if multiphase:
                if pad[n*2] > 0:
                    self.domain.nodes[n] += pad[n*2]
                if pad[n*2+1] > 0:
                    self.domain.nodes[n] += pad[n*2+1]

    def update_domain_size(self,domain_data):
        """Use data from io to set domain size and determine voxel size and coordinates
        """
        self.domain.size_domain = domain_data
        self.domain.get_voxel_size()
        self.get_coordinates()


    def gather_cube_info(self):
        """
        Collect all necessary infromation for faces, edges, and corners as well as all neighbors
        """

        ### Faces
        for n in range(0,Orientation.num_faces):
            face = Orientation.faces[n]
            ID = face['ID']
            i = ID[0] + self.sub_ID[0] + 1
            j = ID[1] + self.sub_ID[1] + 1
            k = ID[2] + self.sub_ID[2] + 1
            n_proc = self.domain.global_map[i,j,k]
            if n_proc not in self.n_procs:
                self.n_procs[n_proc] = [ID]
            else:
                self.n_procs[n_proc].append(ID)

            ### Determine if Periodic Face or Periodic
            periodic = [0,0,0]
            boundary = False
            if self.boundary_ID[n] >= 0:
                boundary = True
                if self.boundary_ID[n] == 2:
                    periodic[face['argOrder'][0]] = face['dir']

            self.faces[n] = Orientation.Face(n,n_proc,boundary,periodic)

        ### Edges
        for n in range(0,Orientation.num_edges):
            ID = Orientation.edges[n]['ID']
            i = ID[0] + self.sub_ID[0] + 1
            j = ID[1] + self.sub_ID[1] + 1
            k = ID[2] + self.sub_ID[2] + 1
            n_proc = self.domain.global_map[i,j,k]
            if n_proc not in self.n_procs:
                self.n_procs[n_proc] = [ID]
            else:
                self.n_procs[n_proc].append(ID)

            ### Determine if Periodic Edge or Global Boundary Edge
            periodic = [0,0,0]
            boundary = False
            global_boundary =  False
            external_faces = []
            for n_face in Orientation.edges[n]['faceIndex']:
                if self.boundary_ID[n_face] >= 0:
                    boundary = True
                    if self.boundary_ID[n_face] == 2:
                        periodic[Orientation.faces[n_face]['argOrder'][0]] = Orientation.faces[n_face]['dir']
                    elif self.boundary_ID[n_face] == 0:
                        external_faces.append(n_face)

            if len(external_faces) == 2:
                global_boundary = True

            self.edges[n] = Orientation.Edge(n,n_proc,boundary,periodic,global_boundary,external_faces)

        ### Corners
        for n in range(0,Orientation.num_corners):
            ID = Orientation.corners[n]['ID']
            i = ID[0] + self.sub_ID[0] + 1
            j = ID[1] + self.sub_ID[1] + 1
            k = ID[2] + self.sub_ID[2] + 1
            n_proc = self.domain.global_map[i,j,k]
            if n_proc not in self.n_procs:
                self.n_procs[n_proc] = [ID]
            else:
                self.n_procs[n_proc].append(ID)

            ### Determine if Periodic Corner or Global Boundary Corner
            periodic = [0,0,0]
            boundary = False
            global_boundary =  False
            external_faces = []
            external_edges = []
            for n_face in Orientation.corners[n]['faceIndex']:
                if self.boundary_ID[n_face] == 2:
                    periodic[Orientation.faces[n_face]['argOrder'][0]] = Orientation.faces[n_face]['dir']
                elif self.boundary_ID[n_face] == 0:
                    boundary = True
                    external_faces.append(n_face)

            for n_edge,edge in enumerate(self.edges):
                if edge.boundary:
                    external_edges.append(n_edge)

            if len(external_faces) == 3 or len(external_edges) == 3:
                global_boundary = True

            self.corners[n] = Orientation.Corner(n,n_proc,boundary,periodic,global_boundary,external_faces,external_edges)
