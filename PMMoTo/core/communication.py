from . import Orientation
import numpy as np
from mpi4py import MPI
import sys
comm = MPI.COMM_WORLD

def raiseError():
    MPI.Finalize()
    sys.exit()

def update_buffer(subdomain,grid):
    """
    """
    buffer_data = buffer_pack(subdomain,grid)
    f,e,c = communicate(subdomain,buffer_data)
    buffer_grid = buffer_unpack(subdomain,grid,f,e,c)
    return buffer_grid

def generate_halo(subdomain,grid,halo):
    """
    """
    halo_data,halo_out = halo_pack(subdomain,grid,halo)
    f,e,c = communicate(subdomain,halo_data)
    halo_grid = halo_unpack(subdomain,grid,halo_out,f,e,c)
    return halo_grid,halo_out

def pass_external_data(subdomain,face_solids,edge_solids,corner_solids):
    """
    """
    external_data = external_nodes_pack(subdomain,face_solids,edge_solids,corner_solids)
    f,e,c = communicate(subdomain,external_data)
    external_solids = external_nodes_unpack(subdomain,f,e,c,face_solids,edge_solids,corner_solids)
    return external_solids


def buffer_pack(subdomain,grid):
    """
    Grab The Slices (based on Buffer Size [1,1,1]) to pack and send to Neighbors
    for faces, edges and corners.
    """

    send_faces,send_edges,send_corners = Orientation.get_send_buffer(subdomain.buffer,grid.shape)

    buffer_data = {}
    for n_proc in subdomain.n_procs:
        if n_proc > -1 and n_proc != subdomain.ID:
            buffer_data[n_proc] = {'ID':{}}
            for feature in subdomain.n_procs[n_proc]:
                buffer_data[n_proc]['ID'][feature] = None

    slices = send_faces
    for face in subdomain.faces:
        if face.n_proc > -1 and face.n_proc != subdomain.ID:
            s = slices[face.ID,:]
            buffer_data[face.n_proc]['ID'][face.info['ID']] = grid[s[0],s[1],s[2]]

    slices = send_edges
    for edge in subdomain.edges:
        if edge.n_proc > -1 and edge.n_proc != subdomain.ID:
            s = slices[edge.ID,:]
            buffer_data[edge.n_proc]['ID'][edge.info['ID']] = grid[s[0],s[1],s[2]]

    slices = send_corners
    for corner in subdomain.corners:
        if corner.n_proc > -1 and corner.n_proc != subdomain.ID:
            s = slices[corner.ID,:]
            buffer_data[corner.n_proc]['ID'][corner.info['ID']] = grid[s[0],s[1],s[2]]

    return buffer_data

def buffer_unpack(subdomain,grid,face_recv,edge_recv,corner_recv):
    """
    Unpack buffer information and account for serial periodic boundary conditions
    """
    buffer_grid = np.copy(grid)
    recv_faces,recv_edges,recv_corners = Orientation.get_recv_buffer(subdomain.buffer,buffer_grid.shape)
    send_faces,send_edges,send_corners = Orientation.get_send_buffer(subdomain.buffer,grid.shape)

    #### Faces ####
    r_slices = recv_faces
    s_slices = send_faces
    for face in subdomain.faces:
        if (face.n_proc > -1 and face.n_proc != subdomain.ID):
            r_s = r_slices[face.ID,:]
            buffer_grid[r_s[0],r_s[1],r_s[2]] = face_recv[face.ID]['ID'][face.opp_info['ID']]
        elif face.n_proc == subdomain.ID:
            r_s = r_slices[face.ID,:]
            s_s = s_slices[face.info['oppIndex'],:]
            buffer_grid[r_s[0],r_s[1],r_s[2]] = grid[s_s[0],s_s[1],s_s[2]]

    #### Edges ####
    r_slices = recv_edges
    s_slices = send_edges
    for edge in subdomain.edges:
        if (edge.n_proc > -1 and edge.n_proc != subdomain.ID):
            r_s = r_slices[edge.ID,:]
            buffer_grid[r_s[0],r_s[1],r_s[2]] = edge_recv[edge.ID]['ID'][edge.opp_info['ID']]
        elif edge.n_proc == subdomain.ID:
            r_s = r_slices[edge.ID,:]
            s_s = s_slices[edge.info['oppIndex'],:]
            buffer_grid[r_s[0],r_s[1],r_s[2]] = grid[s_s[0],s_s[1],s_s[2]]

    #### Corners ####
    r_slices = recv_corners
    s_slices = send_corners
    for corner in subdomain.corners:
        if (corner.n_proc > -1 and corner.n_proc != subdomain.ID):
            r_s = r_slices[corner.ID,:]
            buffer_grid[r_s[0],r_s[1],r_s[2]] = corner_recv[corner.ID]['ID'][corner.opp_info['ID']]
        elif corner.n_proc == subdomain.ID:
            r_s = r_slices[corner.ID,:]
            s_s = s_slices[corner.info['oppIndex'],:]
            buffer_grid[r_s[0],r_s[1],r_s[2]] = grid[s_s[0],s_s[1],s_s[2]]
    
    return buffer_grid

def halo_pack(subdomain,grid,halo):
    """
    Grab The Slices (based on Size) to pack and send to Neighbors
    for faces, edges and corners
    """

    send_faces,send_edges,send_corners = Orientation.get_send_halo(halo,subdomain.buffer,grid.shape)

    halo_out = np.zeros(Orientation.num_faces,dtype=int)
    for face in subdomain.faces:
        if face.n_proc > -1:
            halo_out[face.ID] = halo[face.ID]

    halo_data = {}
    for n_proc in subdomain.n_procs:
        if n_proc > -1 and n_proc != subdomain.ID:
            halo_data[n_proc] = {'ID':{}}
            for feature in subdomain.n_procs[n_proc]:
                halo_data[n_proc]['ID'][feature] = None

    slices = send_faces
    for face in subdomain.faces:
        if face.n_proc > -1 and face.n_proc != subdomain.ID:
            s = slices[face.ID,:]
            halo_data[face.n_proc]['ID'][face.info['ID']] = grid[s[0],s[1],s[2]]

    slices = send_edges
    for edge in subdomain.edges:
        if edge.n_proc > -1 and edge.n_proc != subdomain.ID:
            s = slices[edge.ID,:]
            halo_data[edge.n_proc]['ID'][edge.info['ID']] = grid[s[0],s[1],s[2]]

    slices = send_corners
    for corner in subdomain.corners:
        if corner.n_proc > -1 and corner.n_proc != subdomain.ID:
            s = slices[corner.ID,:]
            halo_data[corner.n_proc]['ID'][corner.info['ID']] = grid[s[0],s[1],s[2]]

    return halo_data,halo_out

def halo_unpack(subdomain,grid,halo,face_recv,edge_recv,corner_recv):
    """
    """
    if all(halo == 0):
        halo_grid = grid
    else:
        halo_grid = np.pad(grid,((halo[0], halo[1]),(halo[2], halo[3]),(halo[4], halo[5])), 'constant', constant_values=255)
        
        recv_faces,recv_edges,recv_corners = Orientation.get_recv_halo(halo,subdomain.buffer,halo_grid.shape)
        send_faces,send_edges,send_corners = Orientation.get_send_halo(halo,subdomain.buffer,grid.shape)

        #### Faces ####
        r_slices = recv_faces
        s_slices = send_faces
        for face in subdomain.faces:
            if (face.n_proc > -1 and face.n_proc != subdomain.ID):
                r_s = r_slices[face.ID,:]
                halo_grid[r_s[0],r_s[1],r_s[2]] = face_recv[face.ID]['ID'][face.opp_info['ID']]
            elif face.n_proc == subdomain.ID:
                r_s = r_slices[face.ID,:]
                s_s = s_slices[face.info['oppIndex'],:]
                halo_grid[r_s[0],r_s[1],r_s[2]] = grid[s_s[0],s_s[1],s_s[2]]

        #### Edges ####
        r_slices = recv_edges
        s_slices = send_edges
        for edge in subdomain.edges:
            if (edge.n_proc > -1 and edge.n_proc != subdomain.ID):
                r_s = r_slices[edge.ID,:]
                halo_grid[r_s[0],r_s[1],r_s[2]] = edge_recv[edge.ID]['ID'][edge.opp_info['ID']]
            elif edge.n_proc == subdomain.ID:
                r_s = r_slices[edge.ID,:]
                s_s = s_slices[edge.info['oppIndex'],:]
                halo_grid[r_s[0],r_s[1],r_s[2]] = grid[s_s[0],s_s[1],s_s[2]]

        #### Corners ####
        r_slices = recv_corners
        s_slices = send_corners
        for corner in subdomain.corners:
            if (corner.n_proc > -1 and corner.n_proc != subdomain.ID):
                r_s = r_slices[corner.ID,:]
                halo_grid[r_s[0],r_s[1],r_s[2]] = corner_recv[corner.ID]['ID'][corner.opp_info['ID']]
            elif corner.n_proc == subdomain.ID:
                r_s = r_slices[corner.ID,:]
                s_s = s_slices[corner.info['oppIndex'],:]
                halo_grid[r_s[0],r_s[1],r_s[2]] = grid[s_s[0],s_s[1],s_s[2]]
    
    return halo_grid

def external_nodes_pack(subdomain,face_solids,edge_solids,corner_solids):
    """
    Send boundary solids data. 
    external_data[neighbor_proc_ID]['ID'][(0,0,0)] = boundary_nodes[feature.ID]
    """
    external_data = {}
    for n_proc in subdomain.n_procs:
        if n_proc != subdomain.ID:
            external_data[n_proc] = {'ID':{}}
            for feature in subdomain.n_procs[n_proc]:
                external_data[n_proc]['ID'][feature] = None
    
    for face in subdomain.faces:
        if face.n_proc != subdomain.ID:
            external_data[face.n_proc]['ID'][face.info['ID']] = face_solids[face.ID]

    for edge in subdomain.edges:
        if edge.n_proc != subdomain.ID:
            external_data[edge.n_proc]['ID'][edge.info['ID']] = edge_solids[edge.ID]

    for corner in subdomain.corners:
        if corner.n_proc != subdomain.ID:
            external_data[corner.n_proc]['ID'][corner.info['ID']] = corner_solids[corner.ID]
    
    return external_data

def external_nodes_unpack(subdomain,face_recv,edge_recv,corner_recv,face_solids,edge_solids,corner_solids):
    """
    Recieve boundary solids data.
    """

    external_solids = {key: None for key in Orientation.features}

    #### FACE ####
    for face in subdomain.faces:
        period_correction = face.periodic_correction*subdomain.domain.length_domain
        if (face.n_proc > -1 and face.n_proc != subdomain.ID):
            opp_ID = face.opp_info['ID']
            external_solids[face.info['ID']] = face_recv[face.ID]['ID'][opp_ID] - period_correction
        elif (face.n_proc == subdomain.ID):
            external_solids[face.info['ID']] = face_solids[face.info['oppIndex']] - period_correction

    #### EDGE ####
    for edge in subdomain.edges:
        period_correction = edge.periodic_correction*subdomain.domain.length_domain
        if (edge.n_proc > -1 and edge.n_proc != subdomain.ID):
            opp_ID = edge.opp_info['ID']
            external_solids[edge.info['ID']] = edge_recv[edge.ID]['ID'][opp_ID] - period_correction
        elif (edge.n_proc == subdomain.ID):
            external_solids[edge.info['ID']] = edge_solids[edge.info['oppIndex']] - period_correction

    #### Corner ####
    for corner in subdomain.corners:
        period_correction = corner.periodic_correction*subdomain.domain.length_domain
        if (corner.n_proc > -1 and corner.n_proc != subdomain.ID):
            opp_ID = corner.opp_info['ID']
            external_solids[corner.info['ID']] = corner_recv[corner.ID]['ID'][opp_ID] - period_correction
        elif (corner.n_proc == subdomain.ID):
            external_solids[corner.info['ID']] = corner_solids[corner.info['oppIndex']] - period_correction

    return external_solids

def communicate(subdomain,send_data):
    """
    Send data between processess
    """

    #### FACES ####
    reqs = [None]*Orientation.num_faces
    reqr = [None]*Orientation.num_faces
    recv_face = [None]*Orientation.num_faces
    for face in subdomain.faces:
        opp_neigh = subdomain.faces[face.info['oppIndex']].n_proc
        if (opp_neigh > -1 and face.n_proc != subdomain.ID and opp_neigh in send_data.keys() ):
            reqs[face.ID] = comm.isend(send_data[opp_neigh],dest=opp_neigh)
        if (face.n_proc > -1 and face.n_proc != subdomain.ID and face.n_proc in send_data.keys() ):
            reqr[face.ID] = comm.recv(source=face.n_proc)

    reqs = [i for i in reqs if i]
    MPI.Request.waitall(reqs)

    for face in subdomain.faces:
        if (face.n_proc > -1 and face.n_proc != subdomain.ID and face.n_proc in send_data.keys() ):
            recv_face[face.ID] = reqr[face.ID]

    #### EDGES ####
    reqs = [None]*Orientation.num_edges
    reqr = [None]*Orientation.num_edges
    recv_edge = [None]*Orientation.num_edges
    for edge in subdomain.edges:
        opp_neigh = subdomain.edges[edge.info['oppIndex']].n_proc
        if (opp_neigh > -1 and edge.n_proc != subdomain.ID and opp_neigh in send_data.keys() ):
            reqs[edge.ID] = comm.isend(send_data[opp_neigh],dest = opp_neigh)
        if (edge.n_proc  > -1 and edge.n_proc  != subdomain.ID and edge.n_proc in send_data.keys() ):
            reqr[edge.ID] = comm.recv(source = edge.n_proc )

    reqs = [i for i in reqs if i]
    MPI.Request.waitall(reqs)

    for edge in subdomain.edges:
        if (edge.n_proc > -1 and edge.n_proc != subdomain.ID and edge.n_proc in send_data.keys() ):
            recv_edge[edge.ID] = reqr[edge.ID]

    #### CORNERS ####
    reqs = [None]*Orientation.num_corners
    reqr = [None]*Orientation.num_corners
    recv_corner = [None]*Orientation.num_corners
    for corner in subdomain.corners:
        opp_neigh = subdomain.corners[corner.info['oppIndex']].n_proc
        if (opp_neigh > -1 and corner.n_proc != subdomain.ID and opp_neigh in send_data.keys() ):
            reqs[corner.ID] = comm.isend(send_data[opp_neigh],dest = opp_neigh)
        if (corner.n_proc > -1 and corner.n_proc != subdomain.ID and corner.n_proc in send_data.keys() ):
            reqr[corner.ID] = comm.recv(source=corner.n_proc)

    reqs = [i for i in reqs if i]
    MPI.Request.waitall(reqs)

    for corner in subdomain.corners:
        if (corner.n_proc > -1 and corner.n_proc != subdomain.ID and corner.n_proc in send_data.keys() ):
            recv_corner[corner.ID] = reqr[corner.ID]

    return recv_face,recv_edge,recv_corner

def set_COMM(subDomain,data):
  """
  Transmit data to Neighboring Processors
  """
  dataRecvFace,dataRecvEdge,dataRecvCorner = communicate(subDomain,data[subDomain.ID]['nProcID'])

  #############
  ### Faces ###
  #############
  for fIndex in Orientation.faces:
    neigh = subDomain.neighborF[fIndex]
    if (neigh > -1 and neigh != subDomain.ID):
      if neigh in data[subDomain.ID]['nProcID'].keys():
        if neigh not in data:
          data[neigh] = {'nProcID':{}}
        data[neigh]['nProcID'][neigh] = dataRecvFace[fIndex]

  #############
  ### Edges ###
  #############
  for eIndex in Orientation.edges:
    neigh = subDomain.neighborE[eIndex]
    if (neigh > -1 and neigh != subDomain.ID):
      if neigh in data[subDomain.ID]['nProcID'].keys():
        if neigh not in data:
          data[neigh] = {'nProcID':{}}
        data[neigh]['nProcID'][neigh] = dataRecvEdge[eIndex]

  ###############
  ### Corners ###
  ###############
  for cIndex in Orientation.corners:
    neigh = subDomain.neighborC[cIndex]
    if (neigh > -1 and neigh != subDomain.ID):
      if neigh in data[subDomain.ID]['nProcID'].keys():
        if neigh not in data:
          data[neigh] = {'nProcID':{}}
        data[neigh]['nProcID'][neigh] = dataRecvCorner[cIndex]

  return data