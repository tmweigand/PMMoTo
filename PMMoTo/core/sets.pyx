# distutils: language = c++
# cython: profile=True
# cython: linetrace=True

import numpy as np
cimport numpy as cnp
cimport cython
from mpi4py import MPI
comm = MPI.COMM_WORLD
from . import communication
from . import nodes
from . import set

from libcpp.vector cimport vector
from libcpp.utility cimport pair
from numpy cimport npy_intp
from libcpp cimport bool
from libcpp.map cimport map
from libcpp.unordered_map cimport unordered_map

from . import Orientation
cOrient = Orientation.cOrientation()
cdef int[26][5] directions
cdef int numNeighbors
directions = cOrient.directions
numNeighbors = cOrient.num_neighbors

class Sets(object):
  def __init__(self,
                 sets = None,
                 setCount = 0,
                 subDomain = None):
    self.sets = sets
    self.setCount = setCount
    self.subDomain = subDomain
    self.boundary_sets = []
    self.boundarySetCount = 0
    self.numConnections = 0
    self.localToGlobal = {}
    self.globalToLocal = {}

  def get_boundary_sets(self):
    """
    Get the Sets the are on a valid subDomain Boundary.
    Organize data so sending procID, boundary nodes.
    """

    nI = self.subDomain.subID[0] + 1  # PLUS 1 because lookUpID is Padded
    nJ = self.subDomain.subID[1] + 1  # PLUS 1 because lookUpID is Padded
    nK = self.subDomain.subID[2] + 1  # PLUS 1 because lookUpID is Padded

    for set in self.sets:
      procList = []
      if set.boundary:

        for face in range(0,numNeighbors):
          if set.boundaryFaces[face] > 0:
            i = directions[face][0]
            j = directions[face][1]
            k = directions[face][2]

            neighborProc = self.subDomain.lookUpID[i+nI,j+nJ,k+nK]
            if neighborProc < 0:
              set.boundaryFaces[face] = 0
            else:
              if neighborProc not in procList:
                procList.append(neighborProc)

        if (np.sum(set.boundaryFaces) == 0):
          set.boundary = False
        else:
          self.boundarySetCount += 1
          set.neighborProcID = procList
          self.boundary_sets.append( c_convert_boundary_set(set) )

  def pack_boundary_data(self):
    """
    Collect the Boundary Set Information to Send to neighbor procs
    """
    send_boundary_data = {self.subDomain.ID: {'nProcID':{}}}
    cdef boundary_set b_set
    cdef int nP,ID

    ID = self.subDomain.ID

    for set in self.boundary_sets:
        for nP in set['nProcID']:
            if nP not in send_boundary_data[ID]['nProcID'].keys():
                send_boundary_data[ID]['nProcID'][nP] = {'setID':{}}
            bD = send_boundary_data[ID]['nProcID'][nP]
            bD['setID'][set['ID']] = set

    return send_boundary_data

  def unpack_boundary_data(self,boundary_data):
    """
    Unpack the boundary data into neighborBoundarySets
    """
    n_boundary_sets = []
    for n_proc_ID in boundary_data[self.subDomain.ID]['nProcID'].keys():
      if n_proc_ID == self.subDomain.ID:
        pass
      else:
        for set in boundary_data[n_proc_ID]['nProcID'][n_proc_ID]['setID'].keys():
          n_boundary_sets.append(boundary_data[n_proc_ID]['nProcID'][n_proc_ID]['setID'][set])

    return n_boundary_sets

  def match_boundary_sets(self,n_boundary_data):
    """
    Loop through own boundary and neighbor boundary procs and match by boundary nodes
    IDEA: Create a ledger of n_boundary_sets.boundary_nodes and setID to speed this up
    """
    cdef int set_count,num_b_sets,num_neigh_b_sets,n_set
    cdef bool match
    cdef vector[boundary_set] boundary_sets,n_boundary_sets
    cdef vector[vector[npy_intp]] all_matches
    cdef vector[npy_intp] matches
    cdef boundary_set b_set

    boundary_sets = self.boundary_sets
    n_boundary_sets = n_boundary_data

    num_neigh_b_sets = len(n_boundary_data)
    num_b_sets = self.boundarySetCount

    for n_set in range(num_b_sets):
        b_set = boundary_sets[n_set]
        set_count = 0
        match = False
        while set_count < num_neigh_b_sets:
            if match_boundary_nodes(b_set.boundary_nodes,n_boundary_sets[set_count].boundary_nodes):
                match = True
                matches.push_back(set_count)
            
            set_count += 1
        
        if not match:
            print("ERROR Boundary Set Did Not Find a Match. Exiting...")
            communication.raiseError()

        all_matches.push_back(matches)
        matches.clear()

    return all_matches

  def get_num_global_nodes(self,all_matches,n_boundary_data):
    """
    Update the number of global nodes due to double counting the buffer nodes
    """
    cdef int n,m,count,ID
    cdef vector[boundary_set] boundary_sets,n_boundary_sets
    cdef vector[npy_intp] match
    boundary_sets = self.boundary_sets
    n_boundary_sets = n_boundary_data

    ID = self.subDomain.ID

    for n,match in enumerate(all_matches):
        for m in match:
            if ID < n_boundary_sets[m].proc_ID:
                boundary_sets[n].num_global_nodes = boundary_sets[n].num_nodes
            else:
                count = count_matched_nodes(boundary_sets[n].boundary_nodes,n_boundary_sets[m].boundary_nodes)
                boundary_sets[n].num_global_nodes = boundary_sets[n].num_nodes - count

  def pack_matched_sets(self,all_matches,n_boundary_data):
    """
    Pack the matched boundary set data to globally update 
    """
    cdef int n_set,num_b_sets,ID,c_set
    cdef vector[npy_intp] match
    cdef vector[boundary_set] boundary_sets,n_boundary_sets
    cdef boundary_set b_set,m_set
    cdef matched_set m_set_data

    boundary_sets = self.boundary_sets
    n_boundary_sets = n_boundary_data
    num_b_sets = self.boundarySetCount
    ID = self.subDomain.ID

    send_matched_set_data = {ID: {}}
    send_matched_set_data[ID] = {'setID':{}}

    for n,match in enumerate(all_matches):

        ### Set Own Properties
        b_set = boundary_sets[n]
        m_set_data.ID = b_set.ID
        m_set_data.proc_ID = b_set.proc_ID
        m_set_data.inlet = False
        m_set_data.outlet = False

        ### Set All Matched Set Properties
        for m in match:

            m_set = n_boundary_sets[m]

            if b_set.inlet or m_set.inlet:
                m_set_data.inlet = True
            if b_set.outlet or m_set.outlet:
                m_set_data.outlet = True

            m_set_data.ID = b_set.ID
            m_set_data.proc_ID = b_set.proc_ID
            m_set_data.n_ID.push_back(m_set.ID)
            m_set_data.nProcID.push_back(m_set.proc_ID)

        ### Load and clear
        send_matched_set_data[ID]['setID'][b_set.ID] = m_set_data
        m_set_data.n_ID.clear()
        m_set_data.nProcID.clear()
        m_set_data.clear()

    return send_matched_set_data

  def unpack_matched_sets(self,all_matched_set_data):
    """
    Unpack all_matched_set_data into all_matched_sets
    """
    cdef pair[npy_intp,npy_intp] ID
    cdef map[pair[npy_intp,npy_intp],npy_intp] index_convert

    all_matched_sets = []
    n = 0
    for proc_ID,proc_data in enumerate(all_matched_set_data):
        for set in proc_data[proc_ID]['setID'].keys():
            all_matched_sets.append(proc_data[proc_ID]['setID'][set])
            ID = [proc_ID,set]
            index_convert[ID] = n
            n += 1

    return all_matched_sets,index_convert
        
  def organize_matched_sets(self,matched_set_data,index_map):
    """
    Propagate matched Set Information - Single Process
    Iterative go through all connected boundary sets
    Grab inlet,outlet,globalID
    """

    cdef int n,num_procs,num_matched_sets
    cdef int con_set,n_set,n_con_set,n_connect,num_connections,index
    cdef bool inlet,outlet
    cdef npy_intp n_ID

    cdef vector[npy_intp] queue
    cdef pair[npy_intp, npy_intp] set_ID
    cdef vector[ pair[npy_intp, npy_intp] ]set_connect
    cdef map[ pair[npy_intp, npy_intp], bool] visited

    cdef matched_set m_set,c_set
    
    cdef vector[matched_set] all_matched_sets
    all_matched_sets = matched_set_data
    
    cdef map[ pair[npy_intp, npy_intp], npy_intp] index_convert
    index_convert = index_map

    num_procs = self.subDomain.size
    num_matched_sets = len(all_matched_sets)

    ### Set Visited to False
    for n_set in range(0,num_matched_sets):
        m_set = all_matched_sets[n_set]
        set_ID.first = m_set.proc_ID
        set_ID.second = m_set.ID
        visited[set_ID] = False

    num_connections = 0

    ### Loop through all_matched_sets
    for n_set in range(0,num_matched_sets):
        m_set = all_matched_sets[n_set]
        set_ID.first = m_set.proc_ID
        set_ID.second = m_set.ID

        inlet = m_set.inlet
        outlet = m_set.outlet

        if not visited[set_ID]:
            visited[set_ID] = True
            queue.push_back(n_set)
            set_connect.push_back(set_ID)

            while queue.size() > 0:
                c_ID = queue.back()
                queue.pop_back()
                c_set = all_matched_sets[c_ID]

                set_ID.first = c_set.proc_ID
                set_ID.second = c_set.ID

                if not inlet:
                    inlet = c_set.inlet
                if not outlet:
                    outlet = c_set.outlet

                ### Add connected sets to set_connect and queue
                for n_con_set in range(c_set.n_ID.size()):
                    set_ID.first = c_set.nProcID[n_con_set]
                    set_ID.second = c_set.n_ID[n_con_set]
                    index = index_convert[set_ID]
                    if visited[set_ID] == False:
                        visited[set_ID] = True
                        queue.push_back(index)
                        set_connect.push_back(set_ID)


            for n_connect in range(set_connect.size()):
                set_ID = set_connect[n_connect]
                index = index_convert[set_ID]
                all_matched_sets[index].inlet = inlet
                all_matched_sets[index].outlet = outlet
                all_matched_sets[index].global_ID = num_connections

            if set_connect.size() > 0:
                num_connections += 1

        set_connect.clear()

    return all_matched_sets,num_connections

  def repack_matched_sets(self,matched_set_data):
    """
    Organize all the matched sets to send distribute to all procs
    """
    cdef int n_set,num_matched_sets,num_procs
    cdef matched_set m_set,c_set
    cdef vector[matched_set] all_matched_sets
    all_matched_sets = matched_set_data

    num_procs = self.subDomain.size
    connections = [[] for _ in range(num_procs)]

    num_matched_sets = len(all_matched_sets)

    for n_set in range(0,num_matched_sets):
        c_set = all_matched_sets[n_set]
        connections[c_set.proc_ID].append({'Sets':(c_set.proc_ID,c_set.ID),'inlet':c_set.inlet,'outlet':c_set.outlet,'globalID':c_set.global_ID})

    return connections

  def organize_global_ID(self,global_ID_data,total_boundary_sets):
    """
    Generate globalID information for all proccess
    Boundary sets get labeled first, then non-boundary
    """
    cdef int num_procs
    num_procs = self.subDomain.size

    ### Generate globalID counters
    local_set_ID_start = np.zeros(num_procs,dtype=np.int64)
    if total_boundary_sets  > 1:
      local_set_ID_start[0] = total_boundary_sets - 1
    else:
      local_set_ID_start[0] = total_boundary_sets
    for n in range(1,num_procs):
        local_set_ID_start[n] = local_set_ID_start[n-1] + global_ID_data[n-1][0] - global_ID_data[n-1][1]
    
    return local_set_ID_start
     
  def update_globalSetID(self,global_matched_sets):
    """
    Update inlet,outlet,globalID and inlet/outlet and also save to update connectedSets so using global Indexing
    Note: multiple localSetIDs can map to a single globalID but not the otherway
    """
    for data in global_matched_sets:
      localID = data['Sets'][1]
      self.sets[localID].globalID = data['globalID']
      self.localToGlobal[localID] = data['globalID']

      if data['globalID'] not in self.globalToLocal:
        self.globalToLocal[data['globalID']] = [localID]
        
      if localID not in self.globalToLocal[data['globalID']]:
        self.globalToLocal[data['globalID']].append(localID)

      if data['inlet']:
        self.sets[localID].inlet = True

      if data['outlet']:
        self.sets[localID].outlet = True

    for s in self.sets:
      if not s.boundary:
        s.globalID = self.local_set_ID_start
        self.localToGlobal[s.localID] = self.local_set_ID_start
        self.globalToLocal[self.local_set_ID_start] = [s.localID]
        self.local_set_ID_start += 1


def collect_sets(grid,phaseID,inlet,outlet,loopInfo,subdomain):

  rank = subdomain.ID
  size = subdomain.size

  Nodes  = nodes.get_node_info(rank,grid,phaseID,inlet,outlet,subdomain.domain,loopInfo,subdomain)
  Sets = nodes.get_connected_sets(subdomain,grid,phaseID,Nodes)


  if size > 1:

    ### Grab boundary sets and send to neighboring procs
    Sets.get_boundary_sets()
    send_boundary_data = Sets.pack_boundary_data()
    recv_boundary_data = communication.set_COMM(subdomain,send_boundary_data)
    n_boundary_data = Sets.unpack_boundary_data(recv_boundary_data)

    ### Match boundary sets from neighboring procs and send to root for global ID generation
    all_matches = Sets.match_boundary_sets(n_boundary_data)
    Sets.get_num_global_nodes(all_matches,n_boundary_data)
    send_matched_set_data = Sets.pack_matched_sets(all_matches,n_boundary_data)
    recv_matched_set_data = comm.gather(send_matched_set_data, root=0)

   ### Connect sets that are not direct neighbors 
    if subdomain.ID == 0:
      all_matched_sets,index_convert = Sets.unpack_matched_sets(recv_matched_set_data)
      all_matched_sets,total_boundary_sets = Sets.organize_matched_sets(all_matched_sets,index_convert)
      all_matched_sets = Sets.repack_matched_sets(all_matched_sets)
    else:
      all_matched_sets = None
    global_matched_sets = comm.scatter(all_matched_sets, root=0)

    ### Generate and Update global ID information
    global_ID_data = comm.gather([Sets.setCount,Sets.boundarySetCount], root=0)
    if subdomain.ID == 0:
        local_set_ID_start = Sets.organize_global_ID(global_ID_data,total_boundary_sets)
    else:
        local_set_ID_start = None
    Sets.local_set_ID_start = comm.scatter(local_set_ID_start, root=0)

    ### Update IDs
    Sets.update_globalSetID(global_matched_sets)

  return Sets