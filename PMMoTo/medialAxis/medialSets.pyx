# distutils: language = c++
# cython: profile=True
# cython: linetrace=True

import numpy as np
cimport cython
from mpi4py import MPI
comm = MPI.COMM_WORLD
from . import medialPath
from .. import communication

from libcpp.vector cimport vector
from libcpp.utility cimport pair
from numpy cimport npy_intp
from libcpp cimport bool
from libcpp.map cimport map
from libcpp.unordered_map cimport unordered_map

from .. import Orientation
cOrient = Orientation.cOrientation()
cdef int[26][5] directions
cdef int numNeighbors
directions = cOrient.directions
numNeighbors = cOrient.num_neighbors


class medSets(object):
  def __init__(self,
                 sets = None,
                 setCount = 0,
                 pathCount = 0,
                 subDomain = None):
    self.sets = sets
    self.setCount = setCount
    self.pathCount = pathCount 
    self.subDomain = subDomain
    self.boundary_sets = []
    self.boundarySetCount = 0
    self.numConnections = 0
    self.localToGlobal = {}
    self.globalToLocal = {}
    self.trimSetData = {self.subDomain.ID: {'setID':{}}}


  def get_boundary_sets(self):
    """
    Get the sets the are on an interal subDomain boundary.
    Check to make sure neighbor is valid proc_ID
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

  def match_and_update_sets(self):
    """
    This code grabs all boundary sets and matches them based on boundaryNodes
    Then all infromation is updates
    """

    ### Grab boundary sets and send to neighboring procs
    self.get_boundary_sets()
    send_boundary_data = self.pack_boundary_data()
    recv_boundary_data = communication.set_COMM(self.subDomain.Orientation,self.subDomain,send_boundary_data)
    n_boundary_data = self.unpack_boundary_data(recv_boundary_data)

    ### Match boundary sets from neighboring procs and send to root for global ID generation
    all_matches = self.match_boundary_sets(n_boundary_data)
    self.get_num_global_nodes(all_matches,n_boundary_data)
    send_matched_set_data = self.pack_matched_sets(all_matches,n_boundary_data)
    recv_matched_set_data = comm.gather(send_matched_set_data, root=0)

    ### Connect sets that are not direct neighbors 
    if self.subDomain.ID == 0:
        all_matched_sets,index_convert = self.unpack_matched_sets(recv_matched_set_data)
        all_matched_sets,total_boundary_sets = self.organize_matched_sets(all_matched_sets,index_convert)
        all_matched_sets = self.repack_matched_sets(all_matched_sets)
    else:
        all_matched_sets = None
    global_matched_sets = comm.scatter(all_matched_sets, root=0)

    ### Generate and Update global ID information
    global_ID_data = comm.gather([self.setCount,self.boundarySetCount], root=0)
    if self.subDomain.ID == 0:
        local_set_ID_start = self.organize_global_ID(global_ID_data,total_boundary_sets)
    else:
        local_set_ID_start = None
    self.local_set_ID_start = comm.scatter(local_set_ID_start, root=0)

    ### Update IDs and connected Set IDs
    self.update_globalSetID(global_matched_sets)
    self.update_connected_sets()

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
    cdef int set_count,num_b_sets,num_neigh_b_sets,matched_b_nodes,n_set
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
        m_set_data.path_ID = b_set.path_ID
        m_set_data.inlet = False
        m_set_data.outlet = False
        for c_set in b_set.connected_sets:
            m_set_data.connected_sets.push_back(c_set)

        ### Set All Matched Set Properties
        for m in match:

            m_set = n_boundary_sets[m]

            if b_set.inlet or m_set.inlet:
                m_set_data.inlet = True
            if b_set.outlet or m_set.outlet:
                m_set_data.outlet = True

            m_set_data.ID = b_set.ID
            m_set_data.proc_ID = b_set.proc_ID
            m_set_data.path_ID = b_set.path_ID
            m_set_data.n_ID.push_back(m_set.ID)
            m_set_data.nProcID.push_back(m_set.proc_ID)
            m_set_data.n_path_ID.push_back(m_set.path_ID)
            for c_set in m_set.connected_sets:
                m_set_data.n_connected_sets.push_back(c_set)

        ### Load and clear
        send_matched_set_data[ID]['setID'][b_set.ID] = m_set_data
        m_set_data.n_ID.clear()
        m_set_data.nProcID.clear()
        m_set_data.n_path_ID.clear()
        m_set_data.connected_sets.clear()
        m_set_data.n_connected_sets.clear()
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
    local_set_ID_start[0] = total_boundary_sets - 1
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

  def update_connected_sets(self):
    """
    Create globalConnectedSets
    """
    for s in self.sets:
      for cSet in s.connectedSets:
        s.globalConnectedSets.append(self.localToGlobal[cSet])

  def collect_paths(self):
    """
    Initialize medialPath and medialPaths
    """
    paths = []
    for nP in range(self.pathCount):
      paths.append( medialPath.medialPath( localID = nP ) )

    for s in self.sets:
      paths[s.pathID].sets.append(s)
      paths[s.pathID].numSets += 1
      if s.inlet:
        paths[s.pathID].inlet = True
      if s.outlet:
        paths[s.pathID].outlet = True
      if s.boundary:
        paths[s.pathID].boundary = True
        paths[s.pathID].boundarySets.append(s)
        paths[s.pathID].boundarySetIDs.append(s.globalID)
        paths[s.pathID].numBoundarySets += 1
    
    medialPaths = medialPath.medialPaths(paths = paths,
                                         pathCount = self.pathCount,
                                         subDomain = self.subDomain)
                            
    return medialPaths

  def trim_sets(self):
    """
    Trim all dead end sets, where dead end is not connected to at least two boundaries.
    The boundaries could be the same (loops).
    First perform a depth first search to grab dead ends. Then trim. 
    """
    
    cdef int n,nn,nnn,trim,c_node,con_set,num_con_sets

    cdef vertex node
    cdef vector[vertex] vertices
    cdef int num_set = self.setCount

    cdef vector[bool] visited
    cdef vector[npy_intp] queue,set_connect

    ### Initialize and Convert
    for n in range(0,num_set):
        visited.push_back(False)
        node = c_convert_vertex(self.sets[n])
        vertices.push_back(node)

    for n in range(0,num_set):
        if vertices[n].trim == True or visited[n] == True:
            continue
        else:
            visited[n] = True
            queue.push_back(n)

            while queue.size() > 0:
                c_node = queue.back()
                queue.pop_back()
                set_connect.push_back(c_node)

                num_sets = vertices[c_node].connected_sets.size()        
                for nn in range(num_sets):
                    con_set = vertices[c_node].connected_sets[nn]

                    if visited[con_set] == False:
                        queue.push_back(con_set)
                        visited[con_set] = True

            for nn in reversed(set_connect):
                if vertices[nn].inlet or vertices[nn].outlet or vertices[nn].boundary:
                    continue
                elif vertices[nn].connected_sets.size() == 1:
                    vertices[nn].trim = True
                else: 
                    trim = 0
                    num_con_sets = vertices[nn].connected_sets.size()
                    for nnn in range(0,num_con_sets):
                        con_set = vertices[nn].connected_sets[nnn]
                        if vertices[con_set].trim:
                            trim += 1
                    if trim >= num_con_sets - 1:
                        vertices[nn].trim = True


    for n in range(0,num_set):
        self.sets[n].trim = vertices[n].trim

  def update_trimmed_connected_sets(self):
    """
    Update connectedSets so only non-trimmed values
    """
    for s in self.sets:
      for cSet in s.connectedSets[:]:
        if self.sets[cSet].trim == True:
          s.connectedSets.remove(cSet)
          s.globalConnectedSets.remove(self.localToGlobal[cSet])

  def pack_untrimmed_sets(self):
    """
    Send all untrimmed sets to root to perform a global trim. 
    """
    for s in self.sets:
      if not s.trim:
          self.trimSetData[self.subDomain.ID]['setID'][s.globalID] = {
                'inlet': s.inlet,
                'outlet': s.outlet,
                'boundary':s.boundary,
                'pathID': s.pathID,
                'connectedSets': s.globalConnectedSets,
            }

  def unpack_untrimmed_sets(self,trimSetData):
    """
    Serial Code!
    Organize all untrimmed sets and combine sets on diffrent processes
    Perform a depth first search and trim if not connected to inlet and outlet
    """

    cdef int n,nn,conSet,sID,nP,index
    cdef vector[vertex] setInfo
    cdef vertex node
    cdef map[npy_intp,npy_intp] indexConvert
    cdef bool check

    if self.subDomain.ID == 0:

      n = 0
      for nP,procData in enumerate(trimSetData):
        for sID in procData[nP]['setID'].keys():

          # Not in Set
          if indexConvert.find(sID) == indexConvert.end():
            node.ID = sID
            indexConvert[sID] = n
            node.proc_ID.push_back( nP )
            node.inlet = procData[nP]['setID'][sID]['inlet']
            node.outlet = procData[nP]['setID'][sID]['outlet']
            node.boundary = procData[nP]['setID'][sID]['boundary']
            node.trim = False

            for conSet in procData[nP]['setID'][sID]['connectedSets']:
              node.connected_sets.push_back(conSet)

            setInfo.push_back(node)

            node.proc_ID.clear()
            node.connected_sets.clear()

            n += 1

          else:
            index = indexConvert[sID]
            check = True
            for nn in range(0,setInfo[index].proc_ID.size()):
              if setInfo[index].proc_ID[nn] == nP:
                check = False
            if check:
              setInfo[index].proc_ID.push_back( nP )


            for conSet in procData[nP]['setID'][sID]['connectedSets']:
              check = True
              for nn in range(0,setInfo[index].connected_sets.size()):
                if setInfo[index].connected_sets[nn] == conSet:
                  check = False
              if check:
                setInfo[index].connected_sets.push_back(conSet)

    return setInfo,indexConvert


  @cython.boundscheck(False)  # Deactivate bounds checking
  @cython.wraparound(False)   # Deactivate negative indexing.
  def serial_trim_sets(self,setInfo,indexMap):
    """
    Serial Code!
    Trim global sets with depth first search
    """
    
    cdef int n,nn,nnn,conSet,trim,cNode,numCSets

    cdef vector[vertex] vertices
    vertices = setInfo
    cdef int numSet = vertices.size()


    cdef vector[bool] visited
    cdef vector[npy_intp] queue,setConnect
    cdef map[npy_intp,npy_intp] indexConvert
    indexConvert = indexMap

    for _ in range(0,numSet):
      visited.push_back(0)

    if self.subDomain.ID == 0: 

      for n in range(0,numSet):
        if vertices[n].trim == True or visited[n] == True:
          continue
        else:
          visited[n] = True
          queue.push_back(n)

          while queue.size() > 0:
            cNode = queue.back()
            queue.pop_back()
            setConnect.push_back(cNode)

            num_sets = vertices[cNode].connected_sets.size()        
            for nn in range(num_sets):
              conSet = vertices[cNode].connected_sets[nn]
              conSetIndex = indexConvert[conSet]
              if visited[conSetIndex] == False:
                queue.push_back(conSetIndex)
                visited[conSetIndex] = True

          for nn in reversed(setConnect):
            if vertices[nn].inlet or vertices[nn].outlet:
              continue
            elif vertices[nn].connected_sets.size() == 1:
              vertices[nn].trim = True
            else: 
              trim = 0
              numCSets = vertices[nn].connected_sets.size()
              for nnn in range(0,numCSets):
                conSet = vertices[nn].connected_sets[nnn]
                conSetIndex = indexConvert[conSet]
                if vertices[conSetIndex].trim:
                  trim += 1
              if trim >= numCSets - 1:
                vertices[nn].trim = True
                        
          setConnect.clear()

    return vertices

  def repack_global_trimmed_sets(self,setInfo):
    """
    Serial Code!
    Re-pack setInfo to send out
    """
    if self.subDomain.ID == 0: 
  
      sendSetInfo = [[] for _ in range(self.subDomain.size)]
      for s in setInfo:
        for nP in s['proc_ID']:
           sendSetInfo[nP].append([s['ID'],s['trim']])

    else:
      sendSetInfo = None

    globalTrimData = comm.scatter(sendSetInfo, root=0)

    for s in globalTrimData:
      for localID in self.globalToLocal[s[0]]:
        self.sets[localID].trim = s[1]

    

    