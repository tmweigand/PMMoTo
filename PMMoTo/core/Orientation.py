import numpy as np

def get_boundary_ID(boundary_index):
    """
    Determine boundary ID
    Input: boundary_ID[3] corresponding to [x,y,z] and values of -1,0,1
    Output: boundary_ID
    """
    params = [[0, 9, 18],[0, 3, 6],[0, 1, 2]]

    ID = 0
    for n in range(0,3):
        if boundary_index[n] < 0:
            ID += params[n][0]
        elif boundary_index[n] > 0:
            ID += params[n][1]
        else:
            ID += params[n][2]

    return ID


def add_faces(boundary_features):
    """
    Since loop_info are by face, need to add face index for edges and corners in case
    edge and corner n_procs are < 0 but face is valid. 
    """
    for n in range(0,num_neighbors):
        if boundary_features[n]:
            for nn in allFaces[n]:
                boundary_features[nn] = True


class CubeFeature(object):
    def __init__(self,ID,n_proc,boundary,periodic):
        self.ID = ID
        self.n_proc = n_proc
        self.periodic = periodic
        self.boundary = boundary
        self.periodic_correction = (0,0,0)

    def get_periodic_correction(self):
        """
        Determin spatial correction factor if periodic
        """
        if self.periodic:
            self.periodic_correction = self.periodic

class Face(CubeFeature):
    def __init__(self,ID,n_proc,boundary,periodic):
        super().__init__(ID,n_proc,boundary,periodic)
        self.info = faces[ID]
        self.feature_ID = get_boundary_ID(self.info['ID'])
        self.opp_info = faces[faces[ID]['oppIndex']]
        self.get_periodic_correction()
        

class Edge(CubeFeature):
    def __init__(self,ID,n_proc,boundary,periodic,global_boundary,external_faces):
        super().__init__(ID,n_proc,boundary,periodic)
        self.global_boundary = global_boundary
        self.external_faces = external_faces
        self.info = edges[ID]
        self.feature_ID = get_boundary_ID(self.info['ID'])
        self.opp_info = edges[edges[ID]['oppIndex']]
        self.get_periodic_correction()
        self.extend = [[0,0],[0,0],[0,0]]

    def get_extension(self,extend_domain,bounds):
        """
        Determine the span of the feature based on extend
        """
        _faces = edges[self.ID]['ID']
        for n,f in enumerate(_faces):
            if f > 0:
                self.extend[n][0] = bounds[n][-1] - extend_domain[n]
                self.extend[n][1] = bounds[n][-1]
            elif f < 0:
                self.extend[n][0] = bounds[n][0]
                self.extend[n][1] = bounds[n][0] + extend_domain[n]
            else:
                self.extend[n][0] = 0
                self.extend[n][1] = 0


class Corner(CubeFeature):
    def __init__(self,ID,n_proc,boundary,periodic,global_boundary,external_faces,external_edges):
        super().__init__(ID,n_proc,boundary,periodic)
        self.global_boundary = global_boundary
        self.external_faces = external_faces
        self.external_edges = external_edges
        self.info = corners[ID]
        self.feature_ID = get_boundary_ID(self.info['ID'])
        self.opp_info = corners[corners[ID]['oppIndex']]
        self.get_periodic_correction()
        self.extend = [[0,0],[0,0],[0,0]]

    def get_extension(self,extend_domain,bounds):
        """
        Determine the span of the feature based on extend
        """
        faces = corners[self.ID]['ID']
        for n,f in enumerate(faces):
            if f > 0:
                self.extend[n][0] = bounds[n][-1] - extend_domain[n]
                self.extend[n][1] = bounds[n][-1]
            elif f < 0:
                self.extend[n][0] = bounds[n][0]
                self.extend[n][1] = bounds[n][0] + extend_domain[n]
            else:
                self.extend[n][0] = 0
                self.extend[n][1] = 0


num_faces = 6
num_edges = 12
num_corners = 8
num_neighbors = 26

faces = {0:{'ID':(-1, 0, 0),'oppIndex':1, 'argOrder':np.array([0,1,2],dtype=np.uint8), 'dir': 1},
         1:{'ID':( 1, 0, 0),'oppIndex':0, 'argOrder':np.array([0,1,2],dtype=np.uint8), 'dir':-1},
         2:{'ID':( 0,-1, 0),'oppIndex':3, 'argOrder':np.array([1,0,2],dtype=np.uint8), 'dir': 1},
         3:{'ID':( 0, 1, 0),'oppIndex':2, 'argOrder':np.array([1,0,2],dtype=np.uint8), 'dir':-1},
         4:{'ID':( 0, 0,-1),'oppIndex':5, 'argOrder':np.array([2,0,1],dtype=np.uint8), 'dir': 1},
         5:{'ID':( 0, 0, 1),'oppIndex':4, 'argOrder':np.array([2,0,1],dtype=np.uint8), 'dir':-1}
        }
        
edges = {0 :{'ID':(-1, 0,-1), 'oppIndex':5, 'faceIndex':(0,4), 'dir':(0,2)},
         1 :{'ID':(-1, 0, 1), 'oppIndex':4, 'faceIndex':(0,5), 'dir':(0,2)},
         2 :{'ID':(-1,-1, 0), 'oppIndex':7, 'faceIndex':(0,2), 'dir':(0,1)},
         3 :{'ID':(-1, 1, 0), 'oppIndex':6, 'faceIndex':(0,3), 'dir':(0,1)},
         4 :{'ID':( 1, 0,-1), 'oppIndex':1, 'faceIndex':(1,4), 'dir':(0,2)},
         5 :{'ID':( 1, 0, 1), 'oppIndex':0, 'faceIndex':(1,5), 'dir':(0,2)},
         6 :{'ID':( 1,-1, 0), 'oppIndex':3, 'faceIndex':(1,2), 'dir':(0,1)},
         7 :{'ID':( 1, 1, 0), 'oppIndex':2, 'faceIndex':(1,3), 'dir':(0,1)},
         8 :{'ID':( 0,-1,-1), 'oppIndex':11,'faceIndex':(2,4), 'dir':(1,2)},
         9 :{'ID':( 0,-1, 1), 'oppIndex':10,'faceIndex':(2,5), 'dir':(1,2)},
         10:{'ID':( 0, 1,-1), 'oppIndex':9, 'faceIndex':(3,4), 'dir':(1,2)},
         11:{'ID':( 0, 1, 1), 'oppIndex':8, 'faceIndex':(3,5), 'dir':(1,2)},
        }

corners = {0:{'ID':(-1,-1,-1),'oppIndex':7, 'faceIndex':(0,2,4), 'edgeIndex':(0,2,8)},
           1:{'ID':(-1,-1, 1),'oppIndex':6, 'faceIndex':(0,2,5), 'edgeIndex':(1,2,9)},
           2:{'ID':(-1, 1,-1),'oppIndex':5, 'faceIndex':(0,3,4), 'edgeIndex':(0,3,10)},
           3:{'ID':(-1, 1, 1),'oppIndex':4, 'faceIndex':(0,3,5), 'edgeIndex':(1,3,11)},
           4:{'ID':( 1,-1,-1),'oppIndex':3, 'faceIndex':(1,2,4), 'edgeIndex':(4,6,8)}, 
           5:{'ID':( 1,-1, 1),'oppIndex':2, 'faceIndex':(1,2,5), 'edgeIndex':(5,6,9)},
           6:{'ID':( 1, 1,-1),'oppIndex':1, 'faceIndex':(1,3,4), 'edgeIndex':(4,7,10)}, 
           7:{'ID':( 1, 1, 1),'oppIndex':0, 'faceIndex':(1,3,5), 'edgeIndex':(5,7,11)}
          }

features = [(-1, 0, 0),
            ( 1, 0, 0),
            ( 0,-1, 0),
            ( 0, 1, 0),
            ( 0, 0,-1),
            ( 0, 0, 1),
            (-1, 0,-1),
            (-1, 0, 1),
            (-1,-1, 0),
            (-1, 1, 0),
            ( 1, 0,-1),
            ( 1, 0, 1),
            ( 1,-1, 0),
            ( 1, 1, 0),
            ( 0,-1,-1),
            ( 0,-1, 1),
            ( 0, 1,-1),
            ( 0, 1, 1),
            (-1,-1,-1),
            (-1,-1, 1),
            (-1, 1,-1),
            (-1, 1, 1),
            ( 1,-1,-1),
            ( 1,-1, 1),
            ( 1, 1,-1),
            ( 1, 1, 1),
           ]

directions = {0 :{'ID':[-1,-1,-1],'index': 0 ,'oppIndex': 25},
              1 :{'ID':[-1,-1,0], 'index': 1 ,'oppIndex': 24},
              2 :{'ID':[-1,-1,1], 'index': 2 ,'oppIndex': 23},
              3 :{'ID':[-1,0,-1], 'index': 3 ,'oppIndex': 22},
              4 :{'ID':[-1,0,0],  'index': 4 ,'oppIndex': 21},
              5 :{'ID':[-1,0,1],  'index': 5 ,'oppIndex': 20},
              6 :{'ID':[-1,1,-1], 'index': 6 ,'oppIndex': 19},
              7 :{'ID':[-1,1,0],  'index': 7 ,'oppIndex': 18},
              8 :{'ID':[-1,1,1],  'index': 8 ,'oppIndex': 17},
              9 :{'ID':[0,-1,-1], 'index': 9 ,'oppIndex': 16},
              10:{'ID':[0,-1,0],  'index': 10 ,'oppIndex': 15},
              11:{'ID':[0,-1,1],  'index': 11 ,'oppIndex': 14},
              12:{'ID':[0,0,-1],  'index': 12 ,'oppIndex': 13},
              13:{'ID':[0,0,1],   'index': 13 ,'oppIndex': 12},
              14:{'ID':[0,1,-1],  'index': 14 ,'oppIndex': 11},
              15:{'ID':[0,1,0],   'index': 15 ,'oppIndex': 10},
              16:{'ID':[0,1,1],   'index': 16 ,'oppIndex': 9},
              17:{'ID':[1,-1,-1], 'index': 17 ,'oppIndex': 8},
              18:{'ID':[1,-1,0],  'index': 18 ,'oppIndex': 7},
              19:{'ID':[1,-1,1],  'index': 19 ,'oppIndex': 6},
              20:{'ID':[1,0,-1],  'index': 20 ,'oppIndex': 5},
              21:{'ID':[1,0,0],   'index': 21 ,'oppIndex': 4},
              22:{'ID':[1,0,1],   'index': 22 ,'oppIndex': 3},
              23:{'ID':[1,1,-1],  'index': 23 ,'oppIndex': 2},
              24:{'ID':[1,1,0],   'index': 24 ,'oppIndex': 1},
              25:{'ID':[1,1,1],   'index': 25 ,'oppIndex': 0},
             }

allFaces = [[0, 2, 6, 8, 18, 20, 24],         # 0
            [1, 2, 7, 8, 19, 20, 25],         # 1
            [2, 8, 20],                       # 2
            [3, 5, 6, 8, 21, 23, 24],         # 3
            [4, 5, 7, 8, 22, 23, 25],         # 4
            [5, 8, 23],                       # 5
            [6, 8, 24],                       # 6
            [7, 8, 25],                       # 7
            [8],                              # 8
            [9, 11, 15, 17, 18, 20, 24],      # 9
            [10, 11, 16, 17, 19, 20, 25],     # 10
            [11, 17, 20],                     # 11
            [12, 14, 15, 17, 21, 23, 24],     # 12
            [13, 14, 16, 17, 22, 23, 25],     # 13
            [14, 17, 23],                     # 14
            [15, 17, 24],                     # 15
            [16, 17, 25],                     # 16
            [17],                             # 17
            [18, 20, 24],                     # 18
            [19, 20, 25],                     # 19
            [20],                             # 20
            [21, 23, 24],                     # 21
            [22, 23, 25],                     # 22
            [23],                             # 23
            [24],                             # 24
            [25]]                             # 25


def get_index_ordering(inlet,outlet):
    """
    This function rearranges the loop_info ordering so
    the inlet and outlet faces are first. 
    """
    order = [0,1,2]
    for n in range(0,3):
        if inlet[n*2] or outlet[n*2] or inlet[n*2+1] or outlet[n*2+1]:
            order.remove(n);
            order.insert(0,n)

    return order

def get_loop_info(grid,subDomain,inlet,outlet,res_pad):
    """
    Grap loop information to cycle through the boundary Faces and internal nodes
    Reservoirs are treated as entire face  
    Order ensures that inlet/outlet edges and corners are included in optimized looping 
    """
    order = get_index_ordering(inlet,outlet)
    loop_info = np.zeros([num_faces+1,3,2],dtype = np.int64)

    range_info = 2*np.ones([6],dtype=np.uint8)
    for f_index in faces:
        if subDomain.boundary_ID[f_index] == 0:
            range_info[f_index] = range_info[f_index] - 1
        if inlet[f_index] > 0:
            range_info[f_index] = range_info[f_index] + res_pad
        if outlet[f_index] > 0:
            range_info[f_index] = range_info[f_index] + res_pad

    for f_index in faces:
        face = faces[f_index]['argOrder'][0]
        g_s = [grid.shape[order[0]],grid.shape[order[1]],grid.shape[order[2]]]

        if faces[f_index]['dir'] == -1:
            if face == order[0]:
                loop_info[f_index,order[0]] = [g_s[0] - range_info[order[0]*2+1], g_s[0]]
                loop_info[f_index,order[1]] = [0, g_s[1]]
                loop_info[f_index,order[2]] = [0, g_s[2]]
            elif face == order[1]:
                loop_info[f_index,order[0]] = [range_info[order[0]*2], g_s[0] - range_info[order[0]*2+1]]
                loop_info[f_index,order[1]] = [g_s[1] - range_info[order[1]*2+1], g_s[1]]
                loop_info[f_index,order[2]] = [0, g_s[2]]
            elif face == order[2]:
                loop_info[f_index,order[0]] = [range_info[order[0]*2], g_s[0] - range_info[order[0]*2+1]]
                loop_info[f_index,order[1]] = [range_info[order[1]*2], g_s[1]-range_info[order[1]*2+1]]
                loop_info[f_index,order[2]] = [g_s[2] - range_info[order[2]*2+1], g_s[2]]

        elif faces[f_index]['dir'] == 1:
            if face == order[0]:
                loop_info[f_index,order[0]] = [0, range_info[order[0]*2]]
                loop_info[f_index,order[1]] = [0, g_s[1]]
                loop_info[f_index,order[2]] = [0, g_s[2]]
            elif face == order[1]:
                loop_info[f_index,order[0]] = [range_info[order[0]*2], g_s[0]-range_info[order[0]*2+1]]
                loop_info[f_index,order[1]] = [0, range_info[order[1]*2]]
                loop_info[f_index,order[2]] = [0, g_s[2]]
            elif face == order[2]:
                loop_info[f_index,order[0]] = [range_info[order[0]*2],g_s[0]-range_info[order[0]*2+1]]
                loop_info[f_index,order[1]] = [range_info[order[1]*2],g_s[1]-range_info[order[1]*2+1]]
                loop_info[f_index,order[2]] = [0,range_info[order[2]*2]]

    loop_info[num_faces][order[0]] = [range_info[order[0]*2],g_s[0]-range_info[order[0]*2+1]]
    loop_info[num_faces][order[1]] = [range_info[order[1]*2],g_s[1]-range_info[order[1]*2+1]]
    loop_info[num_faces][order[2]] = [range_info[order[2]*2],g_s[2]-range_info[order[2]*2+1]]
    
    return loop_info

def get_send_halo(struct_ratio,buffer,dim):
    """
    Determine slices of face, edge, and corner neighbor to send data 
    structRatio is size of voxel window to send and is [nx,ny,nz]
    buffer is the subDomain.buffer
    dim is grid.shape
    Buffer is always updated on edges and corners due to geometry contraints
    """
    send_faces = np.empty([num_faces,3],dtype=object)
    send_edges = np.empty([num_edges,3],dtype=object)
    send_corner = np.empty([num_corners,3],dtype=object)

    #############
    ### Faces ###
    #############
    for f_index in faces:
        f_ID = faces[f_index]['ID']
        for n in range(len(f_ID)):
            if f_ID[n] != 0:
                if f_ID[n] > 0:
                    send_faces[f_index,n] = slice(dim[n]-struct_ratio[n*2+1]-buffer[n*2+1]-1,dim[n]-buffer[n*2+1]-1)
                else:
                    send_faces[f_index,n] = slice(buffer[n*2]+1,buffer[n*2]+struct_ratio[n*2]+1)
            else:
                send_faces[f_index,n] = slice(None,None)
    #############

    #############
    ### Edges ###
    #############
    for e_index in edges:
        e_ID = edges[e_index]['ID']
        for n in range(len(e_ID)):
            if e_ID[n] != 0:
                if e_ID[n] > 0:
                    send_edges[e_index,n] = slice(dim[n]-struct_ratio[n*2+1]-buffer[n*2+1]-1,dim[n]-1)
                else:
                    send_edges[e_index,n] = slice(buffer[n*2],buffer[n*2]+struct_ratio[n*2]+1)
            else:
                send_edges[e_index,n] = slice(None,None)
    #############

    ###############
    ### Corners ###
    ###############
    for c_index in corners:
        c_ID = corners[c_index]['ID']
        for n in range(len(c_ID)):
            if c_ID[n] > 0:
                send_corner[c_index,n] = slice(dim[n]-struct_ratio[n*2+1]-buffer[n*2+1]-1,dim[n]-1)
            else:
                send_corner[c_index,n] = slice(buffer[n*2],buffer[n*2]+struct_ratio[n*2]+1)
    ###############

    return send_faces,send_edges,send_corner

def get_send_buffer(buffer,dim):
    """
    Determine slices of face, edge, and corner neighbor to send data 
    structRatio is size of voxel window to send and is [nx,ny,nz]
    buffer is the subDomain.buffer
    dim is grid.shape
    Buffer is always updated on edges and corners due to geometry contraints
    """

    send_faces = np.empty([num_faces,3],dtype=object)
    send_edges = np.empty([num_edges,3],dtype=object)
    send_corner = np.empty([num_corners,3],dtype=object)

    #############
    ### Faces ###
    #############
    for f_index in faces:
        f_ID = faces[f_index]['ID']
        for n in range(len(f_ID)):
            if f_ID[n] != 0:
                if f_ID[n] > 0:
                    send_faces[f_index,n] = slice(dim[n]-2*buffer[n*2+1],dim[n]-buffer[n*2+1])
                else:
                    send_faces[f_index,n] = slice(buffer[n*2],2*buffer[n*2])
            else:
                send_faces[f_index,n] = slice(buffer[n*2],dim[n]-buffer[n*2+1])
    #############

    #############
    ### Edges ###
    #############
    for e_index in edges:
        e_ID = edges[e_index]['ID']
        for n in range(len(e_ID)):
            if e_ID[n] != 0:
                if e_ID[n] > 0:
                    send_edges[e_index,n] = slice(dim[n]-2*buffer[n*2+1],dim[n]-buffer[n*2+1])
                else:
                    send_edges[e_index,n] = slice(buffer[n*2],2*buffer[n*2])
            else:
                send_edges[e_index,n] = slice(buffer[n*2],dim[n]-buffer[n*2+1])
    #############

    ###############
    ### Corners ###
    ###############
    for c_index in corners:
        c_ID = corners[c_index]['ID']
        for n in range(len(c_ID)):
            if c_ID[n] > 0:
                send_corner[c_index,n] = slice(dim[n]-2*buffer[n*2+1],dim[n]-buffer[n*2+1])
            else:
                send_corner[c_index,n] = slice(buffer[n*2],2*buffer[n*2])
    ###############

    return send_faces,send_edges,send_corner


def get_recv_halo(halo,buffer,dim):
    """
    Determine slices of face, edge, and corner neighbor to recieve data 
    Buffer is always updated on edges and corners due to geometry contraints
    """

    recv_faces = np.empty([num_faces,3],dtype=object)
    recv_edges = np.empty([num_edges,3],dtype=object)
    recv_corners = np.empty([num_corners,3],dtype=object)

    #############
    ### Faces ###
    #############
    for f_index in faces:
        f_ID = faces[f_index]['ID']
        for n in range(len(f_ID)):
            if f_ID[n] != 0:
                if f_ID[n] > 0:
                    recv_faces[f_index,n] = slice(dim[n]-halo[n*2+1],dim[n])
                else:
                    recv_faces[f_index,n] = slice(None,halo[n*2])
            else:
                recv_faces[f_index,n] = slice(halo[n*2],dim[n]-halo[n*2+1])
    #############

    #############
    ### Edges ###
    #############
    for e_index in edges:
        e_ID = edges[e_index]['ID']
        for n in range(len(e_ID)):
            if e_ID[n] != 0:
                if e_ID[n] > 0:
                    recv_edges[e_index,n] = slice(dim[n]-halo[n*2+1]-buffer[n*2+1],dim[n])
                else:
                    recv_edges[e_index,n] = slice(None,halo[n*2]+buffer[n*2])
            else:
                recv_edges[e_index,n] = slice(halo[n*2],dim[n]-halo[n*2+1])
    #############

    ###############
    ### Corners ###
    ###############
    for c_index in corners:
        c_ID = corners[c_index]['ID']
        for n in range(len(c_ID)):
            if c_ID[n] > 0:
                recv_corners[c_index,n] = slice(dim[n]-halo[n*2+1]-buffer[n*2+1],dim[n])
            else:
                recv_corners[c_index,n] = slice(None,halo[n*2]+buffer[n*2])
    ###############

    return recv_faces,recv_edges,recv_corners

def get_recv_buffer(buffer,dim):
    """
    Determine slices of face, edge, and corner neighbor to recieve data 
    Buffer is always updated on edges and corners due to geometry contraints
    """

    recv_faces = np.empty([num_faces,3],dtype=object)
    recv_edges = np.empty([num_edges,3],dtype=object)
    recv_corners = np.empty([num_corners,3],dtype=object)


    #############
    ### Faces ###
    #############
    for f_index in faces:
        f_ID = faces[f_index]['ID']
        for n in range(len(f_ID)):
            if f_ID[n] != 0:
                if f_ID[n] > 0:
                    recv_faces[f_index,n] = slice(dim[n]-buffer[n*2+1],dim[n])
                else:
                    recv_faces[f_index,n] = slice(None,buffer[n*2])
            else:
                recv_faces[f_index,n] = slice(buffer[n*2],dim[n]-buffer[n*2+1])
    #############

    #############
    ### Edges ###
    #############
    for e_index in edges:
        e_ID = edges[e_index]['ID']
        for n in range(len(e_ID)):
            if e_ID[n] != 0:
                if e_ID[n] > 0:
                    recv_edges[e_index,n] = slice(dim[n]-buffer[n*2+1],dim[n])
                else:
                    recv_edges[e_index,n] = slice(None,buffer[n*2])
            else:
                recv_edges[e_index,n] = slice(buffer[n*2],dim[n]-buffer[n*2+1])
    #############

    ###############
    ### Corners ###
    ###############
    for c_index in corners:
        c_ID = corners[c_index]['ID']
        for n in range(len(c_ID)):
            if c_ID[n] > 0:
                recv_corners[c_index,n] = slice(dim[n]-buffer[n*2+1],dim[n])
            else:
                recv_corners[c_index,n] = slice(None,buffer[n*2])
    ###############

    return recv_faces,recv_edges,recv_corners
