import numpy as np
from . import Orientation


def unpad(grid,pad):
    """
    Unpad a padded array
    """
    _dim = grid.shape
    grid_out = grid[pad[0]:_dim[0]-pad[1],
                    pad[2]:_dim[1]-pad[3],
                    pad[4]:_dim[2]-pad[5]]
    return np.ascontiguousarray(grid_out)

def own_grid(grid,own):
    """
    Pass array with only nodes owned py that process
    """
    grid_out =  grid[own[0]:own[1],
                     own[2]:own[3],
                     own[4]:own[5]]
    
    return np.ascontiguousarray(grid_out)


def partition_boundary_solids(subdomain,solids,extend_factor = 0.7):
    """
    Trim solids to minimize communication and reduce KD Tree. Identify on Surfaces, Edges, and Corners
    Keep all face solids, and use extend factor to query which solids to include for edges and corners
    """
    face_solids = [[] for _ in range(len(Orientation.faces))]
    edge_solids = [[] for _ in range(len(Orientation.edges))]
    corner_solids = [[] for _ in range(len(Orientation.corners))]
    
    extend = [extend_factor*x for x in subdomain.size_subdomain]
    coords = subdomain.coords

    ### Faces ###
    for fIndex in Orientation.faces:
        pointsXYZ = []
        points = solids[np.where( (solids[:,0]>-1)
                                & (solids[:,1]>-1)
                                & (solids[:,2]>-1)
                                & (solids[:,3]==fIndex) )][:,0:3]
        for x,y,z in points:
            pointsXYZ.append([coords[0][x],coords[1][y],coords[2][z]] )
        face_solids[fIndex] = np.asarray(pointsXYZ)

    ### Edges ###
    for edge in subdomain.edges:
        edge.get_extension(extend,subdomain.bounds)
        for f,d in zip(edge.info['faceIndex'],reversed(edge.info['dir'])): # Flip dir for correct nodes
            f_solids = face_solids[f]
            values = (edge.extend[d][0] <= f_solids[:,d]) & (f_solids[:,d] <= edge.extend[d][1])
            if len(edge_solids[edge.ID]) == 0:
                edge_solids[edge.ID] = f_solids[np.where(values)]
            else:
                edge_solids[edge.ID] = np.append(edge_solids[edge.ID],f_solids[np.where(values)],axis=0)
        edge_solids[edge.ID] = np.unique(edge_solids[edge.ID],axis=0)

    ### Corners ###
    iterates = [[1,2],[0,2],[0,1]]
    for corner in subdomain.corners:
        corner.get_extension(extend,subdomain.bounds)
        values = [None,None]
        for it,f in zip(iterates,corner.info['faceIndex']):
            f_solids = face_solids[f]
            for n,i in enumerate(it):
                values[n] = (corner.extend[i][0] <= f_solids[:,i]) & (f_solids[:,i] <= corner.extend[i][1])
            if len(corner_solids[corner.ID]) == 0:
                corner_solids[corner.ID] = f_solids[np.where(values[0] & values[1])]
            else:
                corner_solids[corner.ID] = np.append(corner_solids[corner.ID],f_solids[np.where(values[0] & values[1])],axis=0)
        corner_solids[corner.ID] = np.unique(corner_solids[corner.ID],axis=0)

    return face_solids,edge_solids,corner_solids
