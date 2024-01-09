import os
import numpy as np
import gzip 
import pyvista as pv
from . import utils

__all__ = [
    "read_sphere_pack_xyzr",
    "read_vtk_grid",
    "readPorousMediaLammpsDump",
]

def read_sphere_pack_xyzr(input_file,add_periodic_spheres = False):
    """Read in sphere pack given in x,y,z,(r)adius order

        Input File Format:
            x_min x_max
            y_min y_max
            z_min z_max
            x y z r
            x y z r
            x y z r
            x y z r
            x y z r
    """

    # Check input file and proceed of exists
    utils.check_file(input_file)

    domain_file = open(input_file,'r',encoding="utf-8")
    lines = domain_file.readlines()
    num_spheres = len(lines) - 3

    sphere_data = np.zeros([4,num_spheres])
    domain = np.zeros([3,2])

    count_sphere = 0
    for n_line,line in enumerate(lines):
        if n_line < 3: # Grab domain size
            domain[n_line,0] =  float(line.split(" ")[0])
            domain[n_line,1] =  float(line.split(" ")[1])
        else: # Grab sphere
            try:
                for n in range(0,4):
                    sphere_data[n,count_sphere] = float(line.split(" ")[n])
            except ValueError:
                for n in range(0,4):
                    sphere_data[n,count_sphere] = float(line.split("\t")[n])
            count_sphere += 1

    domain_file.close()

    return domain,sphere_data

def read_vtk_grid(rank,size,file):
    """Read in parallel vtk file. Size must equal size when file written

    """
    proc_files = os.listdir(file)
    proc_files.sort()

    # Check num_files is equal to mpi.size
    utils.check_num_files(len(proc_files),size)

    p_file = file + '/' + proc_files[rank]
    data = pv.read(p_file)
    array_name = data.array_names
    grid_data = np.reshape(data[array_name[0]],data.dimensions,'F')

    return np.ascontiguousarray(grid_data)

# def readPorousMediaLammpsDump(file, dataReadkwargs):
    
#     indexVars = ['x', 'y', 'z', 'type']
#     indexes = []
#     rLookupFile = None
#     boundaryLims = None
#     for key,value in dataReadkwargs.items():
#         if key == 'rLookupFile':
#             rLookupFile=value
#         if key == 'boundaryLims':
#             boundaryLims = value
#             for limValues in boundaryLims:
#                 if limValues[0] is None:
#                     limValues[0] = -np.inf
#                 if limValues[1] is None:
#                     limValues[1] = np.inf
#     if rLookupFile:
#         sigmaLJ = []
#         rLookup = open(rLookupFile,'r')
#         lookupLines = rLookup.readlines()
#         for line in lookupLines:
#             ### For the purposes of being able to specify arbitrarily
#             ### sized particles, this definition of cutoff location
#             ### relies on the assumption that the 'test particle'
#             ### (as LJ interactions occur only between > 1 particles)
#             ### has a LJsigma of 0. i.e. this evaluates the maximum possible
#             ### size of the free volume network

#             ### The following would select the energy minimum, i.e.
#             ### movement of a particles center closer than this requires
#             ### input force

#             sigmaLJ.append(1.12246204830937*float(line.split(" ")[2])/2)


    
#     else:
#         print('ERROR: Attempting to read LAMMPS dump file without atomic information.')
#         communication.raiseError()
#     maxR = max(sigmaLJ)
#     if file.endswith('.gz'):
#         domainFile = gzip.open(file,'rt')
#     else:
#         domainFile = open(file,'r')

#     Lines = domainFile.readlines()

#     for i in range(9):
#         preamLine = Lines.pop(0)
#         if (i == 1):
#             timeStep = preamLine
#         elif (i == 3):
#             numObjects = int(preamLine)
#         elif (i == 5):
#             xMin = float(preamLine.split(" ")[0])
#             xMax = float(preamLine.split(" ")[1])
#             xD = xMax-xMin
#         elif (i == 6):
#             yMin = float(preamLine.split(" ")[0])
#             yMax = float(preamLine.split(" ")[1])
#             yD = yMax-yMin
#         elif (i == 7):
#             zMin = float(preamLine.split(" ")[0])
#             zMax = float(preamLine.split(" ")[1])
#             zD = zMax-zMin
#         elif (i == 8):
#             colTypes = [j.strip() for j in preamLine.split(" ")[2:]]
#     if boundaryLims:
#         domainDim = np.array([[max(xMin,boundaryLims[0][0]), min(xMax,boundaryLims[0][1])],
#                               [max(yMin,boundaryLims[1][0]), min(yMax,boundaryLims[1][1])],
#                               [max(zMin,boundaryLims[2][0]), min(zMax,boundaryLims[2][1])]])
#     else:
#         domainDim = np.array([[xMin, xMax],[yMin, yMax],[zMin, zMax]])
#     for var in indexVars:
#         indexes.append(colTypes.index(var))  
        
#     xSphere = np.zeros(numObjects)
#     ySphere = np.zeros(numObjects)
#     zSphere = np.zeros(numObjects)
#     rSphere = np.zeros(numObjects)

#     c = 0
#     for line in Lines:
#         # NEED THIS CORRECTION IF USING UNWRAPPED COORDINATES, maybe flag?
#         # x = float(line.split(" ")[indexes[0]])-xD*math.floor((float(line.split(" ")[indexes[0]])-xMin)/xD)
#         # y = float(line.split(" ")[indexes[1]])-yD*math.floor((float(line.split(" ")[indexes[1]])-yMin)/yD)
#         # z = float(line.split(" ")[indexes[2]])-zD*math.floor((float(line.split(" ")[indexes[2]])-zMin)/zD)
        
#         x = float(line.split(" ")[indexes[0]])
#         y = float(line.split(" ")[indexes[1]])
#         z = float(line.split(" ")[indexes[2]])

#         # Only keep atom if it is in the Domain, by a margin of the largest possible 
#         # radius. Done because subsampled region might not be the same size as the domain 
#         xCheck = domainDim[0][0]-maxR <= x <= domainDim[0][1]+maxR
#         yCheck = domainDim[1][0]-maxR <= y <= domainDim[1][1]+maxR
#         zCheck = domainDim[2][0]-maxR <= z <= domainDim[2][1]+maxR
#         if xCheck and yCheck and zCheck:
#             xSphere[c] = x
#             ySphere[c] = y
#             zSphere[c] = z
#             rSphere[c] = float(sigmaLJ[int(line.split(" ")[indexes[3]])-1])
#             c += 1

#     sphereData = np.zeros([4, c])
#     sphereData[0,:] = xSphere[:c]
#     sphereData[1,:] = ySphere[:c]
#     sphereData[2,:] = zSphere[:c]
#     sphereData[3,:] = rSphere[:c]*rSphere[:c]
#     domainFile.close()

#     return domainDim, sphereData



def readPorousMediaLammpsDump(file, dataReadkwargs):
    
    indexVars = ['x', 'y', 'z', 'type']
    indexes = []
    rLookupFile = None
    boundaryLims = None
    boundaries = None
    nodes = None
    waterMolecule = None
    for key,value in dataReadkwargs.items():
        if key == 'rLookupFile':
            rLookupFile=value
        if key == 'boundaryLims':
            boundaryLims = value
            for limValues in boundaryLims:
                if limValues[0] is None:
                    limValues[0] = -np.inf
                if limValues[1] is None:
                    limValues[1] = np.inf
        if key == 'boundaries':
            boundaries = value
        if key == 'nodes':
            nodes = value
        if key == 'waterMolecule':
            waterMolecule = True
    if rLookupFile:
        sigmaLJ = []
        rLookup = open(rLookupFile,'r')
        lookupLines = rLookup.readlines()
        if waterMolecule:
            for line in lookupLines:
                ### For the purposes of being able to specify arbitrarily
                ### sized particles, this definition of cutoff location
                ### relies on the assumption that the 'test particle'
                ### (as LJ interactions occur only between > 1 particles)
                ### has a LJsigma of 0. i.e. this evaluates the maximum possible
                ### size of the free volume network

                ### The following would select the energy minimum, i.e.
                ### movement of a particles center closer than this requires
                ### input force
                combinedSigma = 1.12246204830937*(float(line.split(" ")[2])+3.178)/2
                sigmaLJ.append(combinedSigma/2)
        else:
            for line in lookupLines:
                ### For the purposes of being able to specify arbitrarily
                ### sized particles, this definition of cutoff location
                ### relies on the assumption that the 'test particle'
                ### (as LJ interactions occur only between > 1 particles)
                ### has a LJsigma of 0. i.e. this evaluates the maximum possible
                ### size of the free volume network

                ### The following would select the energy minimum, i.e.
                ### movement of a particles center closer than this requires
                ### input force

                combinedSigma = 1.12246204830937*(float(line.split(" ")[2])+0.0)/2
                sigmaLJ.append(combinedSigma/2)


    
    else:
        print('ERROR: Attempting to read LAMMPS dump file without atomic information.')
        communication.raiseError()
    maxR = max(sigmaLJ)
    if file.endswith('.gz'):
        domainFile = gzip.open(file,'rt')
    else:
        domainFile = open(file,'r')

    Lines = domainFile.readlines()

    for i in range(9):
        preamLine = Lines.pop(0)
        if (i == 1):
            timeStep = preamLine
        elif (i == 3):
            numObjects = int(preamLine)
        elif (i == 5):
            xMin = float(preamLine.split(" ")[0])
            xMax = float(preamLine.split(" ")[1])
            xD = xMax-xMin
        elif (i == 6):
            yMin = float(preamLine.split(" ")[0])
            yMax = float(preamLine.split(" ")[1])
            yD = yMax-yMin
        elif (i == 7):
            zMin = float(preamLine.split(" ")[0])
            zMax = float(preamLine.split(" ")[1])
            zD = zMax-zMin
        elif (i == 8):
            colTypes = [j.strip() for j in preamLine.split(" ")[2:]]

    domainDim = np.array([[xMin, xMax],[yMin, yMax],[zMin, zMax]])
    for var in indexVars:
        indexes.append(colTypes.index(var))  
        
    xSphere = np.zeros(numObjects)
    ySphere = np.zeros(numObjects)
    zSphere = np.zeros(numObjects)
    rSphere = np.zeros(numObjects)

    c = 0
    for line in Lines:
        # NEED THIS CORRECTION IF USING UNWRAPPED COORDINATES, maybe flag?
        # x = float(line.split(" ")[indexes[0]])-xD*math.floor((float(line.split(" ")[indexes[0]])-xMin)/xD)
        # y = float(line.split(" ")[indexes[1]])-yD*math.floor((float(line.split(" ")[indexes[1]])-yMin)/yD)
        # z = float(line.split(" ")[indexes[2]])-zD*math.floor((float(line.split(" ")[indexes[2]])-zMin)/zD)
        
        x = float(line.split(" ")[indexes[0]])
        y = float(line.split(" ")[indexes[1]])
        z = float(line.split(" ")[indexes[2]])

        # Only keep atom if it is in the Domain, by a margin of the largest possible 
        # radius. Done because subsampled region might not be the same size as the domain 
        xSphere[c] = x
        ySphere[c] = y
        zSphere[c] = z
        rSphere[c] = float(sigmaLJ[int(line.split(" ")[indexes[3]])-1])
        c += 1

    sphereData = np.zeros([4, c])
    sphereData[0,:] = xSphere[:c]
    sphereData[1,:] = ySphere[:c]
    sphereData[2,:] = zSphere[:c]
    sphereData[3,:] = rSphere[:c]*rSphere[:c]
    domainFile.close()

    if [2,2] in boundaries: 
        sphereData = periodicImageSphereData(sphereData,domainDim,boundaries, nodes)

    if boundaryLims:
        domainDim = np.array([[max(xMin,boundaryLims[0][0]), min(xMax,boundaryLims[0][1])],
                        [max(yMin,boundaryLims[1][0]), min(yMax,boundaryLims[1][1])],
                        [max(zMin,boundaryLims[2][0]), min(zMax,boundaryLims[2][1])]])
        sphereData = subSampleSphereData(sphereData, domainDim, nodes)

    return domainDim, sphereData



def subSampleSphereData(sphereData, domainDim, nodes):
    numObjects = sphereData.shape[1]
    xSphere = np.zeros(numObjects)
    ySphere = np.zeros(numObjects)
    zSphere = np.zeros(numObjects)
    rSphere = np.zeros(numObjects)

    domainLength = [domainDim[0,1]-domainDim[0,0],
                    domainDim[1,1]-domainDim[1,0],
                    domainDim[2,1]-domainDim[2,0]]
    res = [domainLength[0]/nodes[0],domainLength[1]/nodes[1],domainLength[2]/nodes[2]]
    
    c = 0
    for i in range(numObjects):
        
        x = sphereData[0,i]
        y = sphereData[1,i]
        z = sphereData[2,i]
        r = sphereData[3,i]
        r_sqrt = np.sqrt(r)
        # Only keep atom if it is in the Domain, by a margin of 2x the
        # radius. Done because subsampled region might not be the same size as the domain 
        xCheck = domainDim[0][0]-r_sqrt-res[0] <= x <= domainDim[0][1]+r_sqrt+res[0]
        yCheck = domainDim[1][0]-r_sqrt-res[1] <= y <= domainDim[1][1]+r_sqrt+res[1]
        zCheck = domainDim[2][0]-r_sqrt-res[2] <= z <= domainDim[2][1]+r_sqrt+res[2]
        if xCheck and yCheck and zCheck:
            xSphere[c] = x
            ySphere[c] = y
            zSphere[c] = z
            rSphere[c] = r
            c += 1

    sampleSphereData = np.zeros([4, c])
    sampleSphereData[0,:] = xSphere[:c]
    sampleSphereData[1,:] = ySphere[:c]
    sampleSphereData[2,:] = zSphere[:c]
    sampleSphereData[3,:] = rSphere[:c]

    return sampleSphereData


def periodicImageSphereData(sphereData,domainDim,boundaries,nodes):
    numObj = sphereData.shape[1]
    replicateSphereData = [[],[],[],[]]
    newSphereData = [[],[],[],[]]
    domainLength = [domainDim[0,1]-domainDim[0,0],
                        domainDim[1,1]-domainDim[1,0],
                        domainDim[2,1]-domainDim[2,0]]
    res = [domainLength[0]/nodes[0],domainLength[1]/nodes[1],domainLength[2]/nodes[2]]
    for i in range(numObj):
        x = sphereData[0,i]
        y = sphereData[1,i]
        z = sphereData[2,i]
        r = sphereData[3,i]
        r_sqrt = np.sqrt(sphereData[3,i])
        nearBotX = x-r_sqrt-res[0] <= domainDim[0,0]
        nearTopX = x+r_sqrt+res[0] >= domainDim[0,1]
        nearBotY = y-r_sqrt-res[1] <= domainDim[1,0]
        nearTopY = y+r_sqrt+res[1] >= domainDim[1,1]
        nearBotZ = z-r_sqrt-res[2] <= domainDim[2,0]
        nearTopZ = z+r_sqrt+res[2] >= domainDim[2,1]
        checks = [nearBotX, nearTopX,
                nearBotY, nearTopY,
                nearBotZ, nearTopZ]
        coords = [x,y,z]
        if not any(checks):
            continue
        if boundaries[2]==[2,2]:
            if checks[4]:
                shiftedZ = z+domainLength[2]
                replicateSphereData[0].append(x)
                replicateSphereData[1].append(y)
                replicateSphereData[2].append(shiftedZ)
                replicateSphereData[3].append(r)                
            elif checks [5]:
                shiftedZ = z-domainLength[2]
                replicateSphereData[0].append(x)
                replicateSphereData[1].append(y)
                replicateSphereData[2].append(shiftedZ)
                replicateSphereData[3].append(r)
                    
        if boundaries[0]==[2,2]:
            if checks[0]:
                shiftedX = x+domainLength[0]
                replicateSphereData[0].append(shiftedX)
                replicateSphereData[1].append(y)
                replicateSphereData[2].append(z)
                replicateSphereData[3].append(r)                
            elif checks[1]:
                shiftedX = x-domainLength[0]
                replicateSphereData[0].append(shiftedX)
                replicateSphereData[1].append(y)
                replicateSphereData[2].append(z)
                replicateSphereData[3].append(r)
            else:
                shiftedX = x
        
            if boundaries[1]==[2,2]:
                if checks[2]:
                    shiftedY = y+domainLength[1]
                    replicateSphereData[0].append(shiftedX)
                    replicateSphereData[1].append(shiftedY)
                    replicateSphereData[2].append(z)
                    replicateSphereData[3].append(r)
                elif checks [3]:
                    shiftedY = y-domainLength[1]
                    replicateSphereData[0].append(shiftedX)
                    replicateSphereData[1].append(shiftedY)
                    replicateSphereData[2].append(z)
                    replicateSphereData[3].append(r)
                else:
                    shiftedY = y
        
                if boundaries[2]==[2,2]:
                    if checks[4]:
                        shiftedZ = z+domainLength[2]
                        replicateSphereData[0].append(shiftedX)
                        replicateSphereData[1].append(shiftedY)
                        replicateSphereData[2].append(shiftedZ)
                        replicateSphereData[3].append(r)
                    elif checks [5]:
                        shiftedZ = z-domainLength[2]
                        replicateSphereData[0].append(shiftedX)
                        replicateSphereData[1].append(shiftedY)
                        replicateSphereData[2].append(shiftedZ)
                        replicateSphereData[3].append(r)
            if boundaries[2]==[2,2]:
                if checks[4]:
                    shiftedZ = z+domainLength[2]
                    replicateSphereData[0].append(shiftedX)
                    replicateSphereData[1].append(y)
                    replicateSphereData[2].append(shiftedZ)
                    replicateSphereData[3].append(r)
                elif checks [5]:
                    shiftedZ = z-domainLength[2]
                    replicateSphereData[0].append(shiftedX)
                    replicateSphereData[1].append(y)
                    replicateSphereData[2].append(shiftedZ)
                    replicateSphereData[3].append(r)
        if boundaries[1]==[2,2]:
            if checks[2]:
                shiftedY = y+domainLength[1]
                replicateSphereData[0].append(x)
                replicateSphereData[1].append(shiftedY)
                replicateSphereData[2].append(z)
                replicateSphereData[3].append(r)                
            elif checks [3]:
                shiftedY = y-domainLength[1]
                replicateSphereData[0].append(x)
                replicateSphereData[1].append(shiftedY)
                replicateSphereData[2].append(z)
                replicateSphereData[3].append(r)
            else:
                shiftedY = y
            if boundaries[2]==[2,2]:
                if checks[4]:
                    shiftedZ = z+domainLength[2]
                    replicateSphereData[0].append(x)
                    replicateSphereData[1].append(shiftedY)
                    replicateSphereData[2].append(shiftedZ)
                    replicateSphereData[3].append(r)
                elif checks [5]:
                    shiftedZ = z-domainLength[2]
                    replicateSphereData[0].append(x)
                    replicateSphereData[1].append(shiftedY)
                    replicateSphereData[2].append(shiftedZ)
                    replicateSphereData[3].append(r)
    newSphereData[0] = sphereData[0].tolist()+replicateSphereData[0]
    newSphereData[1] = sphereData[1].tolist()+replicateSphereData[1]
    newSphereData[2] = sphereData[2].tolist()+replicateSphereData[2]
    newSphereData[3] = sphereData[3].tolist()+replicateSphereData[3]
    return np.array(newSphereData)