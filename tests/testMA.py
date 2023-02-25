import numpy as np
from mpi4py import MPI
from scipy import ndimage
import scipy.ndimage as ndi
from scipy.ndimage import distance_transform_edt
from scipy.spatial import KDTree
import edt
import PMMoTo
from skimage.morphology import skeletonize


def my_function():

    nodes = [50,50,50]
    inlet  = [1,0,0]
    outlet = [-1,0,0]
    boundaries = [0,0,0]
    res = 1
    file = 'testDomains/10pack.out'
    domainFile = open(file, 'r')

    rank = 0; subDomains = [1,1,1]; size = 1
    domain,sDL = PMMoTo.genDomainSubDomain(rank,size,subDomains,nodes,boundaries,inlet,outlet,res,"Sphere",domainFile,PMMoTo.readPorousMediaXYZR)


    domainFile = open(file, 'r')
    _,sphereData = PMMoTo.readPorousMediaXYZR(domainFile)

    ##### To GENERATE SINGLE PROC TEST CASE ######
    x = np.linspace(domain.dX/2, domain.domainSize[0,1]-domain.dX/2, nodes[0])
    y = np.linspace(domain.dY/2, domain.domainSize[1,1]-domain.dY/2, nodes[1])
    z = np.linspace(domain.dZ/2, domain.domainSize[2,1]-domain.dZ/2, nodes[2])

    gridOut = PMMoTo.domainGen(x,y,z,sphereData)
    gridOut = np.asarray(gridOut)

    pG = [0,0,0]
    pgSize = nodes[0]

    if boundaries[0] == 1:
        pG[0] = 1
    if boundaries[1] == 1:
        pG[1] = 1
    if boundaries[2] == 1:
        pG[2] = 1

    periodic = [False,False,False]
    if boundaries[0] == 2:
        periodic[0] = True
        pG[0] = pgSize
    if boundaries[1] == 2:
        periodic[1] = True
        pG[1] = pgSize
    if boundaries[2] == 2:
        periodic[2] = True
        pG[2] = pgSize

    gridOut = np.pad (gridOut, ((pG[0], pG[0]), (pG[1], pG[1]), (pG[2], pG[2])), 'wrap')

    print(pG)

    if boundaries[0] == 1:
        gridOut[0,:,:] = 0
        gridOut[-1,:,:] = 0
    if boundaries[1] == 1:
        gridOut[:,0,:] = 0
        gridOut[:,-1,:] = 0
    if boundaries[2] == 1:
        gridOut[:,:,0] = 0
        gridOut[:,:,-1] = 0



    realDT = edt.edt3d(gridOut, anisotropy=(domain.dX, domain.dY, domain.dZ))
    edtV,indTrue = distance_transform_edt(gridOut,sampling=[domain.dX, domain.dY, domain.dZ],return_indices=True)
    gridCopy = np.copy(gridOut)

    realMA = PMMoTo.medialAxis.skeletonize._compute_thin_image(gridCopy)

    if pG[0] > 0 and pG[1]==0 and pG[2]==0:
        gridOut = gridOut[pG[0]:-pG[0],:,:]
        realDT = realDT[pG[0]:-pG[0],:,:]
        edtV = edtV[pG[0]:-pG[0],:,:]
        realMA = realMA[pG[0]:-pG[0],:,:]

    elif pG[0]==0 and pG[1] > 0 and pG[2]==0:
        gridOut = gridOut[:,pG[1]:-pG[1],:]
        realDT = realDT[:,pG[1]:-pG[1],:]
        edtV = edtV[:,pG[1]:-pG[1],:]
        realMA = realMA[:,pG[1]:-pG[1],:]

    elif pG[0]==0 and pG[1]==0 and pG[2] > 0:
        gridOut = gridOut[:,:,pG[2]:-pG[2]]
        realDT = realDT[:,:,pG[2]:-pG[2]]
        edtV = edtV[:,:,pG[2]:-pG[2]]
        realMA = realMA[:,:,pG[2]:-pG[2]]

    elif pG[0] > 0 and pG[1]==0 and pG[2] > 0:
        gridOut = gridOut[pG[0]:-pG[0],:,pG[2]:-pG[2]]
        realDT = realDT[pG[0]:-pG[0],:,pG[2]:-pG[2]]
        edtV = edtV[pG[0]:-pG[0],:,pG[2]:-pG[2]]
        realMA = realMA[pG[0]:-pG[0],:,pG[2]:-pG[2]]

    elif pG[0] > 0 and pG[1] > 0 and pG[2]==0:
        gridOut = gridOut[pG[0]:-pG[0],pG[1]:-pG[1],:]
        realDT = realDT[pG[0]:-pG[0],pG[1]:-pG[1],:]
        edtV = edtV[pG[0]:-pG[0],pG[1]:-pG[1],:]
        realMA = realMA[pG[0]:-pG[0],pG[1]:-pG[1],:]

    elif pG[0]==0 and pG[1] > 0 and pG[2] > 0:
        gridOut = gridOut[:,pG[1]:-pG[1],pG[2]:-pG[2]]
        realDT = realDT[:,pG[1]:-pG[1],pG[2]:-pG[2]]
        edtV = edtV[:,pG[1]:-pG[1],pG[2]:-pG[2]]
        realMA = realMA[:,pG[1]:-pG[1],pG[2]:-pG[2]]

    elif pG[0] > 0 and pG[1] > 0 and pG[2] > 0:
        gridOut = gridOut[pG[0]:-pG[0],pG[1]:-pG[1],pG[2]:-pG[2]]
        realDT = realDT[pG[0]:-pG[0],pG[1]:-pG[1],pG[2]:-pG[2]]
        edtV = edtV[pG[0]:-pG[0],pG[1]:-pG[1],pG[2]:-pG[2]]
        realMA = realMA[pG[0]:-pG[0],pG[1]:-pG[1],pG[2]:-pG[2]]
    ####################################################################

    c = 0
    printGridOut = np.zeros([gridOut.size,6])
    for i in range(0,gridOut.shape[0]):
        for j in range(0,gridOut.shape[1]):
            for k in range(0,gridOut.shape[2]):
                printGridOut[c,0] = i#x[i]
                printGridOut[c,1] = j#y[j]
                printGridOut[c,2] = k#z[k]
                ci = i 
                cj = j 
                ck = k
                printGridOut[c,3] = gridOut[ci,cj,ck]
                printGridOut[c,4] = realMA[ci,cj,ck]
                printGridOut[c,5] = realDT[ci,cj,ck]
                c = c + 1

    #header = "x,y,z,RealMA,CheckMA,GRID"#,Grid,Dist"
    header = "i,j,k,Grid,MA,Dist"#,Grid,Dist"
    file = "dataDump/3GridOrder.csv"
    np.savetxt(file,printGridOut, delimiter=',',header=header)


if __name__ == "__main__":
    my_function()
