
#### Generate Slice
slice = 0 # 0 = x, 1= y, 2 = z
sliceLoc = domain.domainSize[0,1]/2

if slice == 0:
    sliceLow  = sliceLoc - domain.dX
    sliceHigh = sliceLoc + domain.dX
    indID = np.where( (sDL.x > sliceLow) & (sDL.x < sliceHigh) )[0]
    totalNodes = indID.size*(sDL.ownNodes[1][1]-sDL.ownNodes[1][0])*(sDL.ownNodes[2][1]-sDL.ownNodes[2][0])
    printGridOut = np.zeros([totalNodes,4])

    c = 0
    for i in indID:
        for j in range(sDL.ownNodes[1][0],sDL.ownNodes[1][1]):
            for k in range(sDL.ownNodes[2][0],sDL.ownNodes[2][1]):
                printGridOut[c,0] = sDL.x[i]
                printGridOut[c,1] = sDL.y[j]
                printGridOut[c,2] = sDL.z[k]
                printGridOut[c,3] = drainL.nwpFinal[i,j,k]
                c = c + 1
    header = "x,y,z,NWPFinal"
    file = "dataDump/3dSlice_"+str(rank)+".csv"
    np.savetxt(file,printGridOut, delimiter=',',header=header)

elif slice == 1:
    sliceLow  = sliceLoc - domain.dY
    sliceHigh = sliceLoc + domain.dY
    indID = np.where( (sDL.y > sliceLow) & (sDL.y < sliceHigh) )

    totalNodes = indID.size*(sDL.ownNodes[0][1]-sDL.ownNodes[0][0])*(sDL.ownNodes[2][1]-sDL.ownNodes[2][0])
    printGridOut = np.zeros([totalNodes,4])

    c = 0
    for i in range(sDL.ownNodes[0][0],sDL.ownNodes[0][1]):
        for j in indID:
            for k in range(sDL.ownNodes[2][0],sDL.ownNodes[2][1]):
                printGridOut[c,0] = sDL.x[i]
                printGridOut[c,1] = sDL.y[j]
                printGridOut[c,2] = sDL.z[k]
                printGridOut[c,3] = drainL.nwpFinal[i,j,k]
                c = c + 1
    header = "x,y,z,NWPFinal"
    file = "dataDump/3dSlice_"+str(rank)+".csv"
    np.savetxt(file,printGridOut, delimiter=',',header=header)


elif slice == 2:
    sliceLow  = sliceLoc - domain.dZ
    sliceHigh = sliceLoc + domain.dZ
    indID = np.where( (sDL.z > sliceLow) & (sDL.z < sliceHigh) )

    totalNodes = indID.size*(sDL.ownNodes[0][1]-sDL.ownNodes[0][0])*(sDL.ownNodes[1][1]-sDL.ownNodes[1][0])
    printGridOut = np.zeros([totalNodes,4])

    c = 0
    for i in range(sDL.ownNodes[0][0],sDL.ownNodes[0][1]):
        for j in range(sDL.ownNodes[1][0],sDL.ownNodes[1][1]):
            for k in indID:
                printGridOut[c,0] = sDL.x[i]
                printGridOut[c,1] = sDL.y[j]
                printGridOut[c,2] = sDL.z[k]
                printGridOut[c,3] = drainL.nwpFinal[i,j,k]
                c = c + 1
    header = "x,y,z,NWPFinal"
    file = "dataDump/3dSlice_"+str(rank)+".csv"
    np.savetxt(file,printGridOut, delimiter=',',header=header)
