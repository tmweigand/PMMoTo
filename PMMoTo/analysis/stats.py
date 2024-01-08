def genStats(subdomain,data):
    """
    Get Information (non-zero min/max) of distance tranform
    """
    # own = subdomain.index_own_nodes
    # ownEDT =  data[own[0]:own[1],
    #                     own[2]:own[3],
    #                     own[4]:own[5]]
    # distVals,distCounts  = np.unique(ownEDT,return_counts=True)
    # EDTData = [self.subdomain.ID,distVals,distCounts]
    # EDTData = comm.gather(EDTData, root=0)
    # if self.subdomain.ID == 0:
    #     bins = np.empty([])
    #     for d in EDTData:
    #         if d[0] == 0:
    #             bins = d[1]
    #         else:
    #             bins = np.append(bins,d[1],axis=0)
    #         bins = np.unique(bins)

    #     counts = np.zeros_like(bins,dtype=np.int64)
    #     for d in EDTData:
    #         for n in range(0,d[1].size):
    #             ind = np.where(bins==d[1][n])[0][0]
    #             counts[ind] = counts[ind] + d[2][n]

    #     stats = np.stack((bins,counts), axis = 1)
    #     self.minD = bins[1]
    #     self.maxD = bins[-1]
    #     distData = [self.minD,self.maxD]
    #     print("Minimum distance:",self.minD,"Maximum distance:",self.maxD)
    # else:
    #     distData = None
    # distData = comm.bcast(distData, root=0)
    # self.minD = distData[0]
    # self.maxD = distData[1]