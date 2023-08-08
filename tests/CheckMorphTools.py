import numpy as np
from mpi4py import MPI
import time
import PMMoTo


def my_function():
    
    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()

    if rank==0:
        start_time = time.time()

    subDomains = [2,2,2]
    nodes = [280,60,60]  

    boundaries = [[0,0],[0,0],[0,0]] # 0: Nothing Assumed  1: Walls 2: Periodic
    inlet  = [[0,0],[0,0],[0,0]]
    outlet = [[0,0],[0,0],[0,0]]

    domain,sDL,pML = PMMoTo.genDomainSubDomain(rank,size,subDomains,nodes,boundaries,inlet,outlet,"InkBotle",None,None)
 
    numFluidPhases = 2
    twoPhase = PMMoTo.multiPhase.multiPhase(pML,numFluidPhases)

    wRes  = [[1,0],[0,0],[0,0]]
    nwRes = [[0,1],[0,0],[0,0]]
    mpInlets = {twoPhase.wID:wRes,twoPhase.nwID:nwRes}

    wOut  = [[0,0],[0,0],[0,0]]
    nwOut = [[0,0],[0,0],[0,0]]
    mpOutlets = {twoPhase.wID:wOut,twoPhase.nwID:nwOut}
    
    if rank == 0:
        print("Run Ink Bottle Drainage")
    #Initialize wetting saturated domain
    twoPhase.initializeMPGrid(constantPhase = twoPhase.wID) ##drainage
    twoPhase.getBoundaryInfo(mpInlets,mpOutlets,resSize = 10)
    
    pC = [1.58965, 1.59430, 1.60194, 1.61322, 1.62893,
    1.65002, 1.67755, 1.7127, 1.75678, 1.81122, 1.87764, 1.95783, 2.05388,
    2.16814, 2.30332, 2.46250, 2.64914, 2.86704, 3.12024, 3.41274, 3.74806,
    4.12854, 4.55421, 5.02123, 5.52008, 6.03352, 6.53538, 6.99090, 7.36005, 
    7.60403, 7.69393,8.0] 

    drainL,result = PMMoTo.multiPhase.calcDrainage(pC,twoPhase)
    
    sW = [9.155640e-01,9.091834e-01,9.091834e-01,9.052943e-01,8.937182e-01,
    8.829472e-01,8.739689e-01,8.657501e-01,8.590201e-01,8.536574e-01,
    8.490847e-01,8.454994e-01,8.425066e-01,8.366730e-01,8.348348e-01,
    8.306419e-01,8.293506e-01,8.261147e-01,8.250513e-01,8.226814e-01,
    8.219977e-01,8.199620e-01,8.191417e-01,8.176984e-01,8.169844e-01,
    8.156931e-01,8.155412e-01,8.149487e-01,8.141588e-01,8.138853e-01,
    8.138853e-01,3.539689e-03]
    
    # Check if both lists have the same length
    if rank == 0:
        if len(sW) != len(result):
            print("Error: The lists have different lengths.")
        else:
            result_rounded = ["{:.6e}".format(num) for num in result]
            sW = ["{:.6e}".format(num) for num in sW]
            # Compare each entry in 'sW' to the corresponding entry in 'result'
            found_difference = False
            for i in range(len(sW)):
                if sW[i] != result_rounded[i]:
                    print(f"Error: Non-equivalent entry at index {i}. sW[{i}] = {sW[i]}, result[{i}] = {result_rounded[i]}")
                    found_difference = True
            if not found_difference:
                print("PASS: Ink Bottle Morphological Drainage")
            else:
                print("FAIL: Ink Bottle Morphological Drainage")

    
    if rank == 0:
        print("Run Ink Bottle Imbibition")
        
    twoPhase.initializeMPGrid(constantPhase = twoPhase.nwID)
    twoPhase.getBoundaryInfo(mpInlets,mpOutlets,resSize = 1)

    pC = [1.69,1.68832, 1.65837, 1.63524, 1.61783, 1.60516, 1.59638, 1.59079,
    1.58789, 1.5873, 1.5872]

    drainL,result = PMMoTo.multiPhase.calcImbibition(pC,twoPhase)
    
    sW = [2.743638e-02,2.743638e-02,3.632359e-02,4.666920e-02,1.000000e+00,1.000000e+00,
    1.000000e+00,1.000000e+00,1.000000e+00,1.000000e+00,1.000000e+00]
    
    # Check if both lists have the same length
    if rank == 0:
        if len(sW) != len(result):
            print("Error: The lists have different lengths.")
        else:
            result_rounded = ["{:.6e}".format(num) for num in result]
            sW = ["{:.6e}".format(num) for num in sW]
            # Compare each entry in 'sW' to the corresponding entry in 'result'
            found_difference = False
            for i in range(len(sW)):
                if sW[i] != result_rounded[i]:
                    print(f"Error: Non-equivalent entry at index {i}. sW[{i}] = {sW[i]}, result[{i}] = {result_rounded[i]}")
                    found_difference = True
            if not found_difference:
                print("PASS: Ink Bottle Morphological Imbibition")
            else:
                print("FAIL: Ink Bottle Morphological Imbibition")
    
    


 




if __name__ == "__main__":
    my_function()
    MPI.Finalize()
