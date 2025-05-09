import numpy as np
from mpi4py import MPI
import time
import PMMoTo


def my_function():
    comm = MPI.COMM_WORLD
    size = comm.Get_size()
    rank = comm.Get_rank()

    if rank == 0:
        start_time = time.time()

    subDomains = [2, 2, 2]
    # nodes = [70,15,15]  ##res = 10
    # nodes = [140, 60, 60]  ##res = 10
    # nodes = [280,60,60]   ##res = 20
    nodes = [560, 120, 120]  ##res = 40
    # nodes = [1120,240,240] ##res = 80
    # nodes = [1680,360,360] ##res = 120

    res = 10

    boundaries = [[0, 0], [0, 0], [0, 0]]  # 0: Nothing Assumed  1: Walls 2: Periodic
    inlet = [[0, 0], [0, 0], [0, 0]]
    outlet = [[0, 0], [0, 0], [0, 0]]

    domain, sDL, pML = PMMoTo.genDomainSubDomain(
        rank,
        size,
        subDomains,
        nodes,
        boundaries,
        inlet,
        outlet,
        "InkBottle",
        None,
        None,
    )
    numFluidPhases = 2
    twoPhase = PMMoTo.multiPhase.multiPhase(pML, numFluidPhases)

    wRes = [[1, 0], [0, 0], [0, 0]]
    nwRes = [[0, 1], [0, 0], [0, 0]]
    mpInlets = {twoPhase.wID: wRes, twoPhase.nwID: nwRes}

    wOut = [[0, 0], [0, 0], [0, 0]]
    nwOut = [[0, 0], [0, 0], [0, 0]]
    mpOutlets = {twoPhase.wID: wOut, twoPhase.nwID: nwOut}

    pC = [
        1.58965,
        1.5943,
        1.60194,
        1.61322,
        1.62893,
        1.65002,
        1.67755,
        1.7127,
        1.75678,
        1.81122,
        1.87764,
        1.95783,
        2.05388,
        2.16814,
        2.30332,
        2.4625,
        2.64914,
        2.86704,
        3.12024,
        3.41274,
        3.74806,
        4.12854,
        4.55421,
        5.02123,
        5.52008,
        6.03352,
        6.53538,
        6.9909,
        7.36005,
        7.60403,
        7.69393,
        7.69409,
    ]

    # pC = [7,6,5,4,3,2,1.90,1.8,1.79,1.78,1.77,1.76,1.75,1.74,1.73,1.72,1.71,1.70,1.69,1.68832, 1.65837, 1.63524, 1.61783, 1.60516, 1.59638, 1.59079, 1.58789, 1.5873, 1.5872]

    print("Run Drainage")
    # #Initialize from previous fluid distribution
    # inputFile = 'dataOut/twoPhase/twoPhase_imbibe_pc_1.61783'
    # twoPhase.initializeMPGrid(inputFile = inputFile)

    # Start from sw = 1
    twoPhase.initializeMPGrid(constantPhase=twoPhase.wID)  ##drainage

    twoPhase.getBoundaryInfo(mpInlets, mpOutlets, resSize=res)
    # drainL = PMMoTo.multiPhase.calcDrainage(pC, twoPhase)

    # print("Run Imbibition")

    # # #Initialize from previous fluid distribution
    # inputFile = 'dataOut/twoPhase/twoPhase_drain_pc_7.69393'
    # twoPhase.initializeMPGrid(inputFile = inputFile)

    # Start from sw = 0
    # twoPhase.initializeMPGrid(constantPhase = twoPhase.nwID) ##imbibition

    # twoPhase.getBoundaryInfo(mpInlets,mpOutlets,resSize = 1)
    # drainL = PMMoTo.multiPhase.calcImbibition(pC,twoPhase)

    # # #Initialize from previous fluid distribution
    # inputFile = 'dataOut/twoPhase/twoPhase_pc_1.67755'
    # twoPhase.initializeMPGrid(inputFile = inputFile)

    if rank == 0:
        print(sDL.__dict__)

    dist = PMMoTo.distance.calcEDT(sDL, pML.grid)

    PMMoTo.saveGridData("dataOut/grid", rank, domain, sDL, pML.grid, **{"dist": dist})


# pC = [7.69409,7.69409,7.69393,7.60403,7.36005,6.9909,6.53538,6.03352,5.52008,5.02123,
# 4.55421,4.12854,3.74806,3.41274,3.12024,2.86704,2.64914,2.4625,2.30332,2.16814,2.05388,
# 1.95783,1.87764,1.81122,1.75678,1.7127,1.67755,1.65002,1.62893,1.61322,1.60194,
# 1.5943,1.58965,1.58964,1.5872,0.]
#
# sWAnalytical = [0.,0.808933,0.808965,0.809662,0.810321,0.810973,0.811641,0.81235,0.813122,
#                 0.813978,0.814941,0.816034,0.817281,0.818703,0.820323,0.822162,0.824238,
#                 0.826565,0.829154,0.832009,0.835131,0.838511,0.842134,0.845978,0.850016,
#                 0.854214,0.858539,0.862957,0.867443,0.871985,0.876591,0.881302,0.886197,1.,1.,1.]
#


if __name__ == "__main__":
    my_function()
