"""equilibrium_distribution.py"""

import numpy as np
from .porosimetry import porosimetry
from .distance import edt
from . import morphological_operators
from . import connected_components
from ..core import utils


def drainage(capillary_pressure, multiphase):
    """
    This is a morphological approach to determining the equilibrium
    fluid distribution for a multiphase system
    Args:
        capillary_pressure (_type_): _description_
        multiphase (_type_): _description_

    Returns:
        _type_: _description_
    """
    if capillary_pressure == 0:
        return multiphase.porous_media.img, 1.0

    radius = multiphase.get_probe_radius(capillary_pressure)

    img_temp = porosimetry(
        subdomain=multiphase.subdomain,
        img=multiphase.porous_media.img,
        radius=radius,
        inlet=True,
        mode="morph",
    )

    img_out = np.where(
        (multiphase.porous_media.img == 1) & (img_temp == 0), 2, img_temp
    )

    multiphase.update_img(img_out)
    w_saturation = multiphase.get_saturation(2)

    return img_out, w_saturation


def calcDrainage(capillary_pressure, multiphase):

    ### Get Distance from Solid to Pore Space (Ignore Fluid Phases)
    poreSpaceDist = edt(multiphase.pm_img, multiphase.subdomain)

    ### Get Sphere Radius from Pressure
    radius = multiphase.get_probe_radius(capillary_pressure)

    # Step 1 - Reservoirs are not contained in mdGrid or grid but rather added when needed so this step is unnecessary

    # Step 2 - Dilate Solid Phase and Flag Allowable Fluid Voxels as 1
    ind = np.where((poreSpaceDist >= radius) & (multiphase.pm_img == 1), 1, 0).astype(
        np.uint8
    )
    print("ind sum: ", np.sum(ind))
    continueFlag = utils.phase_exists(ind, 1)
    ind = multiphase.subdomain.update_reservoir(ind, 1)
    if continueFlag:
        nwCheck = utils.phase_exists(multiphase.img, 1)
        # nwCheck = eqDist.checkPoints(mP.mpGrid, mP.nwID)
        if nwCheck:
            # nwGrid = connected_components.inlet_connected_img(
            #     multiphase.subdomain, multiphase.img
            # )
            n_connected = connected_components.inlet_connected_img(
                multiphase.subdomain, multiphase.img, 1
            )

        #     # Step 3b and 3d- Check if W Phases Exists then Collect W Sets
        wCheck = utils.phase_exists(multiphase.img, 2)

        # wCheck = eqDist.checkPoints(mP.mpGrid, mP.wID)
        if wCheck:

            # wGrid = connected_components.inlet_connected_img(
            #     multiphase.subdomain, multiphase.img
            # )

            w_connected = connected_components.outlet_connected_img(
                multiphase.subdomain, multiphase.img, 2
            )

        ind2 = connected_components.inlet_connected_img(multiphase.subdomain, ind)
        ind2 = multiphase.subdomain.update_reservoir(ind2, 0)

        if nwCheck:
            ind = np.where((ind2 != 1) & (n_connected != 1), 0, ind).astype(np.uint8)
            morph = morphological_operators.addition(multiphase.subdomain, ind, radius)
            # morph = morphology.morph(ind, mP.subDomain, eqDist.probeR)
        else:
            morph = morphological_operators.addition(
                multiphase.subdomain, ind2, radius, fft=True
            )
        # morph = morphology.morph(ind2, mP.subDomain, eqDist.probeR)

        # Turn wetting films on or off here
        multiphase.img = np.where((morph == 1) & (w_connected == 2), 1, multiphase.img)

        # Step 4
        multiphase.img = multiphase.subdomain.update_reservoir(multiphase.img, 1)
        multiphase.update_img(multiphase.img)
        w_saturation = multiphase.get_saturation(2)

    else:
        w_saturation = 1

    return multiphase.img, w_saturation

    # if save:
    #     fileName = "dataOut/twoPhase/twoPhase_drain_pc_" + str(p)
    #     dataOutput.saveGrid(fileName, mP.subDomain, mP.mpGrid)

    # return eqDist, result
