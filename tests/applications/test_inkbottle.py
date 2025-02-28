import numpy as np
import pmmoto


def my_function():

    voxels = (140, 30, 30)  # Total Number of Nodes in Domain
    voxels = [560, 120, 120]
    box = ((0.0, 14.0), (-1.5, 1.5), (-1.5, 1.5))
    inlet = ((1, 0), (0, 0), (0, 0))
    reservoir_voxels = 0

    sd = pmmoto.initialize(
        voxels, box=box, inlet=inlet, reservoir_voxels=reservoir_voxels
    )

    pm = pmmoto.domain_generation.gen_pm_inkbottle(sd)

    capillary_pressure = [1.68965]
    radii = pmmoto.filters.porosimetry.get_radii(capillary_pressure, gamma=1)
    morph = pmmoto.filters.porosimetry.porosimetry(
        subdomain=sd, img=pm.img, radius=radii[0], mode="morph", inlet=True
    )

    ### Save Grid Data where kwargs are used for saving other grid data (i.e. EDT, Medial Axis)
    pmmoto.io.save_img_data_parallel(
        "data_out/test_inkbottle", sd, pm.img, additional_img={"morph": morph}
    )


if __name__ == "__main__":
    my_function()
