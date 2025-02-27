import numpy as np
import pmmoto


def my_function():

    voxels = [140, 30, 30]  # Total Number of Nodes in Domain
    box = np.array([[0.0, 14.0], [-1.5, 1.5], [-1.5, 1.5]])
    sd = pmmoto.initialize(voxels, box=box)
    pm = pmmoto.domain_generation.gen_pm_inkbottle(sd)

    capillary_pressure = [
        1.58965,
        1.59430,
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
        2.46250,
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
        6.99090,
        7.36005,
        7.60403,
        7.69393,
        8.0,
    ]

    radii = pmmoto.filters.porosimetry.get_radii(capillary_pressure, gamma=1)

    print(radii)

    ### Save Grid Data where kwargs are used for saving other grid data (i.e. EDT, Medial Axis)
    # pmmoto.io.save_img_data_parallel("data_out/test_inkbottle", sd, pm.img)


if __name__ == "__main__":
    my_function()
