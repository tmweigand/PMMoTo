"""test_pmmoto.py"""

import pmmoto


def test_deconstruct_grid(generate_single_subdomain):
    """Ensure expected behavior of deconstruct_grid"""
    sd = generate_single_subdomain(0, periodic=False)
    img = pmmoto.domain_generation.gen_smoothed_random_binary_grid(
        sd.voxels, p_zero=0.5, seed=1, smoothness=2
    )

    print(sd.voxels, img.shape)

    pmmoto.io.output.save_grid_data_serial("data_out/test_deconstruct_grid", sd, img)

    subdomains, local_img = pmmoto.core.pmmoto.deconstruct_grid(
        sd, img, subdomains=(2, 2, 2)
    )

    pmmoto.io.output.save_grid_data_serial(
        "data_out/test_deconstruct_grid_reconstructed", subdomains, local_img
    )
