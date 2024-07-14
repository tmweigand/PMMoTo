import pytest
import numpy as np
import pmmoto

def test_decomposed_domain(
        domain,
        decomposed_domain
        ):
    """
    Test decomposition of domain  
    """
    pmmoto_domain = pmmoto.core.Domain(
        domain['size_domain'],
        domain['boundaries'],
        domain['inlet'],
        domain['outlet']
    )

    pmmoto_decomposed_domain = pmmoto.core.DecomposedDomain(
        domain = pmmoto_domain,
        subdomain_map = decomposed_domain['subdomain_map']
        )
    

    pmmoto_index = pmmoto_decomposed_domain.get_subdomain_index(0)
    boundaries = pmmoto_decomposed_domain.get_subdomain_boundaries(pmmoto_index)
    coordinates = pmmoto_decomposed_domain.get_subdomain_coordinates(pmmoto_index)
    inlet = pmmoto_decomposed_domain.get_subdomain_inlet(pmmoto_index)
    outlet = pmmoto_decomposed_domain.get_subdomain_outlet(pmmoto_index)
    
    np.testing.assert_array_equal(pmmoto_decomposed_domain.map.flatten(),np.arange(pmmoto_decomposed_domain.num_subdomains))
    assert len(pmmoto_decomposed_domain.subdomains) == pmmoto_decomposed_domain.num_subdomains

    assert pmmoto_index == (0,0,0)
    np.testing.assert_array_equal(boundaries, np.array([[0,-1],[1,-1],[2,-1]]))
    np.testing.assert_array_equal(coordinates, np.array([[0.,50.],[0.,50.],[0.,50.]]))
    np.testing.assert_array_equal(inlet, np.array([[1,0],[0,0],[0,0]]))
    np.testing.assert_array_equal(outlet, np.array([[0,0],[0,0],[0,0]]))


if __name__ == '__main__':
    test_decomposed_domain()