import numpy as np
cimport numpy as np
from libcpp.string cimport string
from libcpp.vector cimport vector
from libcpp.map cimport map
from libcpp.pair cimport pair

# Initialize numpy
np.import_array()

def read_lammps_atoms(str filename, type_map=None):
    """
    Read LAMMPS atom data from file
    
    Args:
        filename (str): Path to LAMMPS data file
        type_map (dict, optional): Mapping of (type, charge) pairs to new types
            Example: {(1, 0.4): 2, (1, -0.4): 3}
        
    Returns:
        tuple: (positions, types, domain_data, timestep)
            - positions: np.ndarray of shape (n_atoms, 3)
            - types: np.ndarray of shape (n_atoms,)
            - domain_data: np.ndarray of shape (3, 2)
            - timestep: float
    """
    # Convert Python string to C++ string
    cdef string cpp_filename = filename.encode('utf-8')
    
    # Convert Python dict to C++ map if provided
    cdef map[pair[int, double], int] cpp_type_map
    if type_map is not None:
        cpp_type_map = type_map
    
    # Call C++ function
    cdef LammpsData data
    if type_map is not None:
        data = LammpsReader.read_lammps_atoms_with_map(cpp_filename, cpp_type_map)
    else:
        data = LammpsReader.read_lammps_atoms(cpp_filename)
    
    # Convert C++ vectors to numpy arrays
    cdef np.ndarray[double, ndim=2] positions = np.array(data.atom_positions)
    cdef np.ndarray[long, ndim=1] types = np.array(data.atom_types)
    cdef np.ndarray[double, ndim=2] domain = np.array(data.domain_data)
    
    return positions, types, domain, data.timestep