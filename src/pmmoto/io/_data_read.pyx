import numpy as np
cimport numpy as np
from libcpp.string cimport string
from libcpp.map cimport map
from libcpp.pair cimport pair
from libc.stdint cimport uint64_t,uint8_t

np.import_array()

cdef class AtomIdMapWrapper:
    def __cinit__(self):
        self._c_map_ptr = new AtomIdMap()

    def __dealloc__(self):
        if self._c_map_ptr is not NULL:
            del self._c_map_ptr
            self._c_map_ptr = NULL

# Convert Python dict[(int,float)->int] -> C++ AtomIdMap (returned by value)
cdef AtomIdMap dict_to_atom_id_map(object py_map):
    cdef AtomIdMap c_map
    cdef object py_items = py_map.items()
    cdef object py_key, py_val
    cdef int k_type
    cdef double k_charge
    cdef int v
    cdef pair[int, double] pkey
    cdef pair[pair[int, double], int] entry
    for py_key, py_val in py_items:
        # expect py_key to be a (int, float) tuple
        k_type = <int>py_key[0]
        k_charge = <double>py_key[1]
        v = <int>py_val
        pkey = pair[int, double](k_type, k_charge)
        entry = pair[pair[int, double], int](pkey, v)
        c_map.insert(entry)
    return c_map

def read_lammps_atoms(str filename, type_map=None):
    cdef string cpp_filename = filename.encode('utf-8')
    cdef LammpsData data
    cdef const AtomIdMap* _type_map = NULL
    cdef AtomIdMap tmp_map
    cdef AtomIdMapWrapper wrapper

    if type_map is not None:
        # if user passed our wrapper, use its pointer directly
        if isinstance(type_map, AtomIdMapWrapper):
            wrapper = <AtomIdMapWrapper> type_map
            _type_map = wrapper._c_map_ptr
        else:
            # convert Python dict to C++ map and pass pointer to it
            tmp_map = dict_to_atom_id_map(type_map)
            _type_map = &tmp_map

    data = LammpsReader.read_lammps_atoms(cpp_filename,_type_map)

    cdef size_t n_atoms = data.atom_ids.size()

    # Zero-copy view of the C++ double vector
    cdef double* pos_ptr = data.atom_positions.data()
    cdef np.ndarray[np.double_t, ndim=2] positions = np.frombuffer(
        (<char*> pos_ptr)[:n_atoms * 3 * sizeof(double)],
        dtype=np.float64
    ).reshape(n_atoms, 3)

    # Zero-copy view for atom_ids
    cdef uint64_t* ids_ptr = <uint64_t*> data.atom_ids.data()
    ids = np.frombuffer(
        (<char*> ids_ptr)[:n_atoms * sizeof(uint64_t)],
        dtype=np.uint64
    )

    # Zero-copy view for atom_types
    cdef uint8_t* types_ptr = <uint8_t*> data.atom_types.data()
    types = np.frombuffer(
        (<char*> types_ptr)[:n_atoms * sizeof(uint8_t)],
        dtype=np.uint8
    )

    domain = np.array(data.domain_data, dtype=np.float64)

    return ids, positions, types, domain, data.timestep