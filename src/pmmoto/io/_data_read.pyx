import numpy as np
cimport numpy as np
from libcpp.string cimport string
from libcpp.vector cimport vector
from libc.string cimport memcpy

np.import_array()

def read_lammps_atoms(str filename, dict data_columns):
    """Read the requested data_columns of a LAMMPS dump file.

    data_columns maps a column name (as in the "ITEM: ATOMS" header) to a dtype.
    Integer dtypes are parsed as integers, floating dtypes as doubles.
    Returns (dict name -> array, domain, timestep).
    """
    cdef string cpp_filename = filename.encode('utf-8')
    cdef vector[LammpsColumnSpec] specs
    cdef LammpsColumnSpec spec
    cdef LammpsData data
    cdef size_t j, n
    cdef np.int64_t[::1] int_view
    cdef np.float64_t[::1] double_view
    cdef dict dtypes = {}

    if not data_columns:
        raise ValueError("columns must contain at least one column")

    for name, dtype in data_columns.items():
        if not isinstance(name, str):
            raise TypeError(f"Column names must be str, got {type(name).__name__}")
        np_dtype = np.dtype(dtype)
        if np_dtype.kind in "biu":
            spec.is_int = True
        elif np_dtype.kind == "f":
            spec.is_int = False
        else:
            raise TypeError(
                f"Unsupported dtype {np_dtype} for column '{name}'; use int or float"
            )
        spec.name = name.encode('utf-8')
        specs.push_back(spec)
        dtypes[name] = np_dtype

    data = LammpsReader.read_lammps_atoms(cpp_filename, specs)

    result = {}
    for j in range(data.data_columns.size()):
        name = data.data_columns[j].name.decode('utf-8')
        if data.data_columns[j].is_int:
            n = data.data_columns[j].int_values.size()
            values = np.empty(n, dtype=np.int64)
            if n > 0:
                int_view = values
                memcpy(&int_view[0], data.data_columns[j].int_values.data(), n * sizeof(np.int64_t))
        else:
            n = data.data_columns[j].double_values.size()
            values = np.empty(n, dtype=np.float64)
            if n > 0:
                double_view = values
                memcpy(&double_view[0], data.data_columns[j].double_values.data(), n * sizeof(np.float64_t))
        result[name] = values.astype(dtypes[name], copy=False)

    domain = np.array(data.domain_data, dtype=np.float64)

    return result, domain, data.timestep