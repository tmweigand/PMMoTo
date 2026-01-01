"""Setup script for PMMoTo.

Builds Cython/C++ extensions and installs the PMMoTo package.
"""

import sys
from setuptools import Extension, setup
from Cython.Build import cythonize
import Cython.Compiler.Options
import numpy

Cython.Compiler.Options.annotate = True

extra_compile_args = []

if sys.platform == "win32":
    # MSVC
    extra_compile_args += ["/std:c++17", "/O2"]
else:
    # GCC / Clang
    extra_compile_args += ["-std=c++17", "-O3", "-pthread"]

if sys.platform == "darwin":
    # Use modern libc++; do NOT pin to ancient macOS
    extra_compile_args += ["-stdlib=libc++"]


cmdclass = {}

ext_modules = [
    Extension(
        "pmmoto.core._voxels",
        ["src/pmmoto/core/_voxels.pyx"],
        include_dirs=["src/pmmoto/core"],
        language="c++",
        extra_compile_args=extra_compile_args,
    ),
    Extension(
        "pmmoto.filters.distance._distance",
        ["src/pmmoto/filters/distance/_distance.pyx"],
        include_dirs=["src/pmmoto/filters/distance", numpy.get_include()],
        language="c++",
        extra_compile_args=extra_compile_args,
    ),
    Extension(
        "pmmoto.analysis._minkowski",
        [
            "src/pmmoto/analysis/_minkowski.pyx",
            "src/pmmoto/analysis/quantimpyc.c",
            "src/pmmoto/analysis/minkowskic.c",
        ],
        include_dirs=["src/pmmoto/analysis"],
    ),
    Extension(
        "pmmoto.io._data_read",
        ["src/pmmoto/io/_data_read.pyx"],
        include_dirs=["src/pmmoto/io"],
        libraries=["z"],
        language="c++",
        extra_compile_args=extra_compile_args,
    ),
    Extension(
        "pmmoto.particles._particles",
        ["src/pmmoto/particles/_particles.pyx"],
        include_dirs=["src/pmmoto/particles/"],
        language="c++",
        extra_compile_args=extra_compile_args,
    ),
    Extension(
        "pmmoto.domain_generation._rdf",
        ["src/pmmoto/domain_generation/_rdf.pyx"],
        include_dirs=["src/pmmoto/domain_generation/", "src/pmmoto/particles/"],
        language="c++",
        extra_compile_args=extra_compile_args,
    ),
    Extension(
        "pmmoto.analysis._bins",
        ["src/pmmoto/analysis/_bins.pyx"],
        include_dirs=["src/pmmoto/analysis/"],
        language="c++",
        extra_compile_args=extra_compile_args,
    ),
    Extension(
        "pmmoto.domain_generation._domain_generation",
        ["src/pmmoto/domain_generation/_domain_generation.pyx"],
        include_dirs=["src/pmmoto/domain_generation/", "src/pmmoto/particles/"],
        language="c++",
        extra_compile_args=extra_compile_args,
    ),
]
setup(
    ext_modules=cythonize(
        ext_modules,
        annotate=True,
        compiler_directives={"language_level": "3"},
    ),
    include_dirs=numpy.get_include(),
)
