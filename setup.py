"""Setup script for PMMoTo.

Builds Cython/C++ extensions and installs the PMMoTo package.
"""

import sys
from setuptools import Extension, setup
from Cython.Build import cythonize
import Cython.Compiler.Options
import numpy

Cython.Compiler.Options.annotate = True

if sys.platform == "win32":
    sys.exit("Windows is not supported for building this package.")

# common fast flags for Unix / macOS (no language-specific flags)
base_compile_args = [
    "-O3",
    "-ffast-math",
    "-funroll-loops",
    "-fomit-frame-pointer",
    "-fno-math-errno",
    "-flto",
    "-pthread",
]

# C++-specific flags (only add to extensions using C++)
cpp_compile_args = ["-std=c++17"] + base_compile_args[:]

extra_link_args = ["-flto"]

if sys.platform.startswith("linux"):
    extra_link_args += ["-lm", "-lmvec"]

if sys.platform == "darwin":
    base_compile_args += ["-stdlib=libc++", "-mmacosx-version-min=10.9"]
    cpp_compile_args += ["-stdlib=libc++", "-mmacosx-version-min=10.9"]
    extra_link_args += ["-stdlib=libc++", "-mmacosx-version-min=10.9"]

ext_modules = [
    Extension(
        "pmmoto.core._voxels",
        ["src/pmmoto/core/_voxels.pyx"],
        include_dirs=["src/pmmoto/core"],
        language="c++",
        extra_compile_args=cpp_compile_args,
        extra_link_args=extra_link_args,
    ),
    Extension(
        "pmmoto.filters.distance._distance",
        ["src/pmmoto/filters/distance/_distance.pyx"],
        include_dirs=["src/pmmoto/filters/distance", numpy.get_include()],
        language="c++",
        extra_compile_args=cpp_compile_args,
        extra_link_args=extra_link_args,
    ),
    Extension(
        "pmmoto.analysis._minkowski",
        [
            "src/pmmoto/analysis/_minkowski.pyx",
            "src/pmmoto/analysis/quantimpyc.c",
            "src/pmmoto/analysis/minkowskic.c",
        ],
        include_dirs=["src/pmmoto/analysis"],
        language="c",  # keep C for the .c sources
        extra_compile_args=base_compile_args,
        extra_link_args=extra_link_args,
    ),
    Extension(
        "pmmoto.io._data_read",
        ["src/pmmoto/io/_data_read.pyx"],
        include_dirs=["src/pmmoto/io"],
        libraries=["z"],
        language="c++",
        extra_compile_args=cpp_compile_args,
        extra_link_args=extra_link_args,
    ),
    Extension(
        "pmmoto.particles._particles",
        ["src/pmmoto/particles/_particles.pyx"],
        include_dirs=["src/pmmoto/particles/"],
        language="c++",
        extra_compile_args=cpp_compile_args,
        extra_link_args=extra_link_args,
    ),
    Extension(
        "pmmoto.domain_generation._rdf",
        ["src/pmmoto/domain_generation/_rdf.pyx"],
        include_dirs=["src/pmmoto/domain_generation/", "src/pmmoto/particles/"],
        language="c++",
        extra_compile_args=cpp_compile_args,
        extra_link_args=extra_link_args,
    ),
    Extension(
        "pmmoto.analysis._bins",
        ["src/pmmoto/analysis/_bins.pyx"],
        include_dirs=["src/pmmoto/analysis/"],
        language="c++",
        extra_compile_args=cpp_compile_args,
        extra_link_args=extra_link_args,
    ),
    Extension(
        "pmmoto.domain_generation._domain_generation",
        ["src/pmmoto/domain_generation/_domain_generation.pyx"],
        include_dirs=["src/pmmoto/domain_generation/", "src/pmmoto/particles/"],
        language="c++",
        extra_compile_args=cpp_compile_args,
        extra_link_args=extra_link_args,
    ),
]

setup(
    name="pmmoto",
    ext_modules=cythonize(
        ext_modules,
        annotate=True,
        compiler_directives={"language_level": "3"},
    ),
    include_dirs=numpy.get_include(),
)
