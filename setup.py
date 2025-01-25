from setuptools import Extension, setup
from Cython.Build import cythonize
import Cython.Compiler.Options
import numpy

Cython.Compiler.Options.annotate = True

import subprocess
import sys

# try:
#     subprocess.run(["mpirun", "--version"], check=True, stdout=subprocess.PIPE)
# except FileNotFoundError:
#     sys.stderr.write("Error: MPI is not installed. Please install OpenMPI or MPICH.\n")
#     sys.exit(1)


extra_compile_args = ["-std=c++17"]
if sys.platform == "win32":
    extra_compile_args += ["/std:c++17", "/O2"]
else:
    extra_compile_args += ["-std=c++17", "-O3", "-ffast-math", "-pthread"]

if sys.platform == "darwin":
    extra_compile_args += ["-stdlib=libc++", "-mmacosx-version-min=10.9"]


cmdclass = {}

ext_modules = [
    # Extension(
    #     "pmmoto.core._set",
    #     ["src/pmmoto/core/_set.pyx"],
    #     include_dirs=["src/pmmoto/core"],
    #     language="c++",
    # ),
    # Extension(
    #     "pmmoto.core._sets",
    #     ["src/pmmoto/core/_sets.pyx"],
    #     include_dirs=["src/pmmoto/core"],
    #     language="c++",
    # ),
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
    # Extension(
    #     "pmmoto.analysis._minkowski",
    #     [
    #         "src/pmmoto/analysis/_minkowski.pyx",
    #         "src/pmmoto/analysis/quantimpyc.c",
    #         "src/pmmoto/analysis/minkowskic.c",
    #     ],
    #     include_dirs=["pmmoto/analysis"],
    # ),
    Extension(
        "pmmoto.domain_generation._domain_generation",
        ["src/pmmoto/domain_generation/_domain_generation.pyx"],
        include_dirs=["src/pmmoto/domain_generation"],
        language="c++",
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
