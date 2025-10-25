"""Setup script for PMMoTo.

Builds Cython/C++ extensions and installs the PMMoTo package.
"""

import sys
import platform
import os
import subprocess
import tempfile
from setuptools import Extension, setup
from Cython.Build import cythonize
import Cython.Compiler.Options
import numpy

Cython.Compiler.Options.annotate = True

if sys.platform == "win32":
    sys.exit("Windows is not supported for building this package.")


def _flag_supported(flag: str, lang: str = "c++") -> bool:
    compiler = os.environ.get("CXX", "clang++" if sys.platform == "darwin" else "c++")
    code = "int main(){return 0;}\n"
    with tempfile.TemporaryDirectory() as td:
        src = os.path.join(td, "t.cc")
        obj = os.path.join(td, "t.o")
        with open(src, "w") as f:
            f.write(code)
        try:
            cmd = [compiler, "-Werror", "-x", lang, src, "-c", "-o", obj, flag]
            return (
                subprocess.run(
                    cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE
                ).returncode
                == 0
            )
        except Exception:
            return False


def _try_add(flags: list, flag: str, lang: str = "c++"):
    if _flag_supported(flag, lang):
        flags.append(flag)


# common fast flags
base_compile_args = [
    "-O3",
    "-ffast-math",
    "-funroll-loops",
    "-fomit-frame-pointer",
    "-fno-math-errno",
    "-flto",
    "-pthread",
]
cpp_compile_args = ["-std=c++17"] + base_compile_args[:]
extra_link_args = ["-flto"]

if sys.platform.startswith("linux"):
    extra_link_args += ["-lm", "-lmvec"]
    _try_add(cpp_compile_args, "-march=native")
    _try_add(cpp_compile_args, "-mtune=native")

if sys.platform == "darwin":
    machine = platform.machine()

    # Always add standard macOS flags
    for f in ["-stdlib=libc++", "-mmacosx-version-min=10.9"]:
        base_compile_args.append(f)
        cpp_compile_args.append(f)
        extra_link_args.append(f)

    if machine == "arm64":
        # Prefer specific Apple CPUs; fall back to armv8*
        for f in [
            "-mcpu=apple-m3",
            "-mcpu=apple-m2",
            "-mcpu=apple-m1",
            "-march=armv8.5-a",
            "-march=armv8.4-a",
            "-march=armv8-a",
        ]:
            if _flag_supported(f):
                cpp_compile_args.append(f)
                break
        _try_add(cpp_compile_args, "-mtune=native")
    elif machine == "x86_64":
        # Only enable native on request; probe support
        if os.environ.get("PMMOTO_NATIVE") == "1":
            _try_add(cpp_compile_args, "-march=native")
            _try_add(cpp_compile_args, "-mtune=native")

# _data_read specific flags
dr_compile_args = cpp_compile_args[:] + [
    "-fno-rtti",
    "-ftree-vectorize",
    "-funroll-loops",
]

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
        language="c",
        extra_compile_args=base_compile_args,
        extra_link_args=extra_link_args,
    ),
    Extension(
        "pmmoto.io._data_read",
        ["src/pmmoto/io/_data_read.pyx"],
        include_dirs=["src/pmmoto/io", "extern/fast_float/include"],
        libraries=["z"],
        language="c++",
        extra_compile_args=dr_compile_args,
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
