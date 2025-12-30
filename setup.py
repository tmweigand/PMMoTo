"""Setup script for PMMoTo.

Builds Cython/C++ extensions and installs the PMMoTo package.
"""

import sys
import platform
import os
import subprocess
import tempfile
import shutil
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


def _pkg_config(name: str):
    try:
        if (
            shutil.which("pkg-config")
            and subprocess.run(["pkg-config", "--exists", name]).returncode == 0
        ):
            cflags = (
                subprocess.check_output(["pkg-config", "--cflags", name])
                .decode()
                .strip()
                .split()
            )
            libs = (
                subprocess.check_output(["pkg-config", "--libs", name])
                .decode()
                .strip()
                .split()
            )
            return cflags, libs
    except Exception:
        pass
    return [], []


def find_libdeflate():
    cflags, libs = _pkg_config("libdeflate")
    if libs:
        return cflags, libs, True
    # Homebrew fallback
    hb = "/opt/homebrew" if os.path.exists("/opt/homebrew") else "/usr/local"
    inc = os.path.join(hb, "opt", "libdeflate", "include")
    lib = os.path.join(hb, "opt", "libdeflate", "lib")
    if os.path.isdir(inc) and os.path.isdir(lib):
        return [f"-I{inc}"], [f"-L{lib}", "-ldeflate", f"-Wl,-rpath,{lib}"], True
    return [], [], False


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
    extra_link_args += ["-lm"]
    if _link_flag_supported("-lmvec"):
        extra_link_args += ["-lmvec"]
    extra_link_args += ["-ldl"]  # for dladdr/Dl_info
    if os.environ.get("PMMOTO_NATIVE") == "1":
        _try_add(cpp_compile_args, "-march=native")
        _try_add(cpp_compile_args, "-mtune=native")

if sys.platform == "darwin":
    machine = platform.machine()

    # Standard macOS flags (portable)
    for f in ["-stdlib=libc++", "-mmacosx-version-min=10.9"]:
        base_compile_args.append(f)
        cpp_compile_args.append(f)
        extra_link_args.append(f)

    # Only add CPU-specific flags when explicitly requested AND supported
    if os.environ.get("PMMOTO_NATIVE") == "1":
        if machine == "arm64":
            for f in [
                "-mcpu=native",
                "-march=armv9-a",
                "-march=armv8.5-a",
                "-march=armv8.4-a",
                "-march=armv8-a",
            ]:
                if _flag_supported(f):
                    cpp_compile_args.append(f)
                    break
        elif machine == "x86_64":
            _try_add(cpp_compile_args, "-march=native")
            _try_add(cpp_compile_args, "-mtune=native")

# _data_read specific flags
dr_compile_args = cpp_compile_args[:] + [
    "-fno-rtti",
    "-ftree-vectorize",
    "-funroll-loops",
]

# Prefer libdeflate; fallback to zlib
ld_cflags, ld_libs, have_ld = find_libdeflate()
if have_ld:
    dr_compile_args = dr_compile_args + ["-DHAVE_LIBDEFLATE=1"]

# Optional debug
if os.environ.get("PMMOTO_SHOW_FLAGS") == "1":
    print("cpp_compile_args:", cpp_compile_args)
    print("dr_compile_args:", dr_compile_args)
    print("extra_link_args:", extra_link_args)
    print("libdeflate:", have_ld, ld_cflags, ld_libs)

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
        language="c++",
        extra_compile_args=dr_compile_args + ld_cflags,
        extra_link_args=extra_link_args + (ld_libs if have_ld else ["-lz"]),
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
