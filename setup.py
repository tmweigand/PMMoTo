from setuptools import Extension, setup
from Cython.Build import cythonize
import Cython.Compiler.Options
import numpy
Cython.Compiler.Options.annotate = True


cmdclass = {}
ext_modules = [
Extension("pmmoto.core.*",
          ["src/pmmoto/core/*.pyx"],
           include_dirs=['src/pmmoto/core'],language='c++'),
Extension("pmmoto.analysis._minkowski",
              ["src/pmmoto/analysis/_minkowski.pyx", "src/pmmoto/analysis/quantimpyc.c", "src/pmmoto/analysis/minkowskic.c"],
              include_dirs=['pmmoto/analysis']),
Extension("pmmoto.domain_generation.*",
          ["src/pmmoto/domain_generation/*.pyx"],
           include_dirs=['src/pmmoto/domain_generation'],language='c++'),
]
setup(
    name="pmmoto",
    ext_modules=cythonize(
        ext_modules, annotate=True, compiler_directives={"language_level": "3"}
    ),
    include_dirs=numpy.get_include(),
)
