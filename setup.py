import os
import sys
from packaging import version
from setuptools import setup
from multiprocessing import cpu_count
import numpy
from distutils.core import setup,Extension
from Cython.Distutils import build_ext
from Cython.Build import cythonize
import Cython.Compiler.Options
Cython.Compiler.Options.annotate = True

CYTHON_VERSION = '0.23.4'

def cython(pyx_files, working_path=''):
    """Use Cython to convert the given files to C.
    Parameters
    ----------
    pyx_files : list of str
        The input .pyx files.
    """
    # Do not build cython files if target is clean
    if len(sys.argv) >= 2 and sys.argv[1] == 'clean':
        return

    try:
        from Cython import __version__
        if version.parse(__version__) < version.parse(CYTHON_VERSION):
            raise RuntimeError(f'Cython >= {CYTHON_VERSION} needed to build scikit-image')

        from Cython.Build import cythonize
    except ImportError:
        # If cython is not found, the build will make use of
        # the distributed .c or .cpp files if present
        c_files_used = [_compiled_filename(os.path.join(working_path, f))
                        for f in pyx_files]

        print(f"Cython >= {CYTHON_VERSION} not found; "
              f"falling back to pre-built {' '.join(c_files_used)}")
    else:
        pyx_files = [os.path.join(working_path, f) for f in pyx_files]
        for i, pyxfile in enumerate(pyx_files):
            if pyxfile.endswith('.pyx.in'):
                process_tempita_pyx(pyxfile)
                pyx_files[i] = pyxfile.replace('.pyx.in', '.pyx')

        # skip cythonize when creating an sdist
        # (we do not want the large cython-generated sources to be included)
        if 'sdist' not in sys.argv:
            # Cython doesn't automatically choose a number of threads > 1
            # https://github.com/cython/cython/blob/a0bbb940c847dfe92cac446c8784c34c28c92836/Cython/Build/Dependencies.py#L923-L925
            cythonize(pyx_files, nthreads=cpu_count(),
                      compiler_directives={'language_level': 3})


def process_tempita_pyx(fromfile):
    try:
        try:
            from Cython import Tempita as tempita
        except ImportError:
            import tempita
    except ImportError:
        raise Exception('Building requires Tempita: '
                        'pip install --user Tempita')
    template = tempita.Template.from_filename(fromfile,
                                              encoding=sys.getdefaultencoding())
    pyxcontent = template.substitute()
    if not fromfile.endswith('.pyx.in'):
        raise ValueError(f"Unexpected extension of {fromfile}.")

    pyxfile = os.path.splitext(fromfile)[0]    # split off the .in ending
    with open(pyxfile, "w") as f:
        f.write(pyxcontent)


medialFunctions = 'pmmoto/filters/medialAxis/medialExtractionFunctions.pyx.in'
medialPYX = os.path.splitext(medialFunctions)[0]
if os.path.isfile(medialPYX):
    if os.path.getmtime(medialPYX) < os.path.getmtime(medialFunctions):
        cython([medialFunctions])
else:
    cython([medialFunctions])

with open("README.md", 'r') as f:
    long_description = f.read()

cmdclass = {}
ext_modules = []

ext_modules += [
    Extension("pmmoto.domain_generation._domainGeneration",
              ["pmmoto/domain_generation/_domainGeneration.pyx"],
              include_dirs=['pmmoto/domain_generation/'],language='c++'),
    Extension("pmmoto.core.nodes",
              ["pmmoto/core/nodes.pyx"],
              include_dirs=['pmmoto/core']),
    Extension("pmmoto.core._nodes",
              ["pmmoto/core/_nodes.pyx"],
              include_dirs=['pmmoto/core'],language='c++'),
    Extension("pmmoto.core._Orientation",
              ["pmmoto/core/_Orientation.pyx"],
              include_dirs=['pmmoto/core'],language='c++'),
    Extension("pmmoto.core._set",
              ["pmmoto/core/_set.pyx"],
              include_dirs=['pmmoto/core'],language='c++'),
    Extension("pmmoto.core._sets",
              ["pmmoto/core/_sets.pyx"],
              include_dirs=['pmmoto/core'],language='c++'),
    Extension("pmmoto.analysis._minkowski",
              ["pmmoto/analysis/_minkowski.pyx", "pmmoto/analysis/quantimpyc.c", "pmmoto/analysis/minkowskic.c"],
              include_dirs=['pmmoto/analysis']),
    Extension("pmmoto.filters.medialAxis.medialExtraction",
              ["pmmoto/filters/medialAxis/medialExtraction.pyx"],
              include_dirs=['pmmoto','pmmoto/filters/medialAxis'],language='c++'),
    Extension("pmmoto.filters.medialAxis.medialNodes",
              ["pmmoto/filters/medialAxis/medialNodes.pyx"],
              include_dirs=['pmmoto','pmmoto/filters/medialAxis']),
    Extension("pmmoto.filters.medialAxis.medialSet",
              ["pmmoto/filters/medialAxis/medialSet.pyx"],
              include_dirs=['pmmoto','pmmoto/filters/medialAxis']),
    Extension("pmmoto.filters.medialAxis.medialSets",
              ["pmmoto/filters/medialAxis/medialSets.pyx"],
              include_dirs=['pmmoto','pmmoto/filters/medialAxis'],language='c++'),
    Extension("pmmoto.filters.medialAxis.medialPath",
              ["pmmoto/filters/medialAxis/medialPath.pyx"],
              include_dirs=['pmmoto','pmmoto/filters/medialAxis']),
    Extension("pmmoto.filters.medialAxis.medialExtractionFunctions", 
              ["pmmoto/filters/medialAxis/medialExtractionFunctions.pyx"],
              include_dirs=['pmmoto','pmmoto/filters/medialAxis'],language='c++'),
]
cmdclass.update({'build_ext': build_ext})


setup(
    name="pmmoto",
    version="0.0.1",
    packages=['pmmoto','pmmoto.filters.medialAxis','pmmoto.filters.multiPhase'],
    author="Timothy M Weigand",
    description="Porous Media Morphological and Topological Analysis Toolkit",
    long_description=long_description,
    long_description_content_type='text/markdown',
    url="https://github.com/tmweigand/pmmoto",
    cmdclass=cmdclass,
    ext_modules=cythonize(ext_modules,annotate=True,compiler_directives={'language_level' : "3"}),
    include_dirs=numpy.get_include(),
    install_requires=[
        'numpy>=1.22.3',
        'mpi4py>=3.1.3',
        'packaging>=21.3',
        'pykdtree>=0.1',
        'edt>=2.3.1',
        'scipy>=1.9.3',
        'pyevtk>=1.5.0',
        'pyvista>=0.39.1',
        'connected_components_3d>=3.12.4'
    ]
)
