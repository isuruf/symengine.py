from __future__ import print_function
from os import getenv, path
import subprocess
import sys
from Cython.Build import cythonize


# use setuptools by default as per the official advice at:
# packaging.python.org/en/latest/current.html#packaging-tool-recommendations
use_setuptools = True
# set the environment variable USE_DISTUTILS=True to force the use of distutils
use_distutils = getenv('USE_DISTUTILS')
if use_distutils is not None:
    if use_distutils.lower() == 'true':
        use_setuptools = False
    else:
        print("Value {} for USE_DISTUTILS treated as False".
              format(use_distutils))

from distutils.command.build import build as _build

if use_setuptools:
    try:
        from setuptools import setup
        from setuptools.command.install import install as _install
        from setuptools import Extension
    except ImportError:
        use_setuptools = False

if not use_setuptools:
    from distutils.core import setup
    from distutils.core import Extension
    from distutils.command.install import install as _install

extensions = []
symengine_dirs = ["/usr/local/"]

def get_flags(symengine_dir):
    symengine_dir.reverse()
    symengine_dir = ':'.join([path.join(dir, 'lib/cmake/symengine') for dir in symengine_dir])
    compile_flags = subprocess.check_output(
        ['cmake', '--find-package', '-DNAME=SymEngine', '-DCOMPILER_ID=GNU', '-DLANGUAGE=CXX',
         '-DMODE=COMPILE', '-DSymEngine_DIR=' + symengine_dir])
    link_flags = subprocess.check_output(
        ['cmake', '--find-package', '-DNAME=SymEngine', '-DCOMPILER_ID=GNU', '-DLANGUAGE=CXX',
         '-DMODE=LINK', '-DSymEngine_DIR=' + symengine_dir])
    return compile_flags.strip().split(), link_flags.strip().split()

def build():
    compile_flags, link_flags = get_flags(symengine_dirs)
    extensions.extend(
        cythonize(
            Extension('symengine.lib.symengine_wrapper',
                      sources=['symengine/lib/symengine_wrapper.pyx'],
                      extra_compile_args=['-std=c++0x'] + compile_flags,
                      extra_link_args=link_flags,
                      language="c++"
                      ),
            compile_time_env={"HAVE_SYMENGINE_MPFR": True, "HAVE_SYMENGINE_MPC": True}
        )
    )

class BuildWithCmake(_build):
    _build_opts = _build.user_options
    user_options = [
        ('symengine-dir=', None, 'If CMake is installed, path to symengine installation or build directory'),
    ]
    user_options.extend(_build_opts)

    def initialize_options(self):
        _build.initialize_options(self)
        self.symengine_dir = None

    def finalize_options(self):
        _build.finalize_options(self)

    def run(self):
        if self.symengine_dir:
            symengine_dirs.extend(self.symengine_dir)
        build()
        # can't use super() here because _build is an old style class in 2.7
        _build.run(self)


class InstallWithCmake(_install):
    _install_opts = _install.user_options
    user_options = [
        ('symengine-dir=', None, 'If CMake is installed, path to symengine installation or build directory'),
    ]
    user_options.extend(_install_opts)

    def initialize_options(self):
        _install.initialize_options(self)
        self.symengine_dir = None

    def finalize_options(self):
        _install.finalize_options(self)

    def run(self):
        if self.symengine_dir:
            symengine_dirs.extend(self.symengine_dir)
        # can't use super() here because _install is an old style class in 2.7
        _install.run(self)


long_description = '''
SymEngine is a standalone fast C++ symbolic manipulation library.
Optional thin Python wrappers (SymEngine) allow easy usage from Python and
integration with SymPy.'''

setup(name="symengine",
      version="0.1.0.dev",
      description="Python library providing wrappers to SymEngine",
      long_description=long_description,
      author="",
      author_email="",
      license="MIT",
      url="https://github.com/sympy/symengine",
      packages=['symengine', 'symengine.lib', 'symengine.tests'],
      ext_modules=extensions,
      package_data={'symengine': ['lib/symengine_wrapper.so']},
      cmdclass={
          'build': BuildWithCmake,
          'install': InstallWithCmake
      },
      zip_safe=False
      )
