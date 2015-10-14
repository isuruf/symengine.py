from __future__ import print_function
from os import getenv, path
import subprocess
import sys

# use setuptools by default as per the official advice at:
# packaging.python.org/en/latest/current.html#packaging-tool-recommendations
use_setuptools = True
# set the environment variable USE_DISTUTILS=True to force the use of distutils
use_distutils = getenv('USE_DISTUTILS')
if use_distutils is not None:
    if use_distutils.lower() == 'true':
        use_setuptools = False
    else:
        print("Value {} for USE_DISTUTILS treated as False".\
              format(use_distutils))

from distutils.command.build import build as _build

if use_setuptools:
    try:
        from setuptools import setup
        from setuptools.command.install import install as _install
    except ImportError:
        use_setuptools = False

if not use_setuptools:
    from distutils.core import setup
    from distutils.command.install import install as _install

cmake_opts = [("PYTHON_BIN", sys.executable)]

def process_opts(opts):
    return ['-D'+'='.join(o) for o in opts]

def cmake_build():
    cmake_cmd = ["cmake", path.dirname(path.realpath(__file__))]
    cmake_cmd.extend(process_opts(cmake_opts))
    if subprocess.call(cmake_cmd) != 0:
        raise EnvironmentError("error calling cmake")

    if subprocess.call("make") != 0:
        raise EnvironmentError("error calling make")

class BuildWithCmake(_build):
    _build_opts = _build.user_options
    user_options = [
        ('symengine-dir=', None, 'path to symengine installation or build directory'),
        ('define=', 'D',
         'options to cmake <var>:<type>=<value>')
    ]
    user_options.extend(_build_opts)

    def initialize_options(self):
        _build.initialize_options(self)
        self.define = None
        self.symengine_dir = None

    def finalize_options(self):
        _build.finalize_options(self)
        # The argument parsing will result in self.define being a string, but
        # it has to be a list of 2-tuples.
        # Multiple symbols can be separated with semi-colons.
        if self.define:
            defines = self.define.split(';')
            self.define = [(s.strip(), None) if '=' not in s else
                           tuple(ss.strip() for ss in s.split('='))
                           for s in defines]
            cmake_opts.extend(self.define)
        if self.symengine_dir:
            cmake_opts.extend(['SymEngine_DIR', self.symengine_dir])

    def run(self):
        cmake_build()
        # can't use super() here because _build is an old style class in 2.7
        _build.run(self)

class InstallWithCmake(_install):
    _install_opts = _install.user_options
    user_options = [
        ('symengine-dir=', None, 'path to symengine installation or build directory'),
        ('define=', 'D',
         'options to cmake <var>:<type>=<value>')
    ]
    user_options.extend(_install_opts)

    def initialize_options(self):
        _install.initialize_options(self)
        self.define = None
        self.symengine_dir = None

    def finalize_options(self):
        _install.finalize_options(self)
        # The argument parsing will result in self.define being a string, but
        # it has to be a list of 2-tuples.
        # Multiple symbols can be separated with semi-colons.
        if self.define:
            defines = self.define.split(';')
            self.define = [(s.strip(), None) if '=' not in s else
                           tuple(ss.strip() for ss in s.split('='))
                           for s in defines]
            cmake_opts.extend(self.define)
        if self.symengine_dir:
            cmake_opts.extend([('SymEngine_DIR', self.symengine_dir)])

    def run(self):
        # can't use super() here because _install is an old style class in 2.7
        _install.run(self)

long_description = '''
SymEngine is a standalone fast C++ symbolic manipulation library.
Optional thin Python wrappers (SymEngine) allow easy usage from Python and
integration with SymPy.'''

setup(name = "symengine",
      version = "0.1.0.dev",
      description = "Python library providing wrappers to SymEngine",
      long_description = "",
      author = "",
      author_email = "",
      license = "MIT",
      url = "https://github.com/sympy/symengine",
      packages=['symengine', 'symengine.lib', 'symengine.tests'],
      package_data= {'symengine' : ['lib/symengine_wrapper.so']},
      cmdclass={
          'build' : BuildWithCmake,
          'install' : InstallWithCmake,
          }
  )
