#!/usr/bin/env python3

from setuptools import setup
import distutils.command.build
import distutils.command.clean
import shutil
import sys
import os
import subprocess

# Ensure supported version of python is used
if sys.version_info < (3, 4):
    sys.exit('ERROR: Python < 3.4 is not supported!')


###############################
# Specify defaults
###############################
DEFAULT_COMPUTE_CAPABILITIES = ["3.5", "5.2", "6.1"]


###############################
# Specify sources
###############################
SOURCES_CUDA = ['gather_columns_functor_gpu.cu.cc',
                'scatter_columns_functor_gpu.cu.cc',
                'scatter_values_functor_gpu.cu.cc']
HEADERS_CUDA = ['gather_columns_functor_gpu.cu.h',
                'scatter_columns_functor_gpu.cu.h',
                'scatter_values_functor_gpu.cu.h']
SOURCES = ['gather_columns.cc',
           'gather_columns_functor.cc',
           'scatter_columns.cc',
           'scatter_columns_functor.cc',
           'scatter_values.cc',
           'scatter_values_functor.cc']
HEADERS = ['gather_columns_functor.h',
           'scatter_columns_functor.h',
           'scatter_values_functor.h']


###############################
# Compute paths
###############################
BUILD_DIR = os.path.abspath(os.path.join('build', 'ops'))
SOURCE_DIR = os.path.abspath(os.path.join('libspn', 'ops'))
TARGET = os.path.join(SOURCE_DIR, "libspn_ops.so")
OBJECTS_CUDA = [os.path.join(BUILD_DIR, os.path.basename(s) + ".o")
                for s in SOURCES_CUDA]
SOURCES_CUDA = [os.path.join(SOURCE_DIR, i) for i in SOURCES_CUDA]
HEADERS_CUDA = [os.path.join(SOURCE_DIR, i) for i in HEADERS_CUDA]
SOURCES = [os.path.join(SOURCE_DIR, i) for i in SOURCES]
HEADERS = [os.path.join(SOURCE_DIR, i) for i in HEADERS]


###############################
# Build command
###############################
class BuildCommand(distutils.command.build.build):
    """Custom build command compiling the C++ code."""

    user_options = [
        ('exec-time', None, 'Compile in execution time measurement code'),
        ('compute-capabilities=', None,
         'List of coma separated compute capabilities to use (default: %s)'
         % (','.join(DEFAULT_COMPUTE_CAPABILITIES)))
    ]

    def initialize_options(self):
        super().initialize_options()
        self.exec_time = None
        self.compute_capabilities = (','.join(DEFAULT_COMPUTE_CAPABILITIES))

    def finalize_options(self):
        super().finalize_options()
        self.compute_capabilities = self.compute_capabilities.split(',')
        try:
            self.compute_capabilities = [c[0] + c[2]
                                         for c in self.compute_capabilities]
        except IndexError:
            sys.exit("ERROR: Incorrect compute capabilities.")

    def _configure(self):
        # Options
        print(self._col_head + "Options:" + self._col_clear)
        print("- Compute capabilities: %s" % ','.join(self.compute_capabilities))
        print("- Debug: %s" % ("NO" if self.debug is None else "YES"))
        print("- Exec time: %s" % ("NO" if self.exec_time is None else "YES"))
        # Detect
        print(self._col_head + "Configuring:" + self._col_clear)
        # CUDA
        # - try finding nvcc in PATH
        self._cuda_nvcc = shutil.which('nvcc')
        if self._cuda_nvcc is not None:
            self._cuda_home = os.path.dirname(os.path.dirname(self._cuda_nvcc))
        else:
            # - try a set of paths
            cuda_paths = ['/usr/local/cuda', '/usr/local/cuda-8.0']
            for p in cuda_paths:
                pb = os.path.join(p, 'bin', 'nvcc')
                if os.path.exists(pb):
                    self._cuda_home = p
                    self._cuda_nvcc = pb
        if not self._cuda_home:
            os.sys.exit("ERROR: CUDA not found!")
        self._cuda_libs = os.path.join(self._cuda_home, 'lib64')

        print("- Found CUDA in %s" % self._cuda_home)
        print("  nvcc: %s" % self._cuda_nvcc)
        print("  libraries: %s" % self._cuda_libs)

        # TensorFlow
        import tensorflow
        self._tf_includes = tensorflow.sysconfig.get_include()
        self._tf_version = tensorflow.__version__
        self._tf_gcc_version = tensorflow.__compiler_version__
        self._tf_gcc_version_major = int(self._tf_gcc_version[0])
        print("- Found TensorFlow %s" % self._tf_version)
        print("  gcc version: %s" % self._tf_gcc_version)
        print("  includes: %s" % self._tf_includes)

        # gcc
        try:
            cmd = ['gcc', '-dumpversion']
            # Used instead of run for 3.4 compatibility
            self._gcc_version = subprocess.check_output(cmd).decode('ascii').strip()
            self._gcc_version_major = int(self._gcc_version[0])
        except subprocess.CalledProcessError:
            os.sys.exit('ERROR: gcc not found!')
        print("- Found gcc %s" % self._gcc_version)
        self._downgrade_abi = (self._gcc_version_major > self._tf_gcc_version_major)
        if self._downgrade_abi:
            print("  TF gcc version < system gcc version: "
                  "using -D_GLIBCXX_USE_CXX11_ABI=0")

    def _run_nvcc(self, obj, source):
        try:
            cmd = ([self._cuda_nvcc, "-c", "-o",
                    obj, source,
                    '-std=c++11', '-x=cu', '-Xcompiler', '-fPIC',
                    '-DGOOGLE_CUDA=1',
                    '--expt-relaxed-constexpr',  # To silence harmless warnings
                    '-I', self._tf_includes] +
                   # Downgrade the ABI if system gcc > TF gcc
                   (['-D_GLIBCXX_USE_CXX11_ABI=0']
                    if self._downgrade_abi else []) +
                   # --exec-time build option
                   (['-DEXEC_TIME_CALC=1']
                    if self.exec_time is not None else []) +
                   # Compute capabilities
                   [('-gencode=arch=compute_%s,\"code=sm_%s,compute_%s\"' %
                     (c, c, c)) for c in self.compute_capabilities])
            print(self._col_cmd + ' '.join(cmd) + self._col_clear)
            subprocess.check_call(cmd)  # Used instead of run for 3.4 compatibility
        except subprocess.CalledProcessError:
            os.sys.exit('ERROR: Build error!')

    def _run_gcc(self, target, inputs):
        try:
            cmd = (['g++', '-shared', '-o', target] +
                   inputs +
                   ['-std=c++11', '-fPIC', '-lcudart',
                    '-DGOOGLE_CUDA=1',
                    '-O2',  # Used in other TF code and sufficient for max opt
                    '-I', self._tf_includes,
                    '-L', self._cuda_libs] +
                   # Downgrade the ABI if system gcc > TF gcc
                   (['-D_GLIBCXX_USE_CXX11_ABI=0']
                    if self._downgrade_abi else []) +
                   # --exec-time build option
                   (['-DEXEC_TIME_CALC=1']
                    if self.exec_time is not None else []))
            print(self._col_cmd + ' '.join(cmd) + self._col_clear)
            subprocess.check_call(cmd)  # Used instead of run for 3.4 compatibility
        except subprocess.CalledProcessError:
            os.sys.exit('ERROR: Build error!')

    def _is_dirty(self, target, sources):
        """Verify if changes have been made to sources and target must
        be re-built."""
        t_date = os.path.getmtime(target) if os.path.exists(target) else 0
        s_date = max(os.path.getmtime(s) for s in sources)
        return t_date <= s_date

    def _build(self):
        print(self._col_head + "Building:" + self._col_clear)
        # Make dirs
        os.makedirs(BUILD_DIR, exist_ok=True)
        # Should rebuild?
        if self._is_dirty(TARGET, SOURCES + HEADERS +
                          SOURCES_CUDA + HEADERS_CUDA):
            # Compile cuda
            for s, h, o in zip(SOURCES_CUDA, HEADERS_CUDA, OBJECTS_CUDA):
                if self._is_dirty(o, [s, h]):
                    self._run_nvcc(o, s)
            # Compile rest and link
            self._run_gcc(TARGET, OBJECTS_CUDA + SOURCES)
        else:
            print("Everything up to date.")

    def _test(self):
        print(self._col_head + "Testing:" + self._col_clear)
        import libspn.ops.ops
        libspn.ops.gather_cols
        libspn.ops.scatter_cols
        libspn.ops.scatter_values
        print("Custom ops loaded correctly!")

    def run(self):
        # Original run
        super().run()

        # Initialize color - here to ensure colorama is installed
        import colorama
        colorama.init()
        self._col_head = colorama.Style.BRIGHT + colorama.Fore.YELLOW
        self._col_cmd = colorama.Fore.BLUE
        self._col_clear = colorama.Style.RESET_ALL

        # Build
        print(self._col_head +
              "====================== BUILDING OPS ======================" +
              self._col_clear)
        self._configure()
        self._build()
        self._test()
        print(self._col_head +
              "========================== DONE ==========================" +
              self._col_clear)


class CleanCommand(distutils.command.clean.clean):
    """Custom clean command including the C++ code."""

    def _clean(self, files):
        print(self._col_head + "Cleaning:" + self._col_clear)
        for f in files:
            if os.path.exists(f):
                print("Removing " + f)
                os.remove(f)

    def run(self):
        # Original run
        super().run()

        # Initialize color - here to ensure colorama is installed
        import colorama
        colorama.init()
        self._col_head = colorama.Style.BRIGHT + colorama.Fore.YELLOW
        self._col_cmd = colorama.Fore.BLUE
        self._col_clear = colorama.Style.RESET_ALL

        # Clean
        self._clean([TARGET] + OBJECTS_CUDA)


def get_readme():
    """Read readme file."""
    with open('README.rst') as f:
        return f.read()


###############################
# Setup
###############################
setup(
    ################
    # General Info
    ################
    name='libspn',
    description='The libsak joke in the world',
    long_description=get_readme(),
    setup_requires=[
        'setuptools_scm',  # Use version from SCM using setuptools_scm
        'setuptools_git >= 0.3',  # Ship files tracked by git in src dist
        'colorama',  # For color output
        # For building docs:
        'sphinx',
        'recommonmark',
        'sphinxcontrib-napoleon',
        'sphinxcontrib-websupport',
        'sphinx_rtd_theme',
        # For testing
        'flake8'
    ],
    use_scm_version=True,  # Use version from SCM using setuptools_scm
    classifiers=[
        'Development Status :: 3 - Alpha',
        # 'License :: ', TODO
        'Programming Language :: Python :: 3.4',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
    ],
    keywords=('libspn spn deep-learning deep-learning-library '
              'machine-learning machine-learning-library tensorflow'),
    url='http://www.libspn.org',
    author='Andrzej Pronobis',
    author_email='a.pronobis@gmail.com',
    license='Custom',

    ################
    # Installation
    ################
    packages=['libspn'],
    install_requires=[
        # 'tensorflow', Disabled, it overwrites a GPU installation with manulinux from pypi
        'numpy',
        'scipy',
        'matplotlib',
        'colorama'  # For color output in tests
    ],
    zip_safe=False,
    cmdclass={"build": BuildCommand,  # Custom build command for C++ code
              "clean": CleanCommand},  # Custom clean command for C++ code
    # Stuff in git repo will be included in source dist
    include_package_data=True,
    # Stuff listed below will be included in binary dist
    package_data={
        'libspn': ['ops/libspn_ops.*'],
    },

    ################
    # Tests
    ################
    test_suite='nose2.collector.collector',
    tests_require=['nose2']
)
