#!/usr/bin/env python3

from setuptools import setup
import distutils.command.build
import shutil
import sys
import os

# Ensure supported version of python is used
if sys.version_info < (3, 4):
    sys.exit('ERROR: Python < 3.4 is not supported!')


def get_readme():
    with open('README.rst') as f:
        return f.read()


def find_in_path(name, paths):
    "Find a file in paths"
    for dir in paths.split(os.pathsep):
        filepath = os.path.join(dir, name)
        if os.path.exists(filepath):
            return os.path.abspath(filepath)
    return None


class BuildCommand(distutils.command.build.build):
    """Custom build command compiling the C++ code."""

    def _configure(self):
        print("Configuring:")
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
        self._cuda_lib = os.path.join(self._cuda_home, 'lib64')

        print("- Found CUDA in %s" % self._cuda_home)
        print("  nvcc: %s" % self._cuda_nvcc)
        print("  libraries: %s" % self._cuda_lib)

        # TensorFlow
        import tensorflow
        self._tf_includes = tensorflow.sysconfig.get_include()
        self._tf_version = tensorflow.__version__
        print("- Found TensorFlow %s" % self._tf_version)
        print("  includes: %s" % self._tf_includes)

    def _build(self):
        print("Building:")
        # subprocess.Popen(["gcc", "demo.c", "-o", "demo"], cwd="./libsak/ops")

    def run(self):
        super().run()

        print("==================================================")
        self._configure()
        self._build()
        print("==================================================")


setup(
    ################
    # General Info
    ################
    name='libspn',
    description='The libsak joke in the world',
    long_description=get_readme(),
    use_scm_version=True,  # Use version from SCM using setuptools_scm
    setup_requires=['setuptools_scm'],
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
        'tensorflow',
    ],
    zip_safe=False,
    cmdclass={"build": BuildCommand},  # Custom build command for C++ code
    # Stuff in git repo will be included in source dist
    include_package_data=True,
    # Stuff listed below will be included in binary dist
    package_data={
        'libsak': ['ops/libspn_ops.*'],
    },

    ################
    # Tests
    ################
    test_suite='nose2.collector.collector',
    tests_require=['nose2']
)
